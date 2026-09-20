"""
football-data.org API client wrapper.
Features:
- Rate limiter enforcing 10 req/min (free-tier limit)
- Automatic retry with exponential backoff on HTTP 429
- Disk cache with configurable TTL
- Graceful degradation when FOOTBALL_DATA_API_KEY is not set or unauthorized
"""

import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import (
    FOOTBALL_DATA_API_KEY,
    FOOTBALL_DATA_BASE_URL,
    FOOTBALL_DATA_PL_CODE,
    FOOTBALL_DATA_RATE_LIMIT_PER_MINUTE,
    RAW_DATA_DIR,
)
from src.logging_config import get_logger
from src.data_layer.cache import DiskCache
from src.data_layer.metadata import update_metadata

logger = get_logger(__name__)

# Sample upcoming fixtures fallback for offline / key-less runs
MOCK_UPCOMING_FIXTURES = {
    "competition": {"id": 2021, "name": "Premier League", "code": "PL"},
    "matches": [
        {
            "id": 500101,
            "utcDate": "2026-09-26T14:00:00Z",
            "status": "TIMED",
            "matchday": 6,
            "homeTeam": {"id": 57, "name": "Arsenal FC", "shortName": "Arsenal"},
            "awayTeam": {"id": 61, "name": "Chelsea FC", "shortName": "Chelsea"},
        },
        {
            "id": 500102,
            "utcDate": "2026-09-26T16:30:00Z",
            "status": "TIMED",
            "matchday": 6,
            "homeTeam": {"id": 65, "name": "Manchester City FC", "shortName": "Man City"},
            "awayTeam": {"id": 66, "name": "Manchester United FC", "shortName": "Man United"},
        },
        {
            "id": 500103,
            "utcDate": "2026-09-27T15:30:00Z",
            "status": "TIMED",
            "matchday": 6,
            "homeTeam": {"id": 64, "name": "Liverpool FC", "shortName": "Liverpool"},
            "awayTeam": {"id": 73, "name": "Tottenham Hotspur FC", "shortName": "Tottenham"},
        },
    ]
}


class RateLimiter:
    """
    Token-bucket / window rate limiter for external APIs.
    """
    def __init__(self, max_calls: int = 10, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self.timestamps: List[float] = []

    def wait_if_needed(self) -> None:
        now = time.time()
        # Discard timestamps older than window
        self.timestamps = [t for t in self.timestamps if now - t < self.period]
        if len(self.timestamps) >= self.max_calls:
            oldest = self.timestamps[0]
            sleep_time = (self.period - (now - oldest)) + 0.1
            if sleep_time > 0:
                logger.info("Rate limit reached (%d/%d). Sleeping for %.2f seconds...", len(self.timestamps), self.max_calls, sleep_time)
                time.sleep(sleep_time)
        self.timestamps.append(time.time())


class FootballDataClient:
    """
    Client for football-data.org REST API.
    """
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.api_key = api_key or FOOTBALL_DATA_API_KEY
        self.base_url = FOOTBALL_DATA_BASE_URL.rstrip("/")
        self.rate_limiter = RateLimiter(max_calls=FOOTBALL_DATA_RATE_LIMIT_PER_MINUTE, period_seconds=60.0)
        cache_path = cache_dir or (RAW_DATA_DIR / "football_data_cache")
        self.cache = DiskCache(cache_dir=cache_path, default_ttl_seconds=3600)

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 5 and self.api_key != "your_api_key_here")

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        ttl_seconds: int = 3600,
        max_retries: int = 3,
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Executes an HTTP GET to football-data.org with caching and 429 retry backoff.
        Returns: (http_status_code, response_json_or_None)
        """
        cache_key = f"{endpoint}_{sorted((params or {}).items())}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache HIT for endpoint: %s", endpoint)
                return 200, cached

        if not self.has_api_key:
            logger.debug("No valid FOOTBALL_DATA_API_KEY set. Cannot make live request to %s.", endpoint)
            return 401, None

        headers = {"X-Auth-Token": self.api_key}
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        backoff = 2.0
        for attempt in range(1, max_retries + 1):
            self.rate_limiter.wait_if_needed()
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=15)
                status = resp.status_code

                if status == 200:
                    data = resp.json()
                    if use_cache:
                        self.cache.set(cache_key, data, ttl_seconds=ttl_seconds)
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    update_metadata(retrieved_at=now_str)
                    return 200, data

                elif status == 429:
                    logger.warning("HTTP 429 Rate limited. Backing off for %.2f seconds (attempt %d/%d)...", backoff, attempt, max_retries)
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue

                else:
                    logger.warning("football-data.org returned status %d for %s: %s", status, endpoint, resp.text[:200])
                    return status, None

            except requests.RequestException as e:
                logger.warning("Network error calling %s (attempt %d/%d): %s", endpoint, attempt, max_retries, e)
                if attempt == max_retries:
                    return 0, None
                time.sleep(backoff)

        return 429, None

    def probe_endpoint(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Tuple[bool, int, str]:
        """
        Probes an endpoint for access check without raising exceptions.
        Returns: (accessible: bool, status_code: int, message: str)
        """
        if not self.has_api_key:
            return False, 401, "API key not set in FOOTBALL_DATA_API_KEY"
        status, data = self._request(endpoint, params=params, use_cache=False)
        if status == 200 and data is not None:
            return True, status, "OK"
        elif status == 403:
            return False, status, "Requires paid plan (Deep Data / Tier upgrade)"
        elif status == 401:
            return False, status, "Unauthorized (invalid API key)"
        elif status == 404:
            return False, status, "Not found"
        else:
            return False, status, f"HTTP status {status}"

    def get_upcoming_fixtures(self, competition: str = FOOTBALL_DATA_PL_CODE) -> Dict[str, Any]:
        """
        Fetches upcoming scheduled fixtures for the competition.
        Falls back to mock data if offline / key is unset.
        """
        endpoint = f"competitions/{competition}/matches"
        status, data = self._request(endpoint, params={"status": "SCHEDULED"}, ttl_seconds=1800)
        if status == 200 and data:
            return data
        logger.info("Using offline/sample fixtures for upcoming matches.")
        return MOCK_UPCOMING_FIXTURES

    def get_standings(self, competition: str = FOOTBALL_DATA_PL_CODE) -> Optional[Dict[str, Any]]:
        """
        Fetches current league standings table.
        """
        endpoint = f"competitions/{competition}/standings"
        status, data = self._request(endpoint, ttl_seconds=3600)
        return data

    def get_basic_team_list(self, competition: str = FOOTBALL_DATA_PL_CODE) -> Optional[Dict[str, Any]]:
        """
        Fetches team list (names, IDs, crests) - does not request squads.
        """
        endpoint = f"competitions/{competition}/teams"
        status, data = self._request(endpoint, ttl_seconds=86400)
        return data
