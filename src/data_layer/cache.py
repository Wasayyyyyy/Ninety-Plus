"""
Disk cache with TTL support for API responses and dataset operations.
Shared across data clients.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional
from src.config import RAW_DATA_DIR
from src.logging_config import get_logger

logger = get_logger(__name__)


class DiskCache:
    """
    Simple file-based cache with time-to-live (TTL) expiration.
    """
    def __init__(self, cache_dir: Optional[Path] = None, default_ttl_seconds: int = 86400):
        self.cache_dir = cache_dir or (RAW_DATA_DIR / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl_seconds

    def _get_file_path(self, key: str) -> Path:
        safe_key = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in key)
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self._get_file_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            expires_at = payload.get("expires_at", 0)
            if expires_at and time.time() > expires_at:
                logger.debug("Cache entry for key '%s' expired.", key)
                path.unlink(missing_ok=True)
                return None
            return payload.get("data")
        except Exception as e:
            logger.warning("Failed to read cache for key '%s': %s", key, e)
            return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        path = self._get_file_path(key)
        ttl = self.default_ttl if ttl_seconds is None else ttl_seconds
        expires_at = (time.time() + ttl) if ttl > 0 else None
        payload = {
            "key": key,
            "saved_at": time.time(),
            "expires_at": expires_at,
            "data": value,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning("Failed to write cache for key '%s': %s", key, e)

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def delete(self, key: str) -> None:
        path = self._get_file_path(key)
        path.unlink(missing_ok=True)

    def clear(self) -> None:
        for p in self.cache_dir.glob("*.json"):
            p.unlink(missing_ok=True)
