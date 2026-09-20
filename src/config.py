"""
Central configuration constants for the Premier League ML System (pl-ml-system).
Single source of truth for paths, codes, thresholds, and random seeds.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Root directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory layout
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROCESSED_DATA_DIR / "figures"

# Ensure essential output directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Competition Code for Premier League
GB1 = "GB1"
PREMIER_LEAGUE_CODE = "GB1"
FOOTBALL_DATA_PL_CODE = "PL"

# Module 1: Transfer Value Predictor Thresholds
MIN_MINUTES_THRESHOLD = 450  # ~5 full 90-minute matches to reduce small-sample noise

# Module 3: Scouting Dashboard Clustering
K_MIN = 3
K_MAX = 8
K_RANGE = range(K_MIN, K_MAX + 1)

# Reproducibility Seed across numpy, sklearn, xgboost
RANDOM_SEED = 42

# football-data.org settings
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
FOOTBALL_DATA_RATE_LIMIT_PER_MINUTE = 10

# Static transfermarkt-datasets R2 Cloudflare distribution base URL
TRANSFERMARKT_R2_BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
TRANSFERMARKT_GITHUB_REPO = "dcaribou/transfermarkt-datasets"
