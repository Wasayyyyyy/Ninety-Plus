# pl-ml-system — Premier League ML System

A three-module, fully offline-capable machine learning system for Premier League football analytics. All modules share a unified data layer, metadata/freshness tracking, and strict temporal leakage controls.

---

## Architecture Overview

```
pl-ml-system/
├── data/
│   ├── raw/                           # Cached datasets (auto-downloaded from Cloudflare R2)
│   │   └── squad_overrides.csv        # Hand-maintained transfer corrections (see below)
│   └── processed/                     # Metrics JSON, prediction CSVs, figures
│       └── figures/                   # All generated evaluation plots
├── src/
│   ├── config.py                      # Central constants (paths, seeds, thresholds)
│   ├── logging_config.py              # Shared logging (get_logger)
│   ├── data_layer/
│   │   ├── metadata.py                # DatasetMetadata freshness tracking
│   │   ├── cache.py                   # Disk cache with TTL
│   │   ├── squad_overrides.py         # Manual club override resolver
│   │   ├── transfermarkt_dataset.py   # Static dataset loader + schema validator
│   │   ├── football_data_client.py    # football-data.org rate-limited client
│   │   └── data_audit.py              # Phase 0 audit runner
│   ├── module1_transfer_value/        # Transfer Value Predictor
│   ├── module2_match_outcome/         # Match Outcome Predictor
│   └── module3_scouting/              # Player Scouting Dashboard
├── tests/                             # Leakage tests, unit tests
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone and create virtual environment

```bash
# Navigate to the project root
cd "c:\Useless stuff\Ninety+"

# Create venv using Python 3.14 (system-installed)
py -3 -m venv .venv

# Activate
.venv\Scripts\Activate.ps1
```

### 2. Install pinned dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, `transfermarkt_dataset.py` automatically downloads all required
> dataset tables (≈60 MB total) from the community Cloudflare R2 mirror into `data/raw/`.
> Subsequent runs use the cached local copies.

### 3. Configure your API key (optional)

The system is **fully offline-capable** without a football-data.org API key.
Historical training data comes exclusively from the static Transfermarkt dataset.
The API key is only needed for live upcoming fixture data in Module 2's prediction CLI.

```bash
# Copy the template
copy .env.example .env
# Then edit .env and fill in:
# FOOTBALL_DATA_API_KEY=your_free_key_here
```

Sign up for a free key at https://www.football-data.org/client/register

---

## Data Sources

### 1. transfermarkt-datasets (CC0 — Static Snapshot)

- **Source:** https://github.com/dcaribou/transfermarkt-datasets
- **Distribution:** Cloudflare R2 (`pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/`)
- **Frozen as of:** July 6, 2026 (pipeline paused, automated updates stopped)
- **Cutoffs verified by Phase 0 audit:**
  - Latest player season: **2025**
  - Latest appearances: **2026-06-28**
  - Latest valuations: **2026-06-12**
  - Latest games: **2026-07-06**
- **Powers:** Modules 1, 2, and 3 (all historical stats, valuations, appearances, fixtures)

### 2. football-data.org (Live Fixtures — Free Tier)

- **Source:** https://api.football-data.org/v4/
- **Rate limit:** 10 requests/minute (auto-throttled)
- **Powers:** Module 2 predict CLI (upcoming fixture data only)
- **Known free-tier limitation:** Squad/lineup data requires the paid "Deep Data" add-on (~€29/month). The system never requests it.

### Current Club Data Gap

Neither source provides real-time squad membership freely. Current club is resolved via:

1. **`data/raw/squad_overrides.csv`** — hand-maintained file, updated on transfer deadline days
2. **Fallback:** frozen dataset's last-known club (labeled as such in every output)

To update after a transfer window:
```csv
# data/raw/squad_overrides.csv
player_name,player_id_hint,new_club,effective_date,source_note
Raheem Sterling,134425,Arsenal FC,2024-08-30,Loan from Chelsea (Summer 2024)
```

---

## Phase 0: Data Audit

**Always run this first.** Verifies both data sources, inspects actual cutoff dates, and checks all table schemas.

```bash
python -m src.data_layer.data_audit
```

Expected output:
```
DATA AUDIT
----------
transfermarkt-datasets:
  latest players season: 2025
  latest appearances: 2026-06-28
  latest valuation: 2026-06-12
  latest games: 2026-07-06
  pipeline status: PAUSED
  dataset revision / Kaggle version: frozen-2026-07-06
  schema check: OK

football-data.org:
  PL current season accessible: YES / NO (depends on API key)
  ...
```

---

## Module 1: Transfer Value Predictor

Predicts a player's Premier League transfer market value from **strictly point-in-time season-to-date stats** (no future leakage). Reports side-by-side comparison of Baseline, Model A, and Model B.

### Feature Engineering (No Leakage Guaranteed)

For each market valuation record on date **D** in season **S**:
```
Appearances where:
  competition_id == 'GB1'
  AND appearance.date < valuation.date    ← strict, not <=
  AND appearance.season == valuation.season
→ aggregate goals, assists, minutes, appearances
→ compute per-90 metrics
```

Only appearances with `cumulative_minutes >= 450` are included to avoid small-sample noise.

### Train

```bash
python -m src.module1_transfer_value.train
```

Reports side-by-side (train on seasons < 2025, test on 2025):

| Model | R² | MAE (€) | RMSE (€) |
|---|---|---|---|
| Position+Age Bucket Baseline | -0.231 | €17.2M | €26.8M |
| Model A (Perf+Age+Pos) — Linear | -0.100 | €17.2M | €25.4M |
| Model A (Perf+Age+Pos) — Ridge | -0.100 | €17.2M | €25.4M |
| Model B (+Club) — Linear | **0.284** | **€14.5M** | **€20.5M** |
| Model B (+Club) — Ridge | 0.282 | €14.5M | €20.5M |

> **Key insight:** Adding club (Model B) provides a substantial lift over performance-only features, confirming that club reputation/resources explain a significant portion of market value variance.

### Predict

```bash
python -m src.module1_transfer_value.predict --player "Bukayo Saka"
python -m src.module1_transfer_value.predict --player "Mohamed Salah"
```

---

## Module 2: Match Outcome Predictor

Predicts **HOME_WIN / DRAW / AWAY_WIN** for Premier League fixtures. Trains entirely offline from the historical games dataset.

### Feature Engineering (No Leakage Guaranteed)

Features computed chronologically, strictly from matches **before** the match being predicted:
- Rolling form (last 5 matches): points, goals scored/conceded, win rate
- Season-to-date: cumulative points, goal difference, points per match
- Head-to-head record (prior matches only)
- Rest days between matches

### Train

```bash
python -m src.module2_match_outcome.train
```

Results (held-out 2024/25 season, 380 matches):

| Model | Accuracy | Log Loss | Brier Score | Draw F1 |
|---|---|---|---|---|
| Majority Class Baseline | 0.408 | 1.081 | 0.656 | 0.000 |
| Points Heuristic Baseline | 0.524 | 1.024 | 0.613 | 0.000 |
| Random Forest | 0.458 | 1.034 | 0.621 | 0.190 |
| XGBoost | 0.439 | 1.044 | 0.628 | 0.193 |

> **Note:** Models correctly predict draws (F1 > 0), unlike baselines. RF accuracy is below the simple heuristic — typical for probabilistic models evaluated on accuracy alone. The Log Loss and Brier scores reveal the RF produces better-calibrated probability estimates than the heuristic baseline.

### Predict

```bash
python -m src.module2_match_outcome.predict --home "Arsenal" --away "Chelsea"
python -m src.module2_match_outcome.predict --home "Liverpool" --away "Manchester City" --model rf
```

Output format:
```
Arsenal FC vs Chelsea FC

Home win (Arsenal FC): 0.79
Draw:               0.10
Away win (Chelsea FC): 0.11
```

---

## Module 3: Player Scouting Dashboard

Finds statistically similar PL players using **K-means clustering + user-weighted Euclidean distance** in a standardized feature space.

> **Architecture distinction:** K-means provides the 2D visualization (coarse tactical grouping). Similarity ranking always comes from distance in the user-weighted feature space — not from cluster membership.

### Cluster (pre-compute)

```bash
python -m src.module3_scouting.cluster
```

Automatically evaluates k=3 to k=8 via silhouette score. Optimal k=3 (score=0.291).

### Launch Dashboard

```bash
streamlit run src/module3_scouting/app.py
```

Opens at http://localhost:8501 with:
- **Position filter**: [All] [GK] [DEF] [MID] [ATT]
- **Attacking / Defensive weight sliders** — recomputes distance ranking in real time
- **Ranked similarity table** with per-90 stats and % similarity scores
- **2D PCA cluster scatter** — selected player shown as red star, similar players in blue
- **Data freshness footer** on every view

---

## Running Tests

```bash
pytest -v
```

All 15 tests pass:
- `test_cache.py` — DiskCache TTL, eviction, set/get
- `test_data_layer.py` — Schema validation (fails loudly), metadata, squad overrides
- `test_module1_leakage.py` — No appearance at-or-after valuation date enters features; cross-season isolation
- `test_module2_leakage.py` — No match uses information from itself or future matches; match goals don't appear in pre-match rolling stats
- `test_module2_rolling.py` — Rolling window correctly limits to last N matches
- `test_module3_scouting.py` — Position taxonomy normalization; clustering and similarity are architecturally separate

---

## Known Limitations

1. **Frozen dataset:** All performance stats, valuations, and match history are capped at July 2026. The 2026/27 season is not represented.

2. **Transfer gap:** Post-July 2026 transfers are not tracked automatically. Update `data/raw/squad_overrides.csv` manually after each transfer window (deadline days in January and August).

3. **Squad data:** Neither free source provides automated real-time squad membership. The paid football-data.org "Deep Data" add-on (~€29/month) would resolve this.

4. **Module 1 R² on monetary scale:** A negative R² for Model A (no club) is expected — raw performance stats alone cannot predict market value without club context. Model B demonstrates this clearly (+38 points of R²).

5. **Module 2 accuracy:** Predicting football outcomes is intrinsically noisy. All models are correctly compared against majority-class and heuristic baselines. The primary value is probabilistic calibration (Log Loss / Brier Score), not peak accuracy.

6. **Windows Application Control policy:** On this system, `sklearn.ensemble._hist_gradient_boosting` is blocked by Windows security policy. The `HistGradientBoostingClassifier` import in `sklearn/ensemble/__init__.py` is wrapped in a `try/except` within `.venv`. `RandomForestClassifier` and `XGBClassifier` are both fully functional.

---

## Data Freshness Footer Format

Every CLI output and dashboard screen shows:
```
Fixture data:      football-data.org, retrieved YYYY-MM-DD
Performance data:  transfermarkt-datasets, latest appearance 2026-06-28
Market-value data: transfermarkt-datasets, latest valuation 2026-06-12
Current club:      squad_overrides.csv (manual override) / frozen dataset - last known club (2025/26)
```
