<div align="center">

# ⚽ MLS 2026 Predictor

An end-to-end Machine Learning, Explainable AI (XAI), and stochastic Monte Carlo simulation platform engineered specifically for the structural and operational mechanics of **Major League Soccer (MLS)**.

<br>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-blueviolet)](https://shap.readthedocs.io)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-3DA639?logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/armando-mio/MLS-2026-Predictor)](https://github.com/armando-mio/MLS-2026-Predictor/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/armando-mio/MLS-2026-Predictor)](https://github.com/armando-mio/MLS-2026-Predictor)

</div>

---

## 📌 Overview

Predicting soccer outcomes in Major League Soccer presents distinct challenges that standard European football models fail to capture: extreme travel distances across North America, artificial turf venues, high-altitude stadia, designated player (DP) quality disparities, schedule congestion during FIFA international windows, and a complex postseason format (Wild Card, Best-of-3 Round One, and single-elimination knockout brackets).

**MLS 2026 Predictor** addresses these domain-specific factors through leak-free feature engineering, calibrated machine learning classifiers, local SHAP explainability, interactive scenario simulations, and a 10,000-iteration Monte Carlo engine enforcing official MLS standings tiebreakers and playoff regulations.

---

## 🎯 Key Capabilities & Dashboard Features

The application is deployed as a high-performance, single-page Streamlit web app structured into **5 interactive modules** alongside an integrated **"How MLS Works"** rulebook:

1. **Match Predictor**
   - Head-to-head matchup simulation generating calibrated **1X2 probabilities** (Home / Draw / Away), **Over/Under 2.5 goals**, and **Both Teams to Score (GG/NG)**.
   - Bivariate Poisson exact-score matrix normalized to 100.0% probability.
   - Historical head-to-head records, recent form metrics, and tactical radar comparisons.

2. **Explainability (SHAP)**
   - Local feature attribution via TreeExplainer waterfall plots for every single predicted match.
   - Transparent breakdown of which factors shifted the outcome toward a home win, draw, or away victory (e.g., Elo differential, travel fatigue index, turf mismatch, rest days).

3. **What-If Scenario Simulator**
   - Real-time parameter tweaking to test hypothetical match conditions: adjust travel fatigue, change venue surface (grass vs. artificial turf), simulate FIFA window player absenteeism, adjust rest days, or modify designated player availability.

4. **Model Performance & Calibration**
   - Walk-forward backtest evaluation on test season 2025 (trained on 2022–2024).
   - Comparison across 6 classification architectures (*Random Forest, Extra Trees, XGBoost, LightGBM, HistGradientBoosting, Logistic Regression*).
   - Reliability curves (calibration), Log-Loss, Brier score, and accuracy metrics.

5. **Season Rankings & Monte Carlo Playoff Engine**
   - 10,000 stochastic season simulations calculating playoff qualification probabilities, Supporters' Shield contention odds, and final conference standings distributions.
   - Complete tournament bracket simulation adhering to official MLS rules: Wild Card (8 vs. 9), Round One (Best-of-3 series), Conference Semifinals/Finals, and the MLS Cup Final.

6. **"How MLS Works" Reference Guide**
   - Multipage interactive reference detailing MLS conference structures, roster mechanisms (Salary Cap, DPs, TAM/GAM), tiebreaker hierarchies, and playoff formats.

---

## 🧠 Architectural Decisions & Engineering Rationale

| Design Decision | Implementation | Technical & Domain Rationale |
| :--- | :--- | :--- |
| **Temporal Walk-Forward Split** | Train: 2022–2024<br>Test: 2025<br>Predict: 2026 | In time-series sports data, standard k-fold cross-validation causes lookahead leakage. A strict temporal walk-forward split simulates real-world deployment where future matches are strictly unobserved during training. |
| **Pre-2022 Data Exclusion** | Filtered dataset from `Season >= 2022` | Structural shifts in MLS (expansion teams like Charlotte, St. Louis, San Diego; post-COVID schedule formats; roster rule modifications) render older seasons unrepresentative of modern league parity. |
| **Domain-Specific Feature Engineering** | TFI, Timezones, Turf, Altitude, DPs, FIFA windows | MLS teams travel thousands of kilometers across up to 4 time zones. Features like the **Travel Fatigue Index (TFI)**, altitude thresholds (Denver/Salt Lake City > 1,200m), artificial turf mismatches, and Designated Player quality differentials capture critical variance standard models miss. |
| **Strict Data Leakage Prevention** | Betting odds excluded from `production` feature set | Pinnacle and market closing odds are isolated into an `odds_augmented` set solely for benchmarking. Production models use only pre-match signals known before kickoff. |
| **Rest Days Capping** | Capped at 14.0 days maximum | Uncapped rest day calculations distort team form metrics across off-season transitions and multi-week international breaks. |
| **Dynamic Elo System** | $K=32$, Goal-Diff multiplier, 33% season regression | Standard Elo ignores victory margins. The implemented system uses goal-difference multipliers and a 33% regression-to-the-mean at season boundaries to account for off-season transfers while preserving team continuity. |
| **Regularized Random Forest Baseline** | `max_depth=6`, `min_samples_leaf=8`, `class_weight='balanced'` | Soccer outcome prediction has high intrinsic noise. Deep trees overfit quickly. Pruned and regularized trees yield well-calibrated class probabilities with robust generalization. |
| **Official MLS Playoff & Tiebreaker Rules** | Points $\to$ Wins $\to$ GD $\to$ GF $\to$ Best-of-3 Round 1 | Standard European tiebreakers (Goal Difference first) are invalid in MLS. The Monte Carlo engine strictly enforces MLS tiebreakers (Total Wins first) and simulates the Best-of-3 Round 1 format where the higher seed hosts Games 1 & 3. |
| **Zero-Latency Streamlit UI** | Tokenized CSS, `@st.cache_data`, precalculated lookups | Custom CSS tokens and module-level memoization eliminate redundant DataFrame computations, enabling instant tab switching and smooth slider interactions. |

---

## 📁 Repository Structure

```plaintext
MLS-2026-Predictor/
├── .devcontainer/              # Dev container configuration
├── .streamlit/                 # Streamlit configuration & theme settings
├── data/
│   ├── lookups/                # JSON lookups: stadiums, DPs, FIFA windows, rivalries
│   ├── processed/              # Precomputed features & Elo parquets
│   └── USA.csv                 # MLS historical match records (football-data.co.uk)
├── mls_predictor/              # Core library package
│   ├── config.py               # Constants, team mappings, paths, and conference rosters
│   ├── data_loader.py          # Data ingestion, geographic Haversine, lookup loaders
│   ├── elo.py                  # Elo rating calculator with season regression
│   ├── feature_descriptions.py # UI metadata & human-readable feature descriptions
│   ├── feature_engine.py       # Pipeline: geography, rest, roster, rolling stats, odds
│   ├── model_utils.py          # Training, imputation, temporal split, evaluation
│   ├── monte_carlo.py          # Season & playoff bracket simulation engine
│   └── shap_utils.py           # SHAP TreeExplainer & feature attribution helpers
├── models/                     # Saved joblib models and preprocessing transformers
├── notebooks/                  # Step-by-step exploratory data analysis & experiments
├── streamlit_app/              # Web application
│   ├── assets/                 # Custom stylesheet (style.css)
│   ├── pages/                  # Multipage apps (How_MLS_Works.py)
│   └── app.py                  # Main 5-tab dashboard
├── main.py                     # CLI entrypoint for training and running the app
├── requirements.txt            # Project dependencies
├── test_pipeline.py            # Quick feature pipeline validation script
└── verify_all_domains.py       # 5-Domain architectural test & audit suite
```

---

## ⚠️ Limitations & Methodological Nuances

Transparency regarding model constraints and the probabilistic boundaries of sports analytics is essential:

- **Parity & High Inherent Stochasticity:** Unlike top-heavy European leagues where a small financial elite dominates, MLS enforces strict parity through Salary Caps, single-entity ownership, and Allocation Money (GAM/TAM). Consequently, single-match predictability faces an inherent entropy ceiling. A 3-way (1X2) accuracy of ~44–48% with a Log-Loss of ~1.05 is well-calibrated and outperforms random baselines (~33%), but individual match outcomes remain susceptible to low-scoring variance (deflections, red cards, weather extremes).
- **Roster Granularity vs. Real-Time Lineups:** Designated Player (DP) quality differentials and international absences are modeled via curated seasonal registries and FIFA international calendar windows. The model does not ingest live, 60-minute pre-match official team sheets or late warm-up injury scratchings.
- **Expansion Team Cold-Start:** When inaugural expansion franchises enter the league (e.g., *St. Louis City SC in 2023, San Diego FC in 2025/2026*), rolling feature windows (3, 5, 10 matches) suffer from initial data sparsity during early matchweeks. While compensated using a 1500 baseline Elo prior, franchise expansion flags, and median imputation, early-season predictions carry wider uncertainty bounds.
- **Event-Level Tracking Proxies:** Data is ingested from historical league records and closing market distributions (`football-data.co.uk`). Advanced metrics (xG efficiency, travel fatigue, rest congestion) are calculated as rolling proxies rather than raw spatial coordinate tracking streams (e.g., Opta / Second Spectrum).
- **Postseason Knockout Volatility:** While the 10,000-iteration Monte Carlo engine faithfully replicates official MLS playoff formats (Best-of-3 Round 1, higher-seed hosting, single elimination), knockout tournament soccer involves high sudden-death variance and penalty shootout randomness.
- **Research & Non-Commercial Disclaimer:** This platform is designed exclusively for quantitative research, academic benchmarking, and sports analytics exploration. Probabilities and simulations do not constitute betting or financial advice.