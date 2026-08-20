"""
MLS 2026 Predictor: Single-Page Streamlit Dashboard
5 Tabs: Match Predictor · Explainability · What-If Simulator · Model Performance · Season Rankings

Run with: streamlit run streamlit_app/app.py
"""
import streamlit as st
import sys
from pathlib import Path

# ── Project root on path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Page Config (must be first Streamlit call) ──
st.set_page_config(
    page_title="MLS 2026 Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Lucide Icons via CDN ──
st.markdown("""
<link href="https://unpkg.com/lucide-static@latest/font/lucide.css" rel="stylesheet">
<style>
    .lucide { font-family: 'lucide' !important; font-style: normal; font-weight: normal; }
    .icon-sm { font-size: 1rem; vertical-align: middle; }
    .icon-md { font-size: 1.2rem; vertical-align: middle; }
    .icon-lg { font-size: 1.5rem; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson

import importlib
import mls_predictor.config
importlib.reload(mls_predictor.config)
import mls_predictor.feature_engine
importlib.reload(mls_predictor.feature_engine)
import mls_predictor.monte_carlo
importlib.reload(mls_predictor.monte_carlo)
import mls_predictor.model_utils
importlib.reload(mls_predictor.model_utils)

from mls_predictor.config import (
    EASTERN_CONFERENCE, WESTERN_CONFERENCE,
    MLS_EXPANSION_SEASONS, standardize_team_name,
)


# ═══════════════════════════════════════════════════════════════════════════
#  TEAM LOGOS (ESPN CDN)
# ═══════════════════════════════════════════════════════════════════════════

TEAM_ESPN_IDS = {
    "Atlanta Utd": 18418,
    "Austin FC": 20906,
    "CF Montreal": 9720,
    "Charlotte": 21300,
    "Chicago Fire": 182,
    "Colorado Rapids": 184,
    "Columbus Crew": 183,
    "DC United": 193,
    "FC Cincinnati": 18267,
    "FC Dallas": 185,
    "Houston Dynamo": 6077,
    "Inter Miami": 20232,
    "Los Angeles FC": 18966,
    "Los Angeles Galaxy": 187,
    "Minnesota United": 17362,
    "Nashville SC": 18986,
    "New England Revolution": 189,
    "New York City": 17606,
    "New York Red Bulls": 190,
    "Orlando City": 12011,
    "Philadelphia Union": 10739,
    "Portland Timbers": 9723,
    "Real Salt Lake": 4771,
    "San Diego FC": 22529,
    "San Jose Earthquakes": 191,
    "Seattle Sounders": 9726,
    "Sporting Kansas City": 186,
    "St. Louis City": 21812,
    "Toronto FC": 7318,
    "Vancouver Whitecaps": 9727,
}


def get_team_logo_url(team_name: str, size: int = 500) -> str:
    """Get the ESPN CDN dark-mode logo URL for a team."""
    espn_id = TEAM_ESPN_IDS.get(team_name)
    if espn_id:
        return f"https://a.espncdn.com/i/teamlogos/soccer/500-dark/{espn_id}.png"
    return ""


def team_logo_html(team_name: str, size: int = 32) -> str:
    """Generate an <img> tag for the team logo."""
    url = get_team_logo_url(team_name)
    if url:
        return f'<img src="{url}" width="{size}" height="{size}" style="vertical-align: middle; border-radius: 4px;" alt="{team_name}">'
    return ""


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING & MODEL TRAINING (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Load raw data (2022-2025), compute Elo, build features."""
    from mls_predictor.data_loader import load_raw_data
    from mls_predictor.elo import compute_elo_history, get_current_elo
    from mls_predictor.feature_engine import build_all_features

    df = load_raw_data(exclude_current_season=True)  # Only 2022-2025
    df = compute_elo_history(df)
    df_feat = build_all_features(df)
    current_elo = get_current_elo(df)
    return df, df_feat, current_elo


@st.cache_resource(show_spinner=False)
def train_cached_model():
    """Train the RandomForest model (cached as resource: survives reruns)."""
    from mls_predictor.feature_engine import get_feature_columns
    from mls_predictor.model_utils import (
        temporal_train_test_split, prepare_features,
        train_model, evaluate_model,
    )

    df, df_feat, current_elo = load_data()
    feature_cols = get_feature_columns()["all"]

    split = temporal_train_test_split(df_feat, "target_1x2", feature_cols)
    X_train_imp, X_test_imp, imputer, scaler = prepare_features(
        split["X_train"], split["X_test"]
    )

    model = train_model(X_train_imp, split["y_train"].values, "1x2")
    metrics = evaluate_model(model, X_test_imp, split["y_test"].values, "1x2")

    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "feature_cols": split["feature_cols"],
        "metrics": metrics,
        "X_test": X_test_imp,
        "y_test": split["y_test"].values,
    }


@st.cache_data(ttl=3600, show_spinner="Building season schedule & running simulation…")
def run_full_season_pipeline(_df_feat, _current_elo, _model_artifacts):
    """Build a balanced 34-game schedule and run Monte Carlo simulation."""
    from mls_predictor.config import EASTERN_CONFERENCE, WESTERN_CONFERENCE
    from mls_predictor.monte_carlo import simulate_season_detailed

    all_conf_teams = EASTERN_CONFERENCE + WESTERN_CONFERENCE

    # Zero-based standings: every team starts fresh
    standings = {
        team: {"points": 0, "gd": 0, "gf": 0, "ga": 0,
               "played": 0, "w": 0, "d": 0, "l": 0}
        for team in all_conf_teams
    }

    def _build_match(home_t, away_t):
        fv = build_feature_vector(
            home_t, away_t, _df_feat, _current_elo,
            _model_artifacts["feature_cols"]
        )
        if fv is None:
            return None
        p = predict_match(fv, _model_artifacts)
        # Poisson lambdas: attack + defense + Elo + home advantage
        _h_atk = fv.get("home_attack_r5", 1.3)
        _a_def = fv.get("away_defense_r5", 1.3)
        _a_atk = fv.get("away_attack_r5", 1.0)
        _h_def = fv.get("home_defense_r5", 1.3)
        _h_elo = fv.get("home_elo_before", 1500)
        _a_elo = fv.get("away_elo_before", 1500)
        _elo_r = _h_elo / max(_a_elo, 1)
        _ha = 1.10
        return {
            "home": home_t, "away": away_t,
            "prob_h": p[2], "prob_d": p[1], "prob_a": p[0],
            "lambda_h": max(0.3, (_h_atk + _a_def) / 2 * _ha * (_elo_r ** 0.15)),
            "lambda_a": max(0.3, (_a_atk + _h_def) / 2 / _ha * ((1 / _elo_r) ** 0.15)),
        }

    matches = []

    # 1) Intra-conference: home & away vs each of 14 rivals = 28 games
    for conf in [EASTERN_CONFERENCE, WESTERN_CONFERENCE]:
        for home_t in conf:
            for away_t in conf:
                if home_t != away_t:
                    m = _build_match(home_t, away_t)
                    if m:
                        matches.append(m)

    # 2) Cross-conference: 3 home + 3 away = 6 games per team
    n_cross = 3
    cross_pairs_added = set()

    for conf_own, conf_opp in [
        (EASTERN_CONFERENCE, WESTERN_CONFERENCE),
        (WESTERN_CONFERENCE, EASTERN_CONFERENCE),
    ]:
        for i, team in enumerate(conf_own):
            home_opps = [
                conf_opp[(i * n_cross + j) % len(conf_opp)]
                for j in range(n_cross)
            ]
            for opp in home_opps:
                pair = (team, opp)
                if pair not in cross_pairs_added:
                    cross_pairs_added.add(pair)
                    m = _build_match(team, opp)
                    if m:
                        matches.append(m)

    # Run Monte Carlo simulation (1000 sims for UI speed)
    summary_df, position_dist_df = simulate_season_detailed(
        standings, matches, 1000
    )
    return summary_df, position_dist_df


@st.cache_data(ttl=1800, show_spinner=False)
def run_playoff_sim(conf_name, seeds, _elo, n_sims):
    """Cached playoff bracket simulation."""
    from mls_predictor.monte_carlo import simulate_playoff_bracket
    return simulate_playoff_bracket(seeds, _elo, n_sims)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e8edf2"),
    hoverlabel=dict(
        bgcolor="#141e2b",
        bordercolor="#2a3545",
        font=dict(family="Inter", size=13, color="#ffffff"),
    ),
)

COLORS = {
    "home": "#27ae60",
    "draw": "#f1c40f",
    "away": "#e74c3c",
    "blue": "#1D428A",
    "red": "#C8102E",
}


def norm(val, min_v, max_v):
    """Normalize a value to 0-100 range."""
    return max(0, min(100, (val - min_v) / max(max_v - min_v, 1) * 100))


def compute_team_unbeaten_streak(df_or_feat, team):
    """Compute the current unbroken unbeaten (W or D) streak for a team."""
    matches = df_or_feat[((df_or_feat["HomeTeam"] == team) | (df_or_feat["AwayTeam"] == team))]
    if len(matches) == 0:
        return 0
    streak = 0
    for i in range(len(matches) - 1, -1, -1):
        row = matches.iloc[i]
        is_home = row["HomeTeam"] == team
        won = (is_home and row["FTR"] == "H") or (not is_home and row["FTR"] == "A")
        drew = (row["FTR"] == "D")
        if won or drew:
            streak += 1
        else:
            break
    return streak


def build_feature_vector(
    home_team: str,
    away_team: str,
    df_feat: pd.DataFrame,
    current_elo: dict,
    feature_cols: list[str],
    rest_override_h: int | None = None,
    rest_override_a: int | None = None,
    fatigue_mult: float = 1.0,
) -> dict | None:
    from mls_predictor.config import standardize_team_name
    from mls_predictor.data_loader import load_stadiums, load_rivalries, haversine_km

    home_team = standardize_team_name(home_team)
    away_team = standardize_team_name(away_team)

    # Historical slices for individual team form
    home_history = df_feat[(df_feat["HomeTeam"] == home_team) | (df_feat["AwayTeam"] == home_team)]
    away_history = df_feat[(df_feat["HomeTeam"] == away_team) | (df_feat["AwayTeam"] == away_team)]

    if len(home_history) == 0 or len(away_history) == 0:
        return None

    last_home_match = home_history.iloc[-1]
    last_away_match = away_history.iloc[-1]

    feature_vector = {}
    stadiums = load_stadiums()
    h_info = stadiums.get(home_team, {})
    a_info = stadiums.get(away_team, {})

    # 1. Geographic & Logistics (Exact Home vs Away Stadiums)
    if h_info and a_info:
        feature_vector["travel_distance_km"] = haversine_km(
            a_info["lat"], a_info["lon"], h_info["lat"], h_info["lon"]
        )
        tz_offset_map = {
            "America/New_York": -5, "America/Toronto": -5, "America/Chicago": -6,
            "America/Denver": -7, "America/Los_Angeles": -8, "America/Vancouver": -8,
        }
        h_tz = tz_offset_map.get(h_info.get("timezone", ""), -5)
        a_tz = tz_offset_map.get(a_info.get("timezone", ""), -5)
        feature_vector["timezones_crossed"] = abs(h_tz - a_tz)
        feature_vector["venue_is_turf"] = 1 if h_info.get("surface") == "turf" else 0
        feature_vector["surface_mismatch"] = 1 if h_info.get("surface") != a_info.get("surface") else 0
        feature_vector["venue_altitude_m"] = h_info.get("altitude_m", 0)
        feature_vector["high_altitude"] = 1 if h_info.get("altitude_m", 0) >= 1200 else 0
    else:
        feature_vector["travel_distance_km"] = 0.0
        feature_vector["timezones_crossed"] = 0
        feature_vector["venue_is_turf"] = 0
        feature_vector["surface_mismatch"] = 0
        feature_vector["venue_altitude_m"] = 0
        feature_vector["high_altitude"] = 0

    # 2. Dynamic Rivalry Matchup
    rivalries = load_rivalries()
    rivalry_key = tuple(sorted([home_team, away_team]))
    rivalry_dict = {tuple(sorted(r["teams"])): r["intensity"] for r in rivalries}
    feature_vector["is_rivalry"] = 1 if rivalry_key in rivalry_dict else 0
    feature_vector["rivalry_intensity"] = rivalry_dict.get(rivalry_key, 0)

    # 3. Dynamic Head-to-Head (Last 5 meetings between THESE two teams)
    h2h_matches = df_feat[
        ((df_feat["HomeTeam"] == home_team) & (df_feat["AwayTeam"] == away_team)) |
        ((df_feat["HomeTeam"] == away_team) & (df_feat["AwayTeam"] == home_team))
    ].tail(5)

    hw, dw, aw = 0, 0, 0
    for _, row in h2h_matches.iterrows():
        if row["FTR"] == "H":
            if row["HomeTeam"] == home_team: hw += 1
            else: aw += 1
        elif row["FTR"] == "A":
            if row["AwayTeam"] == home_team: hw += 1
            else: aw += 1
        else:
            dw += 1
    feature_vector["h2h_home_wins"] = float(hw)
    feature_vector["h2h_draws"] = float(dw)
    feature_vector["h2h_away_wins"] = float(aw)

    # 4. Rolling Attack & Defense Proxies (Extracted from each team's own timeline)
    for w in [3, 5, 10]:
        h_atk = last_home_match.get(f"home_attack_r{w}", 1.3) if last_home_match["HomeTeam"] == home_team else last_home_match.get(f"away_attack_r{w}", 1.3)
        h_def = last_home_match.get(f"home_defense_r{w}", 1.3) if last_home_match["HomeTeam"] == home_team else last_home_match.get(f"away_defense_r{w}", 1.3)
        a_atk = last_away_match.get(f"away_attack_r{w}", 1.1) if last_away_match["AwayTeam"] == away_team else last_away_match.get(f"home_attack_r{w}", 1.1)
        a_def = last_away_match.get(f"away_defense_r{w}", 1.3) if last_away_match["AwayTeam"] == away_team else last_away_match.get(f"home_defense_r{w}", 1.3)

        feature_vector[f"home_attack_r{w}"] = float(h_atk)
        feature_vector[f"home_defense_r{w}"] = float(h_def)
        feature_vector[f"away_attack_r{w}"] = float(a_atk)
        feature_vector[f"away_defense_r{w}"] = float(a_def)
        feature_vector[f"home_total_r{w}"] = feature_vector[f"home_attack_r{w}"] + feature_vector[f"home_defense_r{w}"]
        feature_vector[f"away_total_r{w}"] = feature_vector[f"away_attack_r{w}"] + feature_vector[f"away_defense_r{w}"]
        feature_vector[f"attack_diff_r{w}"] = feature_vector[f"home_attack_r{w}"] - feature_vector[f"away_attack_r{w}"]
        feature_vector[f"defense_diff_r{w}"] = feature_vector[f"home_defense_r{w}"] - feature_vector[f"away_defense_r{w}"]

        h_bts = last_home_match.get(f"home_bts_r{w}", 0.5) if last_home_match["HomeTeam"] == home_team else last_home_match.get(f"away_bts_r{w}", 0.5)
        a_bts = last_away_match.get(f"away_bts_r{w}", 0.5) if last_away_match["AwayTeam"] == away_team else last_away_match.get(f"home_bts_r{w}", 0.5)
        feature_vector[f"home_bts_r{w}"] = float(h_bts if pd.notna(h_bts) else 0.5)
        feature_vector[f"away_bts_r{w}"] = float(a_bts if pd.notna(a_bts) else 0.5)

    # 5. Dynamic Elo & Differentials
    feature_vector["home_elo_before"] = float(current_elo.get(home_team, 1500))
    feature_vector["away_elo_before"] = float(current_elo.get(away_team, 1500))
    feature_vector["elo_diff"] = feature_vector["home_elo_before"] - feature_vector["away_elo_before"]

    # 6. Rest Days, Congestion & Overrides
    h_rest = last_home_match.get("home_rest_days", 7.0) if last_home_match["HomeTeam"] == home_team else last_home_match.get("away_rest_days", 7.0)
    a_rest = last_away_match.get("away_rest_days", 7.0) if last_away_match["AwayTeam"] == away_team else last_away_match.get("home_rest_days", 7.0)
    feature_vector["home_rest_days"] = float(rest_override_h if rest_override_h is not None else (h_rest if pd.notna(h_rest) else 7.0))
    feature_vector["away_rest_days"] = float(rest_override_a if rest_override_a is not None else (a_rest if pd.notna(a_rest) else 7.0))
    feature_vector["rest_diff"] = feature_vector["home_rest_days"] - feature_vector["away_rest_days"]

    h_cong = last_home_match.get("home_congestion", 1.0) if last_home_match["HomeTeam"] == home_team else last_home_match.get("away_congestion", 1.0)
    a_cong = last_away_match.get("away_congestion", 1.0) if last_away_match["AwayTeam"] == away_team else last_away_match.get("home_congestion", 1.0)
    feature_vector["home_congestion"] = float(h_cong if pd.notna(h_cong) else 1.0)
    feature_vector["away_congestion"] = float(a_cong if pd.notna(a_cong) else 1.0)

    # 7. Designated Players (Roster Quality)
    from mls_predictor.data_loader import load_designated_players
    dp_data = load_designated_players()
    season_data = dp_data.get("2026", dp_data.get("2025", {}))
    h_players = season_data.get(home_team, [])
    a_players = season_data.get(away_team, [])
    h_q = np.mean([p["quality"] for p in h_players]) if h_players else 6.0
    a_q = np.mean([p["quality"] for p in a_players]) if a_players else 6.0
    feature_vector["home_dp_quality"] = float(h_q)
    feature_vector["away_dp_quality"] = float(a_q)
    feature_vector["dp_quality_diff"] = float(h_q - a_q)

    # 8. FIFA International Windows
    feature_vector["during_fifa_window"] = 0.0
    feature_vector["days_since_fifa_window"] = 30.0
    feature_vector["near_fifa_window"] = 0.0

    # 9. Unbeaten Streaks & Form
    feature_vector["home_unbeaten_streak"] = float(compute_team_unbeaten_streak(df_feat, home_team))
    feature_vector["away_unbeaten_streak"] = float(compute_team_unbeaten_streak(df_feat, away_team))

    h_ppg = last_home_match.get("home_season_ppg", 1.4) if last_home_match["HomeTeam"] == home_team else last_home_match.get("away_season_ppg", 1.4)
    a_ppg = last_away_match.get("away_season_ppg", 1.2) if last_away_match["AwayTeam"] == away_team else last_away_match.get("home_season_ppg", 1.2)
    feature_vector["home_season_ppg"] = float(h_ppg if pd.notna(h_ppg) else 1.4)
    feature_vector["away_season_ppg"] = float(a_ppg if pd.notna(a_ppg) else 1.2)
    feature_vector["ppg_diff"] = feature_vector["home_season_ppg"] - feature_vector["away_season_ppg"]

    h_games = df_feat[df_feat["HomeTeam"] == home_team]
    h_wins = h_games[h_games["FTR"] == "H"]
    h_hwr = len(h_wins) / max(len(h_games), 1) if len(h_games) > 0 else 0.50

    a_games = df_feat[df_feat["AwayTeam"] == away_team]
    a_wins = a_games[a_games["FTR"] == "A"]
    a_awr = len(a_wins) / max(len(a_games), 1) if len(a_games) > 0 else 0.25

    feature_vector["home_home_win_rate"] = float(h_hwr)
    feature_vector["away_away_win_rate"] = float(a_awr)

    # 10. Travel Fatigue Index
    tfi = last_away_match.get("travel_fatigue_index", 1500.0)
    feature_vector["travel_fatigue_index"] = float(tfi if pd.notna(tfi) else 1500.0) * fatigue_mult

    # 11. Expansion Team Status
    feature_vector["home_expansion"] = 1.0 if (2026 - MLS_EXPANSION_SEASONS.get(home_team, 1996)) < 2 else 0.0
    feature_vector["away_expansion"] = 1.0 if (2026 - MLS_EXPANSION_SEASONS.get(away_team, 1996)) < 2 else 0.0

    # Ensure all required features are populated
    for col in feature_cols:
        if col not in feature_vector:
            feature_vector[col] = 0.0

    return feature_vector


def predict_match(feature_vector, model_artifacts):
    """Run prediction and return probabilities [away, draw, home]."""
    model = model_artifacts["model"]
    imputer = model_artifacts["imputer"]
    feature_cols = model_artifacts["feature_cols"]

    X_pred = pd.DataFrame([{c: feature_vector.get(c, 0) for c in feature_cols}])
    X_imp = pd.DataFrame(
        imputer.transform(X_pred), columns=feature_cols, index=X_pred.index
    )
    probs = model.predict_proba(X_imp)[0]
    return probs  # [P(Away), P(Draw), P(Home)]


def get_recent_form(df, team, n=5):
    """Get last n results for a team as list of (date, opponent, result_char, goals_for, goals_against)."""
    matches = df[((df["HomeTeam"] == team) | (df["AwayTeam"] == team))].tail(n)
    form = []
    for _, row in matches.iterrows():
        if row["HomeTeam"] == team:
            opponent = row["AwayTeam"]
            gf, ga = int(row["FTHG"]), int(row["FTAG"])
            if row["FTR"] == "H":
                result = "W"
            elif row["FTR"] == "D":
                result = "D"
            else:
                result = "L"
        else:
            opponent = row["HomeTeam"]
            gf, ga = int(row["FTAG"]), int(row["FTHG"])
            if row["FTR"] == "A":
                result = "W"
            elif row["FTR"] == "D":
                result = "D"
            else:
                result = "L"
        form.append((row["Date"], opponent, result, gf, ga))
    return form


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR: Walkthrough / Guide
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render the sidebar with walkthrough content."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 0.1rem 0 0.35rem 0;">
            <div style="font-size: 1.4rem; font-weight: 800; color: #f8fafc;">
                MLS 2026 Predictor
            </div>
            <div style="color: #cbd5e1; font-size: 0.82rem; margin-top: 0.15rem; font-weight: 500;">
                Machine Learning Match Prediction
            </div>
        </div>
        """, unsafe_allow_html=True)

        nav_page = st.segmented_control(
            "Page Navigation",
            options=["Dashboard", "How MLS Works"],
            default="Dashboard",
            label_visibility="collapsed",
            key="sidebar_nav_toggle_app"
        )
        if nav_page and nav_page != "Dashboard":
            st.switch_page("pages/How_MLS_Works.py")

        # About
        with st.expander("About this tool", expanded=True):
            st.markdown("""
            An AI-powered prediction system for MLS 2026 season matches. 
            Uses a **RandomForest** classifier trained on historical data 
            combined with **Elo ratings** to estimate match outcomes.
            """)

        # Data Sources
        with st.expander("Data Sources"):
            st.markdown("""
            **Training Data**  
            - football-data.co.uk (2022–2025)  
            - 1,500+ MLS regular season matches  
            - Full-time results, goals, odds  

            **Features Used**  
            - Elo ratings (dynamic, per-match)  
            - Rolling attack/defense (3, 5, 10 games)  
            - Home/away form & win rates  
            - Travel distance & fatigue index  
            - Rest days between matches  
            - Head-to-head record  
            - Rivalry flags  
            - Surface type, altitude, timezone  
            """)

        # How It Works
        with st.expander("How It Works"):
            st.markdown("""
            **Step 1: Data Collection**  
            Raw match data is loaded and cleaned from 
            football-data.co.uk for seasons 2022–2025.

            **Step 2: Elo Ratings**  
            A dynamic Elo system computes per-team 
            strength ratings updated after each match.

            **Step 3: Feature Engineering**  
            40+ features computed: rolling stats, venue effects, 
            travel fatigue, form, head-to-head, and more.

            **Step 4: Model Training**  
            A RandomForest classifier is trained on 2022–2024 
            data and evaluated on the 2025 season.

            **Step 5: Prediction**  
            For any home/away pair, a feature vector is built 
            and the model outputs probabilities for Home Win, 
            Draw, and Away Win.
            """)

        # Feature Glossary
        with st.expander("Feature Glossary"):
            st.markdown("""
            | Feature | Description |
            |---------|-------------|
            | **Elo** | Strength rating (1500 baseline) |
            | **Attack R5** | Avg goals scored (last 5) |
            | **Defense R5** | Avg goals conceded (last 5) |
            | **PPG** | Points per game (season) |
            | **Travel km** | Away team travel distance |
            | **Rest days** | Days since last match |
            | **H2H** | Head-to-head record (last 5) |
            | **Rivalry** | Derby / rivalry match flag |
            """)

        # Tab Guide
        with st.expander("Tab Guide"):
            st.markdown("""
            **Match Predictor**: Select two teams and see 
            win/draw/loss probabilities with team comparison.

            **Explainability**: SHAP analysis showing which 
            features drive each prediction.

            **What-If Simulator**: Adjust rest days, star 
            players, and fatigue to see how predictions change.

            **Model Performance**: Accuracy, confusion matrix, 
            calibration, and global feature importance.

            **Season Rankings**: Full Monte Carlo simulation 
            of the 2026 season with conference standings and 
            playoff probabilities.
            """)

        st.markdown("""
        <div class="model-simple-badge">
            <div class="model-simple-title">
                <span class="model-simple-pill">v4.0</span>
                <span>RandomForest Model</span>
            </div>
            <div class="model-simple-sub">
                Train: 2022–2024 · Test: 2025 · <strong style="color: #38bdf8;">2026 Season Prediction</strong>
            </div>
        </div>

        <div class="creator-card">
            <div class="created-by-label">Created by:</div>
            <a href="https://www.linkedin.com/in/armando-mio" target="_blank" class="linkedin-btn">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
                </svg>
                <span>Armando Mio</span>
            </a>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Sidebar Walkthrough ──
    render_sidebar()

    # ── Load data ──
    with st.spinner("Loading data…"):
        df, df_feat, current_elo = load_data()

    teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))

    # ── Header ──
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 0.3rem 0;">
        <h1 style="font-size: 2rem; font-weight: 800; color: #e8edf2; margin-bottom: 0;">
        MLS 2026 Predictor
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # ── Team Selector (in main area, below title) ──
    sel_col1, sel_col2, sel_col3 = st.columns([5, 1, 5])

    with sel_col1:
        home_team = st.selectbox(
            "Home Team",
            teams,
            index=teams.index("Los Angeles FC") if "Los Angeles FC" in teams else 0,
            key="home_team",
            format_func=lambda x: x,
        )


    with sel_col2:
        st.markdown("""
        <div style="text-align: center; padding: 2.5rem 0 0 0;">
            <div style="font-size: 1.4rem; font-weight: 800; color: #cbd5e1;">VS</div>
        </div>
        """, unsafe_allow_html=True)

    with sel_col3:
        away_team = st.selectbox(
            "Away Team",
            teams,
            index=teams.index("Los Angeles Galaxy") if "Los Angeles Galaxy" in teams else 1,
            key="away_team",
            format_func=lambda x: x,
        )


    if home_team == away_team:
        st.warning("Select two different teams.")
        st.stop()

    st.divider()

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Match Predictor", "Explainability",
        "What-If Simulator", "Model Performance", "Season Rankings"
    ])

    # Train model (cached)
    with st.spinner("Training model…"):
        model_artifacts = train_cached_model()

    # Build base feature vector
    feature_vector = build_feature_vector(
        home_team, away_team, df_feat, current_elo,
        model_artifacts["feature_cols"]
    )

    if feature_vector is None:
        st.error("Not enough historical data for one of the selected teams.")
        st.stop()

    # Get predictions
    probs = predict_match(feature_vector, model_artifacts)
    # probs order: [P(Away=0), P(Draw=1), P(Home=2)]
    prob_a, prob_d, prob_h = probs[0], probs[1], probs[2]

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 1: MATCH PREDICTOR
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        # Match header with logos
        col_h, col_vs, col_a = st.columns([5, 2, 5])
        with col_h:
            logo_h = team_logo_html(home_team, 48)
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem;">
                <div>{logo_h}</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: {COLORS['home']}; margin-top: 0.3rem;">
                    {home_team}
                </div>
                <div style="color: #cbd5e1; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.5px;">HOME</div>
            </div>
            """, unsafe_allow_html=True)
        with col_vs:
            st.markdown("""
            <div style="text-align: center; padding: 0.8rem 0;">
                <div style="font-size: 1.6rem; font-weight: 800; color: #cbd5e1;">VS</div>
            </div>
            """, unsafe_allow_html=True)
        with col_a:
            logo_a = team_logo_html(away_team, 48)
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem;">
                <div>{logo_a}</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: {COLORS['away']}; margin-top: 0.3rem;">
                    {away_team}
                </div>
                <div style="color: #cbd5e1; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.5px;">AWAY</div>
            </div>
            """, unsafe_allow_html=True)



        # ── Probability Bar ──
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=["Prediction"], x=[prob_h], name=f"Home ({prob_h*100:.0f}%)",
            orientation="h", marker_color=COLORS["home"],
            text=f"{prob_h*100:.0f}%", textposition="inside",
            textfont=dict(size=14, family="Inter", color="white"),
            hovertemplate=f"<b>{home_team}</b> (Home Win)<br>Outcome Probability: <b>{prob_h*100:.1f}%</b><extra></extra>",
        ))
        fig_bar.add_trace(go.Bar(
            y=["Prediction"], x=[prob_d], name=f"Draw ({prob_d*100:.0f}%)",
            orientation="h", marker_color=COLORS["draw"],
            text=f"{prob_d*100:.0f}%", textposition="inside",
            textfont=dict(size=14, family="Inter", color="#1a2332"),
            hovertemplate=f"<b>Draw (X)</b><br>Outcome Probability: <b>{prob_d*100:.1f}%</b><extra></extra>",
        ))
        fig_bar.add_trace(go.Bar(
            y=["Prediction"], x=[prob_a], name=f"Away ({prob_a*100:.0f}%)",
            orientation="h", marker_color=COLORS["away"],
            text=f"{prob_a*100:.0f}%", textposition="inside",
            textfont=dict(size=14, family="Inter", color="white"),
            hovertemplate=f"<b>{away_team}</b> (Away Win)<br>Outcome Probability: <b>{prob_a*100:.1f}%</b><extra></extra>",
        ))
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            barmode="stack",
            height=80,
            margin=dict(l=0, r=0, t=32, b=4),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.35,
                xanchor="center",
                x=0.5,
                traceorder="normal",
            ),
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Confidence Indicator ──
        confidence = max(prob_h, prob_d, prob_a)
        if confidence == prob_h:
            conf_outcome = f"Home Win: {home_team}"
        elif confidence == prob_a:
            conf_outcome = f"Away Win: {away_team}"
        else:
            conf_outcome = "Draw"

        if confidence > 0.55:
            conf_color = "#27ae60"
            conf_label = "High"
        elif confidence > 0.40:
            conf_color = "#f1c40f"
            conf_label = "Medium"
        else:
            conf_color = "#e74c3c"
            conf_label = "Low"

        st.divider()

        # ── Radar Chart: Team Comparison ──
        st.markdown("### Team Comparison")
        st.markdown("*Radar chart comparing attack, defense, form, Elo rating, and unbeaten streak on a normalized 0–100 scale.*")
        home_elo = current_elo.get(home_team, 1500)
        away_elo = current_elo.get(away_team, 1500)

        categories = ["Attack", "Defense", "Form", "Elo Rating", "Streak"]
        home_vals = [
            norm(feature_vector.get("home_attack_r5", 1), 0, 3),
            100 - norm(feature_vector.get("home_defense_r5", 1), 0, 3),
            norm(feature_vector.get("home_season_ppg", 1.5), 0, 3),
            norm(home_elo, 1300, 1700),
            norm(feature_vector.get("home_unbeaten_streak", 0), 0, 5),
        ]
        away_vals = [
            norm(feature_vector.get("away_attack_r5", 1), 0, 3),
            100 - norm(feature_vector.get("away_defense_r5", 1), 0, 3),
            norm(feature_vector.get("away_season_ppg", 1.5), 0, 3),
            norm(away_elo, 1300, 1700),
            norm(feature_vector.get("away_unbeaten_streak", 0), 0, 5),
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=home_vals + [home_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(39, 174, 96, 0.15)",
            line=dict(color=COLORS["home"], width=2.5),
            marker=dict(size=7, color=COLORS["home"]),
            name=home_team,
            hovertemplate=f"<b>{home_team}</b> (Home)<br>Metric: <b>%{{theta}}</b><br>Score: <b>%{{r:.1f}} / 100</b><extra></extra>",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=away_vals + [away_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(231, 76, 60, 0.15)",
            line=dict(color=COLORS["away"], width=2.5),
            marker=dict(size=7, color=COLORS["away"]),
            name=away_team,
            hovertemplate=f"<b>{away_team}</b> (Away)<br>Metric: <b>%{{theta}}</b><br>Score: <b>%{{r:.1f}} / 100</b><extra></extra>",
        ))
        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
                angularaxis=dict(linecolor="rgba(255,255,255,0.08)"),
            ),
            height=420,
            margin=dict(l=60, r=60, t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()

        # ── Head-to-Head Score Probability Matrix ──
        st.markdown("### Score Probability Matrix")
        st.markdown(f"*Most probable exact scorelines for **{home_team}** vs **{away_team}** (Poisson model).*")

        # Poisson lambdas: combine attack strength with opponent's defensive weakness
        # attack_r5 = avg goals scored (last 5), defense_r5 = avg goals conceded (last 5)
        home_attack = feature_vector.get("home_attack_r5", 1.3)
        away_defense = feature_vector.get("away_defense_r5", 1.3)  # goals conceded by away team
        away_attack = feature_vector.get("away_attack_r5", 0.9)
        home_defense = feature_vector.get("home_defense_r5", 1.3)  # goals conceded by home team

        # Elo-based adjustment: stronger team gets a small boost
        home_elo_val = feature_vector.get("home_elo_before", 1500)
        away_elo_val = feature_vector.get("away_elo_before", 1500)
        elo_ratio = home_elo_val / max(away_elo_val, 1)  # >1 if home is stronger

        # Expected goals = (team's attack + opponent's goals conceded) / 2
        # with home advantage (~10%) and Elo adjustment
        home_advantage = 1.10
        lambda_h_h2h = max(0.3, (home_attack + away_defense) / 2 * home_advantage * (elo_ratio ** 0.15))
        lambda_a_h2h = max(0.3, (away_attack + home_defense) / 2 / home_advantage * ((1 / elo_ratio) ** 0.15))

        max_goals_h2h = 6
        score_matrix = np.zeros((max_goals_h2h + 1, max_goals_h2h + 1))
        for hg in range(max_goals_h2h + 1):
            for ag in range(max_goals_h2h + 1):
                score_matrix[hg][ag] = poisson.pmf(hg, lambda_h_h2h) * poisson.pmf(ag, lambda_a_h2h)

        # Exact 100% probability conservation normalization across grid domain
        if score_matrix.sum() > 0:
            score_matrix = (score_matrix / score_matrix.sum()) * 100.0

        fig_score_matrix = go.Figure(data=go.Heatmap(
            z=score_matrix,
            x=[str(i) for i in range(max_goals_h2h + 1)],
            y=[str(i) for i in range(max_goals_h2h + 1)],
            colorscale=[
                [0, "#0f1923"],
                [0.15, "#1a2a40"],
                [0.3, "#1D428A"],
                [0.5, "#2e86c1"],
                [0.75, "#27ae60"],
                [1, "#2ecc71"],
            ],
            text=np.round(score_matrix, 1),
            texttemplate="%{text:.1f}%",
            textfont=dict(size=11, family="Inter"),
            hovertemplate=(
                f"<b>Match Scoreline</b><br>"
                f"{home_team} (Home Goals): <b>%{{y}}</b><br>"
                f"{away_team} (Away Goals): <b>%{{x}}</b><br>"
                "Score Probability: <b>%{z:.2f}%</b><extra></extra>"
            ),
            colorbar=dict(title="%", ticksuffix="%"),
        ))
        fig_score_matrix.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title=f"{away_team} Goals",
            yaxis_title=f"{home_team} Goals",
            yaxis=dict(autorange="reversed"),
            height=420,
            margin=dict(l=60, r=60, t=20, b=50),
        )
        st.plotly_chart(fig_score_matrix, use_container_width=True)

        # Summary stats from the matrix
        total_home_win = sum(
            score_matrix[hg][ag]
            for hg in range(max_goals_h2h + 1)
            for ag in range(max_goals_h2h + 1)
            if hg > ag
        )
        total_draw = sum(score_matrix[g][g] for g in range(max_goals_h2h + 1))
        total_away_win = sum(
            score_matrix[hg][ag]
            for hg in range(max_goals_h2h + 1)
            for ag in range(max_goals_h2h + 1)
            if hg < ag
        )
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Home Win (Poisson)", f"{total_home_win:.1f}%")
        sm2.metric("Draw (Poisson)", f"{total_draw:.1f}%")
        sm3.metric("Away Win (Poisson)", f"{total_away_win:.1f}%")

        st.caption(
            "The **Score Matrix** uses a simplified Poisson model (attack + defense rates, "
            "Elo, home advantage) to estimate exact scorelines. The **main prediction** above "
            "uses the full RandomForest model with 40+ features. Minor differences are normal; "
            "the RF model is the primary prediction."
        )

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 2: EXPLAINABILITY
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### Why This Prediction?")
        st.markdown(f"*Understanding why the model predicts this outcome for **{home_team}** vs **{away_team}**.*")

        # ── SHAP Analysis ──
        try:
            from mls_predictor.shap_utils import single_match_shap
            from mls_predictor.feature_descriptions import (
                get_feature_label, get_feature_description,
                format_feature_value, wrap_text_html,
            )

            model = model_artifacts["model"]
            feature_cols = model_artifacts["feature_cols"]
            imputer = model_artifacts["imputer"]

            X_pred = pd.DataFrame([{c: feature_vector.get(c, 0) for c in feature_cols}])
            X_imp = pd.DataFrame(
                imputer.transform(X_pred), columns=feature_cols, index=X_pred.index
            )

            class_names = [
                f"{away_team} (Away Win)",
                "Draw",
                f"{home_team} (Home Win)",
            ]
            shap_result = single_match_shap(
                model, X_imp, feature_names=feature_cols, class_names=class_names,
            )

            if shap_result:
                pred_idx = int(np.argmax([prob_a, prob_d, prob_h]))
                pred_class = class_names[pred_idx]
                if pred_class in shap_result:
                    explanation = shap_result[pred_class]
                elif isinstance(shap_result, dict) and "features" in shap_result:
                    explanation = shap_result
                else:
                    explanation = list(shap_result.values())[0]

                top_features = explanation["features"][:12]

                y_labels = []
                hover_labels = []
                hover_descs = []
                hover_vals = []

                for f in reversed(top_features):
                    feat_name = f["feature"]
                    val = feature_vector.get(feat_name, None)
                    label = get_feature_label(feat_name, home_team, away_team)
                    desc = get_feature_description(feat_name, home_team, away_team)
                    val_str = format_feature_value(feat_name, val)

                    y_labels.append(label)
                    hover_labels.append(label)
                    hover_descs.append(wrap_text_html(desc, max_chars=48))
                    hover_vals.append(val_str)

                fig_shap = go.Figure()
                fig_shap.add_trace(go.Bar(
                    y=y_labels,
                    x=[f["shap_value"] for f in reversed(top_features)],
                    orientation="h",
                    marker=dict(
                        color=[
                            COLORS["home"] if f["shap_value"] > 0 else COLORS["away"]
                            for f in reversed(top_features)
                        ]
                    ),
                    customdata=list(zip(hover_labels, hover_descs, hover_vals)),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br><br>"
                        "<b>Explanation:</b><br>%{customdata[1]}<br><br>"
                        "<b>Match Value:</b> %{customdata[2]}<br>"
                        "<b>SHAP Impact:</b> %{x:+.4f}"
                        "<extra></extra>"
                    ),
                ))
                fig_shap.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(text=f"Feature Impact on \u00ab{pred_class}\u00bb", font=dict(size=14)),
                    xaxis_title="SHAP Value (impact on prediction)",
                    height=max(320, len(top_features) * 34),
                    margin=dict(l=0, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.info("SHAP analysis unavailable: install `shap` package.")
        except Exception as e:
            st.info(f"SHAP explanation not available: {e}")

        st.divider()

        # ── Match Context ──
        st.markdown("### Match Context")
        ctx1, ctx2, ctx3, ctx4 = st.columns(4)
        ctx1.metric("Travel Distance", f"{feature_vector.get('travel_distance_km', 0):.0f} km")
        ctx2.metric("Elo Difference", f"{feature_vector.get('elo_diff', 0):+.0f}")
        ctx3.metric("Home Rest", f"{feature_vector.get('home_rest_days', 7):.0f} days")
        ctx4.metric("Away Rest", f"{feature_vector.get('away_rest_days', 7):.0f} days")

        ctx5, ctx6, ctx7, ctx8 = st.columns(4)
        surface_val = "Grass" if feature_vector.get("venue_is_turf", 0) == 0 else "Turf"
        rivalry_val = "Yes" if feature_vector.get("is_rivalry", 0) == 1 else "No"
        fifa_val = "Near Window" if feature_vector.get("near_fifa_window", 0) == 1 else "Clear"
        ctx5.metric("Surface", surface_val)
        ctx6.metric("Rivalry", rivalry_val)
        ctx7.metric("FIFA Window", fifa_val)
        ctx8.metric("H2H Home Wins", f"{feature_vector.get('h2h_home_wins', 0):.0f} / 5")

        st.divider()

        # ── Recent Form ──
        st.markdown("### Recent Form (Last 5)")
        col_form_h, col_form_a = st.columns(2)

        for col_form, team, color in [
            (col_form_h, home_team, COLORS["home"]),
            (col_form_a, away_team, COLORS["away"]),
        ]:
            with col_form:
                logo_html = team_logo_html(team, 20)
                st.markdown(f"{logo_html} **{team}**", unsafe_allow_html=True)
                form = get_recent_form(df, team, 5)
                for date, opponent, result, gf, ga in form:
                    r_color = {"W": "#27ae60", "D": "#f1c40f", "L": "#e74c3c"}[result]
                    r_badge = f'<span style="color: {r_color}; font-weight: 700;">{result}</span>'
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 0.5rem;
                    background: #1a2332; border-radius: 6px; padding: 0.4rem 0.8rem;
                    margin-bottom: 0.3rem; border-left: 3px solid {r_color};">
                        <span style="color: #94a3b8; font-size: 0.78rem; min-width: 50px; font-weight: 500;">{date.strftime('%b %d')}</span>
                        {r_badge}
                        <span style="color: #f8fafc; font-weight: 700;
                        min-width: 30px;">{gf}-{ga}</span>
                        <span style="color: #cbd5e1; font-size: 0.85rem;">vs {opponent}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 3: WHAT-IF SIMULATOR
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### What-If Simulator")
        st.markdown("*Adjust key variables and see how the prediction changes in real-time.*")

        # ── Interactive Sliders ──
        sl1, sl2 = st.columns(2)
        with sl1:
            rest_h = st.slider(
                "Home Rest Days",
                min_value=2, max_value=14,
                value=int(feature_vector.get("home_rest_days", 7)),
                key="whatif_rest_h",
            )
        with sl2:
            rest_a = st.slider(
                "Away Rest Days",
                min_value=2, max_value=14,
                value=int(feature_vector.get("away_rest_days", 7)),
                key="whatif_rest_a",
            )

        sl3_col1, sl3_col2 = st.columns(2)
        with sl3_col1:
            fatigue = st.slider(
                "Travel Fatigue",
                min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                help="1.0 = normal. >1 = extra fatigue for away team.",
                key="whatif_fatigue",
            )

        # ── Recompute with What-If ──
        fv_whatif = build_feature_vector(
            home_team, away_team, df_feat, current_elo,
            model_artifacts["feature_cols"],
            rest_override_h=rest_h, rest_override_a=rest_a,
            fatigue_mult=fatigue,
        )

        if fv_whatif:
            probs_wif = predict_match(fv_whatif, model_artifacts)
            wif_a, wif_d, wif_h = probs_wif[0], probs_wif[1], probs_wif[2]

            # Show adjusted probabilities
            w1, w2, w3 = st.columns(3)
            delta_h = (wif_h - prob_h) * 100
            delta_d = (wif_d - prob_d) * 100
            delta_a = (wif_a - prob_a) * 100
            w1.metric("Home Win", f"{wif_h*100:.1f}%", delta=f"{delta_h:+.1f}pp")
            w2.metric("Draw", f"{wif_d*100:.1f}%", delta=f"{delta_d:+.1f}pp")
            w3.metric("Away Win", f"{wif_a*100:.1f}%", delta=f"{delta_a:+.1f}pp")

            st.divider()

            # ── Poisson Simulation: Most Probable Exact Scores ──
            st.markdown("### Exact Score Probabilities (Poisson)")
            st.markdown("*Top 10 most likely final scores based on a Poisson model with the adjusted parameters above.*")

            # Poisson lambdas: attack + defense + Elo + home advantage
            _wif_h_atk = fv_whatif.get("home_attack_r5", 1.3)
            _wif_a_def = fv_whatif.get("away_defense_r5", 1.3)
            _wif_a_atk = fv_whatif.get("away_attack_r5", 0.9)
            _wif_h_def = fv_whatif.get("home_defense_r5", 1.3)
            _wif_h_elo = fv_whatif.get("home_elo_before", 1500)
            _wif_a_elo = fv_whatif.get("away_elo_before", 1500)
            _wif_elo_ratio = _wif_h_elo / max(_wif_a_elo, 1)
            _wif_ha = 1.10
            lambda_home = max(0.3, (_wif_h_atk + _wif_a_def) / 2 * _wif_ha * (_wif_elo_ratio ** 0.15))
            lambda_away = max(0.3, (_wif_a_atk + _wif_h_def) / 2 / _wif_ha * ((1 / _wif_elo_ratio) ** 0.15))

            max_goals = 6
            score_probs = {}
            for hg in range(max_goals + 1):
                for ag in range(max_goals + 1):
                    p = poisson.pmf(hg, lambda_home) * poisson.pmf(ag, lambda_away)
                    score_probs[f"{hg}-{ag}"] = p

            top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:10]
            scores_df = pd.DataFrame(top_scores, columns=["Score", "Probability"])

            fig_scores = px.bar(
                scores_df, x="Score", y="Probability",
                color="Probability",
                color_continuous_scale=[[0, "#1D428A"], [0.5, "#2e86c1"], [1, "#27ae60"]],
            )
            fig_scores.update_layout(
                **PLOTLY_LAYOUT,
                coloraxis_showscale=False,
                yaxis_title="Probability",
                xaxis_title="Exact Score (Home - Away)",
                height=350,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            fig_scores.update_traces(
                texttemplate="%{y:.1%}", textposition="outside",
                textfont=dict(size=11, family="Inter"),
                hovertemplate=f"<b>Scoreline: %{{x}}</b> ({home_team} – {away_team})<br>Probability: <b>%{{y:.1%}}</b><extra></extra>",
            )
            st.plotly_chart(fig_scores, use_container_width=True)

            # ── Mini Monte Carlo ──
            st.markdown("### Monte Carlo Simulation (1,000 runs)")
            st.markdown("*1,000 random match simulations using the Poisson model to estimate outcome distributions and average goals.*")
            n_mc = 1000
            mc_results = {"H": 0, "D": 0, "A": 0}
            mc_goals_h = []
            mc_goals_a = []

            for _ in range(n_mc):
                gh = np.random.poisson(lambda_home)
                ga = np.random.poisson(lambda_away)
                mc_goals_h.append(gh)
                mc_goals_a.append(ga)
                if gh > ga:
                    mc_results["H"] += 1
                elif gh == ga:
                    mc_results["D"] += 1
                else:
                    mc_results["A"] += 1

            mc_col1, mc_col2, mc_col3 = st.columns(3)
            mc_col1.metric("Home Wins", f"{mc_results['H']/n_mc*100:.1f}%")
            mc_col2.metric("Draws", f"{mc_results['D']/n_mc*100:.1f}%")
            mc_col3.metric("Away Wins", f"{mc_results['A']/n_mc*100:.1f}%")

            mc_col4, mc_col5 = st.columns(2)
            mc_col4.metric("Avg Home Goals", f"{np.mean(mc_goals_h):.2f}")
            mc_col5.metric("Avg Away Goals", f"{np.mean(mc_goals_a):.2f}")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 4: MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### Model Performance (Test Set: 2025 Season)")

        metrics = model_artifacts["metrics"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        m2.metric("Log-Loss", f"{metrics['log_loss']:.4f}")
        m3.metric("Precision (wt.)", f"{metrics['precision']*100:.1f}%")

        st.caption(
            "**Benchmarks**: For a 3-class football prediction model (Home/Draw/Away), "
            "a random baseline achieves ~33% accuracy. Typical ML models on MLS data reach "
            "40–55% accuracy and 0.95–1.10 log-loss. MLS has higher parity than European "
            "leagues, making prediction inherently harder."
        )

        st.divider()

        # ── Confusion Matrix ──
        st.markdown("### Confusion Matrix")
        st.markdown("*Actual outcomes vs. model predictions on the 2025 test set. Diagonal cells = correct predictions.*")
        from sklearn.metrics import confusion_matrix

        X_test = model_artifacts["X_test"]
        y_test = model_artifacts["y_test"]
        model = model_artifacts["model"]

        y_pred = model.predict(X_test)
        class_names = ["Away (0)", "Draw (1)", "Home (2)"]

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Outcome", y="Actual Outcome", color="Matches"),
            x=class_names, y=class_names,
            text_auto=True,
            color_continuous_scale=[[0, "#0f1923"], [0.5, "#1D428A"], [1, "#2e86c1"]],
        )
        fig_cm.update_traces(
            hovertemplate="<b>Actual Outcome:</b> %{y}<br><b>Model Prediction:</b> %{x}<br><b>Match Count:</b> %{z}<extra></extra>"
        )
        fig_cm.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        st.divider()

        # ── Feature Importance (SHAP Global) ──
        st.markdown("### Global Feature Importance (SHAP)")
        st.markdown("*Which features have the most impact across all predictions. Higher bars = stronger influence on model output.*")
        try:
            from mls_predictor.shap_utils import global_feature_importance
            from mls_predictor.feature_descriptions import (
                get_feature_label, get_feature_description, wrap_text_html,
            )

            with st.spinner("Computing SHAP values…"):
                importance_df = global_feature_importance(
                    model, X_test[:200], model_artifacts["feature_cols"],
                )

            if importance_df is not None:
                top_n = min(20, len(importance_df))
                top_feat = importance_df.head(top_n)

                feats_rev = list(reversed(top_feat["feature"].tolist()))
                y_labels_global = [get_feature_label(f) for f in feats_rev]
                hover_descs_global = [wrap_text_html(get_feature_description(f), max_chars=48) for f in feats_rev]

                fig_imp = go.Figure()
                fig_imp.add_trace(go.Bar(
                    y=y_labels_global,
                    x=list(reversed(top_feat["mean_abs_shap"].tolist())),
                    orientation="h",
                    marker=dict(
                        color=list(reversed(top_feat["mean_abs_shap"].tolist())),
                        colorscale=[[0, "#1D428A"], [0.5, "#2e86c1"], [1, "#27ae60"]],
                    ),
                    customdata=list(zip(y_labels_global, hover_descs_global)),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br><br>"
                        "<b>Explanation:</b><br>%{customdata[1]}<br><br>"
                        "<b>Mean Absolute SHAP:</b> %{x:.4f}"
                        "<extra></extra>"
                    ),
                ))
                fig_imp.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis_title="Mean |SHAP Value|",
                    height=max(350, top_n * 25),
                    margin=dict(l=0, r=20, t=10, b=30),
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Install `shap` for feature importance: `pip install shap`")
        except Exception as e:
            st.info(f"SHAP analysis not available: {e}")

        st.divider()

        # ── Predicted vs Actual Distribution ──
        st.markdown("### Predicted vs Actual Distribution")
        st.markdown("*Comparing the model\'s predicted outcome distribution against actual results. Similar shapes indicate good calibration.*")
        y_proba_test = model.predict_proba(X_test)
        pred_classes = np.argmax(y_proba_test, axis=1)

        class_map = {0: "Away (2)", 1: "Draw (X)", 2: "Home (1)"}
        col_pred, col_actual = st.columns(2)

        with col_pred:
            st.markdown("**Predicted**")
            pred_counts = pd.Series(pred_classes).map(class_map).value_counts()
            fig_p = px.pie(
                names=pred_counts.index, values=pred_counts.values,
                color_discrete_sequence=[COLORS["home"], COLORS["draw"], COLORS["away"]],
                hole=0.55,
            )
            fig_p.update_traces(
                hovertemplate="<b>Predicted: %{label}</b><br>Matches: <b>%{value}</b> (%{percent})<extra></extra>"
            )
            fig_p.update_layout(**PLOTLY_LAYOUT, height=280, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_p, use_container_width=True)

        with col_actual:
            st.markdown("**Actual**")
            actual_counts = pd.Series(y_test).map(class_map).value_counts()
            fig_a = px.pie(
                names=actual_counts.index, values=actual_counts.values,
                color_discrete_sequence=[COLORS["home"], COLORS["draw"], COLORS["away"]],
                hole=0.55,
            )
            fig_a.update_traces(
                hovertemplate="<b>Actual: %{label}</b><br>Matches: <b>%{value}</b> (%{percent})<extra></extra>"
            )
            fig_a.update_layout(**PLOTLY_LAYOUT, height=280, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_a, use_container_width=True)

        st.divider()

        # ── Calibration Curves ──
        st.markdown("### Probability Calibration")
        st.markdown("*Closer to the diagonal = better calibrated model.*")

        n_bins = 10
        for cls_idx, cls_name in class_map.items():
            probs_cal = y_proba_test[:, cls_idx]
            actuals_cal = (y_test == cls_idx).astype(float)

            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_means = []
            bin_true = []

            for i in range(n_bins):
                is_last = (i == n_bins - 1)
                mask = (probs_cal >= bin_edges[i]) & ((probs_cal <= bin_edges[i + 1]) if is_last else (probs_cal < bin_edges[i + 1]))
                if mask.sum() > 0:
                    bin_means.append(probs_cal[mask].mean())
                    bin_true.append(actuals_cal[mask].mean())

            if len(bin_means) > 2:
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(
                    x=bin_means, y=bin_true,
                    mode="markers+lines",
                    marker=dict(size=8, color="#2e86c1"),
                    line=dict(color="#2e86c1", width=2),
                    name=cls_name,
                    hovertemplate=f"<b>{cls_name}</b><br>Predicted Prob: <b>%{{x:.1%}}</b><br>Actual Win Rate: <b>%{{y:.1%}}</b><extra></extra>",
                ))
                fig_cal.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    line=dict(color="#3a4a5e", width=1, dash="dash"),
                    name="Perfect Calibration",
                    hoverinfo="skip",
                ))
                fig_cal.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(text=f"Calibration: {cls_name}", font=dict(size=13)),
                    xaxis_title="Predicted Probability",
                    yaxis_title="Actual Frequency",
                    height=280,
                    margin=dict(l=40, r=20, t=40, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig_cal, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 5: SEASON RANKINGS
    # ══════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("### 2026 Season Simulation")
        st.markdown("*Pre-season Monte Carlo simulation: every team starts from zero. Team strength is derived from Elo ratings and historical features.*")

        from mls_predictor.config import EASTERN_CONFERENCE, WESTERN_CONFERENCE
        from mls_predictor.monte_carlo import (
            simulate_season_detailed,
            simulate_playoff_bracket,
        )

        summary_df, position_dist_df = run_full_season_pipeline(
            df_feat, current_elo, model_artifacts
        )

        # ── Render BOTH conferences sequentially ──
        for conf_teams, conf_label, conf_accent in [
            (EASTERN_CONFERENCE, "Eastern Conference", "#1D428A"),
            (WESTERN_CONFERENCE, "Western Conference", "#C8102E"),
        ]:
            conf_summary = summary_df[summary_df["team"].isin(conf_teams)].copy()
            conf_summary = conf_summary.sort_values("avg_pts", ascending=False).reset_index(drop=True)
            conf_summary.index += 1  # 1-based ranking

            conf_pos_dist_c = position_dist_df[position_dist_df["team"].isin(conf_teams)].copy()

            # ── Conference Header ──
            st.markdown(f"""
            <div style="font-size: 1.2rem; font-weight: 800; color: {conf_accent}; margin: 1rem 0 0.6rem 0;
                padding: 0.6rem 1rem; background: {conf_accent}12; border-left: 4px solid {conf_accent};
                border-radius: 0 8px 8px 0; letter-spacing: 0.5px;">
                <i class="lucide icon-landmark" style="font-size: 1.1rem; vertical-align: middle;"></i> {conf_label} Standings
            </div>
            """, unsafe_allow_html=True)

            # ── 1. Summary Standings Table ──
            table_html = """<div style="overflow-x: auto;">
<table style="width: 100%; border-collapse: collapse; font-family: Inter, sans-serif; font-size: 0.85rem;">
<thead>
<tr style="border-bottom: 2px solid #2a3545; color: #cbd5e1; background: #0b131b;">
<th style="padding: 0.6rem; text-align: center; color: #cbd5e1; font-weight: 700;">#</th>
<th style="padding: 0.6rem; text-align: left; color: #cbd5e1; font-weight: 700;">Team</th>
<th style="padding: 0.6rem; text-align: center; color: #f8fafc; font-weight: 700;">Pts</th>
<th style="padding: 0.6rem; text-align: center; color: #cbd5e1; font-weight: 600;">W</th>
<th style="padding: 0.6rem; text-align: center; color: #cbd5e1; font-weight: 600;">D</th>
<th style="padding: 0.6rem; text-align: center; color: #cbd5e1; font-weight: 600;">L</th>
<th style="padding: 0.6rem; text-align: center; color: #cbd5e1; font-weight: 600;">GF</th>
<th style="padding: 0.6rem; text-align: center; color: #cbd5e1; font-weight: 600;">GA</th>
<th style="padding: 0.6rem; text-align: center; color: #cbd5e1; font-weight: 600;">GD</th>
<th style="padding: 0.6rem; text-align: center; color: #cbd5e1; font-weight: 700;">Playoff %</th>
<th style="padding: 0.6rem; text-align: center; color: #cbd5e1; font-weight: 600;">Bye %</th>
<th style="padding: 0.6rem; text-align: center; color: #cbd5e1; font-weight: 600;">Top 4 %</th>
</tr>
</thead>
<tbody>
"""

            for rank, (_, row) in enumerate(conf_summary.iterrows(), 1):
                team = row["team"]
                logo = team_logo_html(team, 24)

                if rank == 1:
                    row_bg = "rgba(39, 174, 96, 0.08)"
                    border_left = "3px solid #27ae60"
                elif rank <= 4:
                    row_bg = "rgba(46, 134, 193, 0.08)"
                    border_left = "3px solid #2e86c1"
                elif rank <= 9:
                    row_bg = "rgba(241, 196, 15, 0.05)"
                    border_left = "3px solid #f1c40f"
                else:
                    row_bg = "rgba(231, 76, 60, 0.05)"
                    border_left = "3px solid #e74c3c"

                table_html += f"""<tr style="background: {row_bg}; border-bottom: 1px solid #1a2332; border-left: {border_left};">
<td style="padding: 0.5rem; text-align: center; color: #cbd5e1; font-weight: 700;">{rank}</td>
<td style="padding: 0.5rem; text-align: left;">{logo} <span style="color: #f8fafc; font-weight: 600; margin-left: 0.3rem;">{team}</span></td>
<td style="padding: 0.5rem; text-align: center; color: #ffffff; font-weight: 800;">{row['avg_pts']:.1f}</td>
<td style="padding: 0.5rem; text-align: center; color: #f1f5f9;">{row['avg_w']:.1f}</td>
<td style="padding: 0.5rem; text-align: center; color: #f1f5f9;">{row['avg_d']:.1f}</td>
<td style="padding: 0.5rem; text-align: center; color: #f1f5f9;">{row['avg_l']:.1f}</td>
<td style="padding: 0.5rem; text-align: center; color: #f1f5f9;">{row['avg_gf']:.1f}</td>
<td style="padding: 0.5rem; text-align: center; color: #f1f5f9;">{row['avg_ga']:.1f}</td>
<td style="padding: 0.5rem; text-align: center; color: {'#4ade80' if row['avg_gd'] >= 0 else '#f87171'}; font-weight: 700;">{row['avg_gd']:+.1f}</td>
<td style="padding: 0.5rem; text-align: center; color: {'#4ade80' if row['playoff_pct'] > 50 else '#facc15' if row['playoff_pct'] > 20 else '#f87171'}; font-weight: 700;">{row['playoff_pct']:.1f}%</td>
<td style="padding: 0.5rem; text-align: center; color: #cbd5e1;">{row['bye_pct']:.1f}%</td>
<td style="padding: 0.5rem; text-align: center; color: #cbd5e1;">{row['top4_pct']:.1f}%</td>
</tr>
"""

            table_html += "</tbody></table></div>"

            # Legend
            table_html += """<div style="display: flex; flex-wrap: wrap; gap: 1.5rem; margin-top: 0.8rem; font-size: 0.78rem; color: #cbd5e1; font-weight: 500;">
<span><span style="display:inline-block;width:10px;height:10px;background:#27ae60;border-radius:2px;margin-right:4px;"></span>1st Seed (Hosts Round 1 vs Wild Card winner / Shield contender)</span>
<span><span style="display:inline-block;width:10px;height:10px;background:#2e86c1;border-radius:2px;margin-right:4px;"></span>Home Advantage: 2nd–4th seed (Hosts Game 1 & 3 in Round 1)</span>
<span><span style="display:inline-block;width:10px;height:10px;background:#f1c40f;border-radius:2px;margin-right:4px;"></span>Playoff Contender (5th–9th seed)</span>
<span><span style="display:inline-block;width:10px;height:10px;background:#e74c3c;border-radius:2px;margin-right:4px;"></span>Eliminated (10th+)</span>
</div>
"""

            st.markdown(table_html, unsafe_allow_html=True)

            st.divider()

        st.divider()

        # ── 4. MLS Cup Playoff Bracket Simulation ──
        st.markdown("### MLS Cup Playoff Simulation")
        st.markdown("*Playoff bracket probabilities for **both** conferences, plus the MLS Cup Final.*")

        # Build seeds for BOTH conferences
        east_summary = summary_df[summary_df["team"].isin(EASTERN_CONFERENCE)].copy()
        east_summary = east_summary.sort_values("avg_pts", ascending=False).reset_index(drop=True)
        west_summary = summary_df[summary_df["team"].isin(WESTERN_CONFERENCE)].copy()
        west_summary = west_summary.sort_values("avg_pts", ascending=False).reset_index(drop=True)

        east_seeds = east_summary["team"].tolist()[:9]
        west_seeds = west_summary["team"].tolist()[:9]



        def render_playoff_table(playoff_df, seeds, conf_label, accent_color):
            """Render a styled playoff bracket table for one conference."""
            po_table = f"""<div style="margin-bottom: 0.6rem;">
<div style="font-size: 1.1rem; font-weight: 800; color: {accent_color}; margin-bottom: 0.6rem;
    padding: 0.5rem 1rem; background: {accent_color}15; border-left: 3px solid {accent_color};
    border-radius: 0 8px 8px 0;">
    {conf_label}
</div>
<table style="width: 100%; border-collapse: collapse; font-family: Inter, sans-serif; font-size: 0.85rem;">
<thead>
<tr style="border-bottom: 2px solid #2a3545; color: #cbd5e1; background: #0b131b;">
<th style="padding: 0.6rem; text-align: left; color: #cbd5e1; font-weight: 700;">Team</th>
<th style="padding: 0.6rem; text-align: center; color: #cbd5e1; font-weight: 700;">Round 1+ %</th>
<th style="padding: 0.6rem; text-align: center; color: #cbd5e1; font-weight: 700;">Conf. Semi %</th>
<th style="padding: 0.6rem; text-align: center; color: #cbd5e1; font-weight: 700;">Conf. Champion %</th>
</tr>
</thead>
<tbody>
"""
            playoff_display = playoff_df.sort_values("champion_pct", ascending=False)
            for idx, (_, row) in enumerate(playoff_display.iterrows(), 1):
                team = row["team"]
                logo = team_logo_html(team, 22)
                cup_color = "#4ade80" if row["champion_pct"] > 15 else accent_color if row["champion_pct"] > 5 else "#cbd5e1"
                r1_val = row.get("round1_pct", row.get("semifinal_pct", 0))
                semi_val = row.get("conf_semi_pct", row.get("conf_final_pct", 0))

                po_table += f"""<tr style="border-bottom: 1px solid #1a2332;">
<td style="padding: 0.5rem; text-align: left;">{logo} <span style="color: #f8fafc; font-weight: 600; margin-left: 0.3rem;">{team}</span></td>
<td style="padding: 0.5rem; text-align: center; color: #f1f5f9;">{r1_val:.1f}%</td>
<td style="padding: 0.5rem; text-align: center; color: #f1f5f9;">{semi_val:.1f}%</td>
<td style="padding: 0.5rem; text-align: center; color: {cup_color}; font-weight: 700;">{row['champion_pct']:.1f}%</td>
</tr>
"""
            po_table += "</tbody></table></div>"
            return po_table

        if len(east_seeds) >= 9 and len(west_seeds) >= 9:
            with st.spinner("Simulating playoff brackets for both conferences…"):
                east_playoff_df = run_playoff_sim("east", east_seeds, current_elo, 1000)
                west_playoff_df = run_playoff_sim("west", west_seeds, current_elo, 1000)

            # Render both conference brackets
            east_table = render_playoff_table(east_playoff_df, east_seeds, "<i class='lucide icon-trophy' style='font-size: 1rem; vertical-align: middle;'></i> Eastern Conference Playoffs", "#1D428A")
            west_table = render_playoff_table(west_playoff_df, west_seeds, "<i class='lucide icon-trophy' style='font-size: 1rem; vertical-align: middle;'></i> Western Conference Playoffs", "#C8102E")

            st.markdown(east_table, unsafe_allow_html=True)
            st.markdown(west_table, unsafe_allow_html=True)

            st.divider()

            # ── MLS Cup Final Simulation ──
            st.markdown("""<div style="font-size: 1.2rem; font-weight: 800; color: #e8edf2; margin-bottom: 0.6rem;"><i class="lucide icon-trophy" style="font-size: 1.1rem; vertical-align: middle;"></i> MLS Cup Final</div>""", unsafe_allow_html=True)
            st.markdown("*Overall MLS Cup winner probabilities, computed from each team\'s conference champion chance weighted by Elo-based final matchup odds.*")

            from mls_predictor.elo import expected_score

            # Compute MLS Cup win probability for each team
            # P(team wins MLS Cup) = P(team wins conf) × P(team beats other conf champion)
            mls_cup_results = []
            east_champs = east_playoff_df[["team", "champion_pct"]].copy()
            west_champs = west_playoff_df[["team", "champion_pct"]].copy()

            for _, e_row in east_champs.iterrows():
                e_team = e_row["team"]
                e_conf_pct = e_row["champion_pct"] / 100.0
                if e_conf_pct <= 0:
                    continue
                e_elo = current_elo.get(e_team, 1500)

                cup_win_pct = 0.0
                for _, w_row in west_champs.iterrows():
                    w_team = w_row["team"]
                    w_conf_pct = w_row["champion_pct"] / 100.0
                    if w_conf_pct <= 0:
                        continue
                    w_elo = current_elo.get(w_team, 1500)
                    p_east_wins_final = expected_score(e_elo, w_elo)
                    cup_win_pct += e_conf_pct * w_conf_pct * p_east_wins_final

                mls_cup_results.append({
                    "team": e_team,
                    "conference": "East",
                    "conf_champion_pct": e_row["champion_pct"],
                    "mls_cup_pct": cup_win_pct * 100,
                })

            for _, w_row in west_champs.iterrows():
                w_team = w_row["team"]
                w_conf_pct = w_row["champion_pct"] / 100.0
                if w_conf_pct <= 0:
                    continue
                w_elo = current_elo.get(w_team, 1500)

                cup_win_pct = 0.0
                for _, e_row in east_champs.iterrows():
                    e_team = e_row["team"]
                    e_conf_pct = e_row["champion_pct"] / 100.0
                    if e_conf_pct <= 0:
                        continue
                    e_elo = current_elo.get(e_team, 1500)
                    p_west_wins_final = expected_score(w_elo, e_elo)
                    cup_win_pct += w_conf_pct * e_conf_pct * p_west_wins_final

                mls_cup_results.append({
                    "team": w_team,
                    "conference": "West",
                    "conf_champion_pct": w_row["champion_pct"],
                    "mls_cup_pct": cup_win_pct * 100,
                })

            cup_df = pd.DataFrame(mls_cup_results).sort_values("mls_cup_pct", ascending=False).head(10)

            if len(cup_df) > 0:
                fig_cup = go.Figure()
                bar_colors = ["#1D428A" if c == "East" else "#C8102E" for c in cup_df["conference"]]
                fig_cup.add_trace(go.Bar(
                    y=list(reversed(cup_df["team"].tolist())),
                    x=list(reversed(cup_df["mls_cup_pct"].tolist())),
                    orientation="h",
                    marker=dict(color=list(reversed(bar_colors))),
                    text=[f"{v:.1f}%" for v in reversed(cup_df["mls_cup_pct"].tolist())],
                    textposition="outside",
                    textfont=dict(size=12, family="Inter"),
                    hovertemplate="<b>%{y}</b><br>MLS Cup Win Probability: <b>%{x:.1f}%</b><extra></extra>",
                ))
                fig_cup.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis_title="MLS Cup Win Probability (%)",
                    height=max(350, len(cup_df) * 38),
                    margin=dict(l=0, r=60, t=10, b=30),
                )
                st.plotly_chart(fig_cup, use_container_width=True)

                # Legend for conference colors
                st.markdown("""
                <div style="display: flex; gap: 1.5rem; justify-content: center; font-size: 0.82rem; color: #cbd5e1; font-weight: 500; margin-top: 0.5rem;">
                    <span><span style="display:inline-block;width:12px;height:12px;background:#1D428A;border-radius:3px;margin-right:5px;vertical-align:middle;"></span>Eastern Conference</span>
                    <span><span style="display:inline-block;width:12px;height:12px;background:#C8102E;border-radius:3px;margin-right:5px;vertical-align:middle;"></span>Western Conference</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Not enough teams to simulate playoff bracket.")


if __name__ == "__main__":
    main()
