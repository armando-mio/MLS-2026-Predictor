"""
Feature engineering for MLS match prediction.

Computes all feature groups:
  1. Geography & Logistics (TFI, timezone, surface, altitude)
  2. Roster & Absences (DP availability, FIFA windows, squad quality)
  3. Tactical / Advanced (xG rolling, xT proxy, goals efficiency, rolling windows)
  4. Context & Motivation (playoff pressure, congestion, H2H, streaks)
  5. Betting Odds (implied probabilities, market confidence, odds movement)
  6. Elo ratings (added externally via elo.py, referenced here for completeness)
"""

import logging
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd

from mls_predictor.config import (
    ROLLING_WINDOWS, standardize_team_name, MLS_EXPANSION_SEASONS,
)
from mls_predictor.data_loader import (
    load_stadiums, load_fifa_windows, load_rivalries,
    load_designated_players, haversine_km,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  1. GEOGRAPHY & LOGISTICS
# ═══════════════════════════════════════════════════════════════════════════

def add_geography_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add travel fatigue, timezone, surface, altitude features."""
    stadiums = load_stadiums()
    df = df.copy()

    travel_dist = []
    tz_crossed = []
    surface_venue = []
    surface_mismatch = []
    altitude = []
    high_altitude = []
    tfi_values = []

    # Pre-compute timezone offsets (simplified: use fixed UTC offsets per zone)
    tz_offset_map = {
        "America/New_York": -5, "America/Toronto": -5,
        "America/Chicago": -6,
        "America/Denver": -7,
        "America/Los_Angeles": -8, "America/Vancouver": -8,
    }

    for idx, row in df.iterrows():
        ht = row["HomeTeam"]
        at = row["AwayTeam"]

        home_info = stadiums.get(ht, {})
        away_info = stadiums.get(at, {})

        # Travel distance (away team traveling to home venue)
        if home_info and away_info:
            dist = haversine_km(
                away_info["lat"], away_info["lon"],
                home_info["lat"], home_info["lon"]
            )
        else:
            dist = 0.0
        travel_dist.append(dist)

        # Timezone difference
        h_tz = tz_offset_map.get(home_info.get("timezone", ""), -5)
        a_tz = tz_offset_map.get(away_info.get("timezone", ""), -5)
        tz_crossed.append(abs(h_tz - a_tz))

        # Surface
        venue_surf = home_info.get("surface", "grass")
        surface_venue.append(1 if venue_surf == "turf" else 0)
        away_surf = away_info.get("surface", "grass")
        surface_mismatch.append(1 if venue_surf != away_surf else 0)

        # Altitude
        alt = home_info.get("altitude_m", 0)
        altitude.append(alt)
        high_altitude.append(1 if alt >= 1200 else 0)

    df["travel_distance_km"] = travel_dist
    df["timezones_crossed"] = tz_crossed
    df["venue_is_turf"] = surface_venue
    df["surface_mismatch"] = surface_mismatch
    df["venue_altitude_m"] = altitude
    df["high_altitude"] = high_altitude

    # Travel Fatigue Index: sum of distances traveled by away team in last 3 matches
    df = _compute_travel_fatigue_index(df, stadiums, n_matches=3)

    logger.info("Geography features added (%d rows)", len(df))
    return df


def _compute_travel_fatigue_index(
    df: pd.DataFrame, stadiums: dict, n_matches: int = 3
) -> pd.DataFrame:
    """
    For each match, compute the cumulative travel (km) the away team
    covered in their last `n_matches` away games.
    """
    tfi = []
    for idx, row in df.iterrows():
        away_team = row["AwayTeam"]
        match_date = row["Date"]

        # Find last n_matches for this team (home or away) before this date
        past = df[(
            ((df["HomeTeam"] == away_team) | (df["AwayTeam"] == away_team)) &
            (df["Date"] < match_date)
        )].tail(n_matches)

        total_km = 0.0
        team_info = stadiums.get(away_team, {})
        if not team_info:
            tfi.append(0.0)
            continue

        for _, past_row in past.iterrows():
            # Determine the venue of each past match
            if past_row["HomeTeam"] == away_team:
                venue_team = away_team  # played at home
            else:
                venue_team = past_row["HomeTeam"]  # played away at this venue

            venue_info = stadiums.get(venue_team, {})
            if venue_info:
                total_km += haversine_km(
                    team_info["lat"], team_info["lon"],
                    venue_info["lat"], venue_info["lon"]
                )

        tfi.append(total_km)

    df["travel_fatigue_index"] = tfi
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  2. ROSTER & ABSENCES
# ═══════════════════════════════════════════════════════════════════════════

def add_roster_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add DP availability and FIFA window impact features."""
    fifa_windows = load_fifa_windows()
    dp_data = load_designated_players()
    df = df.copy()

    # ── FIFA Window Impact ──
    during_window = []
    days_since_window = []

    for _, row in df.iterrows():
        match_date = row["Date"]
        in_window = 0
        min_days_since = 999

        for w in fifa_windows:
            if w["start"] <= match_date <= w["end"]:
                in_window = 1
            days_after = (match_date - w["end"]).days
            if 0 <= days_after < min_days_since:
                min_days_since = days_after

        during_window.append(in_window)
        days_since_window.append(min(min_days_since, 60))  # cap at 60

    df["during_fifa_window"] = during_window
    df["days_since_fifa_window"] = days_since_window
    df["near_fifa_window"] = ((df["days_since_fifa_window"] <= 7) | (df["during_fifa_window"] == 1)).astype(int)

    # ── DP Quality (simplified) ──
    home_dp_quality = []
    away_dp_quality = []

    for _, row in df.iterrows():
        season_str = str(int(row["Season"]))
        season_data = dp_data.get(season_str, {})

        h_players = season_data.get(row["HomeTeam"], [])
        a_players = season_data.get(row["AwayTeam"], [])

        # Average quality of DPs; 6.0 baseline if not explicitly listed
        h_q = np.mean([p["quality"] for p in h_players]) if h_players else 6.0
        a_q = np.mean([p["quality"] for p in a_players]) if a_players else 6.0

        home_dp_quality.append(h_q)
        away_dp_quality.append(a_q)

    df["home_dp_quality"] = home_dp_quality
    df["away_dp_quality"] = away_dp_quality
    df["dp_quality_diff"] = df["home_dp_quality"] - df["away_dp_quality"]

    logger.info("Roster features added (%d rows)", len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  3. TACTICAL / ADVANCED — Rolling Windows
# ═══════════════════════════════════════════════════════════════════════════

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling averages for attack / defense metrics at multiple
    window sizes. Uses shifted values to prevent data leakage.
    """
    df = df.copy()
    logger.info("Computing rolling features with windows %s …", ROLLING_WINDOWS)

    # Build a team-level timeline
    home_rows = pd.DataFrame({
        "Date": df["Date"], "MatchIndex": df.index,
        "Team": df["HomeTeam"],
        "GoalsScored": df["FTHG"], "GoalsConceded": df["FTAG"],
        "IsHome": 1,
    })
    away_rows = pd.DataFrame({
        "Date": df["Date"], "MatchIndex": df.index,
        "Team": df["AwayTeam"],
        "GoalsScored": df["FTAG"], "GoalsConceded": df["FTHG"],
        "IsHome": 0,
    })
    team_df = pd.concat([home_rows, away_rows]).sort_values(["Team", "Date"]).reset_index(drop=True)

    # Shift within each team so we only see past matches
    team_df["prev_gs"] = team_df.groupby("Team")["GoalsScored"].shift(1)
    team_df["prev_gc"] = team_df.groupby("Team")["GoalsConceded"].shift(1)

    # Total goals in match (for O/U and efficiency features)
    team_df["prev_total"] = team_df["prev_gs"] + team_df["prev_gc"]
    # Both teams scored? (for GG/NG feature)
    team_df["prev_bts"] = ((team_df.groupby("Team")["GoalsScored"].shift(1) > 0) &
                            (team_df.groupby("Team")["GoalsConceded"].shift(1) > 0)).astype(float)

    for w in ROLLING_WINDOWS:
        team_df[f"attack_r{w}"] = team_df.groupby("Team")["prev_gs"].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        team_df[f"defense_r{w}"] = team_df.groupby("Team")["prev_gc"].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        team_df[f"total_r{w}"] = team_df.groupby("Team")["prev_total"].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        team_df[f"bts_r{w}"] = team_df.groupby("Team")["prev_bts"].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )

    # Map back to original df
    home_stats = team_df[team_df["IsHome"] == 1].set_index("MatchIndex")
    away_stats = team_df[team_df["IsHome"] == 0].set_index("MatchIndex")

    for w in ROLLING_WINDOWS:
        df[f"home_attack_r{w}"] = home_stats[f"attack_r{w}"]
        df[f"home_defense_r{w}"] = home_stats[f"defense_r{w}"]
        df[f"away_attack_r{w}"] = away_stats[f"attack_r{w}"]
        df[f"away_defense_r{w}"] = away_stats[f"defense_r{w}"]
        df[f"home_total_r{w}"] = home_stats[f"total_r{w}"]
        df[f"away_total_r{w}"] = away_stats[f"total_r{w}"]
        df[f"home_bts_r{w}"] = home_stats[f"bts_r{w}"]
        df[f"away_bts_r{w}"] = away_stats[f"bts_r{w}"]

    # Goal-scoring efficiency: goals / expected (using rolling avg as proxy)
    for w in ROLLING_WINDOWS:
        # Attack differential
        df[f"attack_diff_r{w}"] = df[f"home_attack_r{w}"] - df[f"away_attack_r{w}"]
        df[f"defense_diff_r{w}"] = df[f"home_defense_r{w}"] - df[f"away_defense_r{w}"]

    logger.info("Rolling features added for windows %s", ROLLING_WINDOWS)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  4. CONTEXT & MOTIVATION
# ═══════════════════════════════════════════════════════════════════════════

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest days, congestion, H2H, streaks, rivalry, expansion penalty."""
    df = df.copy()
    rivalries = load_rivalries()

    # Build a quick rivalry lookup
    rivalry_map = {}
    for r in rivalries:
        key = tuple(sorted(r["teams"]))
        rivalry_map[key] = r["intensity"]

    # ── Single-pass Chronological State Tracking (O(N) vs O(N^2)) ──
    last_match_date: dict[str, pd.Timestamp] = {}
    match_dates_history = defaultdict(list)
    h2h_history = defaultdict(list)  # (min_t, max_t) -> list of (home_team, away_team, ftr)
    unbeaten_streaks = defaultdict(int)
    season_pts = defaultdict(lambda: [0, 0])  # (team, season) -> [points, games]
    season_home = defaultdict(lambda: [0, 0])  # (team, season) -> [home_wins, home_games]
    season_away = defaultdict(lambda: [0, 0])  # (team, season) -> [away_wins, away_games]

    home_rest, away_rest = [], []
    home_cong, away_cong = [], []
    h2h_hw, h2h_d, h2h_aw = [], [], []
    home_unbeaten, away_unbeaten = [], []
    home_ppg, away_ppg = [], []
    home_hwr, away_awr = [], []
    is_rivalry, rivalry_intensity = [], []
    home_expansion, away_expansion = [], []

    for row in df.itertuples():
        m_date = row.Date
        season = row.Season
        ht = row.HomeTeam
        at = row.AwayTeam
        ftr = getattr(row, "FTR", None)

        # 1. Rest Days (capped at 14 to avoid off-season skew)
        h_last = last_match_date.get(ht)
        a_last = last_match_date.get(at)
        h_r = min((m_date - h_last).days, 14) if h_last is not None else 14
        a_r = min((m_date - a_last).days, 14) if a_last is not None else 14
        home_rest.append(h_r)
        away_rest.append(a_r)

        # 2. Congestion (matches in prior 14 days)
        w_start = m_date - timedelta(days=14)
        h_c = sum(1 for d in match_dates_history[ht] if w_start <= d < m_date)
        a_c = sum(1 for d in match_dates_history[at] if w_start <= d < m_date)
        home_cong.append(h_c)
        away_cong.append(a_c)

        # 3. Head-to-Head (last 5 meetings between these two specific clubs)
        pair_key = tuple(sorted([ht, at]))
        past_h2h = h2h_history[pair_key][-5:]
        hw, dw, aw = 0, 0, 0
        for h_past, a_past, res in past_h2h:
            if res == "H":
                if h_past == ht: hw += 1
                else: aw += 1
            elif res == "A":
                if a_past == ht: hw += 1
                else: aw += 1
            else:
                dw += 1
        h2h_hw.append(hw)
        h2h_d.append(dw)
        h2h_aw.append(aw)

        # 4. Unbeaten Streaks
        home_unbeaten.append(unbeaten_streaks[ht])
        away_unbeaten.append(unbeaten_streaks[at])

        # 5. Season PPG prior to this match
        h_p, h_g = season_pts[(ht, season)]
        a_p, a_g = season_pts[(at, season)]
        home_ppg.append(h_p / max(h_g, 1) if h_g > 0 else 1.4)
        away_ppg.append(a_p / max(a_g, 1) if a_g > 0 else 1.2)

        # 6. Home / Away win rates (season)
        hh_w, hh_g = season_home[(ht, season)]
        aa_w, aa_g = season_away[(at, season)]
        home_hwr.append(hh_w / max(hh_g, 1) if hh_g > 0 else 0.50)
        away_awr.append(aa_w / max(aa_g, 1) if aa_g > 0 else 0.20)

        # 7. Rivalry
        is_riv = 1 if pair_key in rivalry_map else 0
        is_rivalry.append(is_riv)
        rivalry_intensity.append(rivalry_map.get(pair_key, 0))

        # 8. Expansion Team Status
        home_expansion.append(1 if (season - MLS_EXPANSION_SEASONS.get(ht, 1996)) < 2 else 0)
        away_expansion.append(1 if (season - MLS_EXPANSION_SEASONS.get(at, 1996)) < 2 else 0)

        # ── State Updates (Post-Match) ──
        last_match_date[ht] = m_date
        last_match_date[at] = m_date
        match_dates_history[ht].append(m_date)
        match_dates_history[at].append(m_date)

        if ftr in ["H", "D", "A"]:
            h2h_history[pair_key].append((ht, at, ftr))

            # Streak update
            if ftr == "H":
                unbeaten_streaks[ht] += 1
                unbeaten_streaks[at] = 0
                season_pts[(ht, season)][0] += 3
                season_pts[(ht, season)][1] += 1
                season_pts[(at, season)][1] += 1
                season_home[(ht, season)][0] += 1
                season_home[(ht, season)][1] += 1
                season_away[(at, season)][1] += 1
            elif ftr == "A":
                unbeaten_streaks[ht] = 0
                unbeaten_streaks[at] += 1
                season_pts[(at, season)][0] += 3
                season_pts[(ht, season)][1] += 1
                season_pts[(at, season)][1] += 1
                season_home[(ht, season)][1] += 1
                season_away[(at, season)][0] += 1
                season_away[(at, season)][1] += 1
            else:  # Draw
                unbeaten_streaks[ht] += 1
                unbeaten_streaks[at] += 1
                season_pts[(ht, season)][0] += 1
                season_pts[(ht, season)][1] += 1
                season_pts[(at, season)][0] += 1
                season_pts[(at, season)][1] += 1
                season_home[(ht, season)][1] += 1
                season_away[(at, season)][1] += 1

    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["home_congestion"] = home_cong
    df["away_congestion"] = away_cong
    df["h2h_home_wins"] = h2h_hw
    df["h2h_draws"] = h2h_d
    df["h2h_away_wins"] = h2h_aw
    df["home_unbeaten_streak"] = home_unbeaten
    df["away_unbeaten_streak"] = away_unbeaten
    df["home_season_ppg"] = home_ppg
    df["away_season_ppg"] = away_ppg
    df["ppg_diff"] = df["home_season_ppg"] - df["away_season_ppg"]
    df["home_home_win_rate"] = home_hwr
    df["away_away_win_rate"] = away_awr
    df["is_rivalry"] = is_rivalry
    df["rivalry_intensity"] = rivalry_intensity
    df["home_expansion"] = home_expansion
    df["away_expansion"] = away_expansion

    logger.info("Context features added (%d rows)", len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  5. BETTING ODDS — Implied Probabilities
# ═══════════════════════════════════════════════════════════════════════════

def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert closing odds to implied probabilities (vig-removed).
    Add market confidence and odds movement features.
    """
    df = df.copy()

    # ── Pinnacle implied probs (vig-removed) ──
    odds_sets = [
        ("PSCH", "PSCD", "PSCA", "pin"),
        ("AvgCH", "AvgCD", "AvgCA", "avg"),
        ("B365CH", "B365CD", "B365CA", "b365"),
    ]

    for h_col, d_col, a_col, prefix in odds_sets:
        if h_col in df.columns and d_col in df.columns and a_col in df.columns:
            # Cast to numeric — odds may arrive as strings from pyarrow
            h_odds = pd.to_numeric(df[h_col], errors="coerce").replace(0, np.nan)
            d_odds = pd.to_numeric(df[d_col], errors="coerce").replace(0, np.nan)
            a_odds = pd.to_numeric(df[a_col], errors="coerce").replace(0, np.nan)

            raw_h = 1.0 / h_odds
            raw_d = 1.0 / d_odds
            raw_a = 1.0 / a_odds
            overround = raw_h + raw_d + raw_a

            df[f"{prefix}_impl_home"] = raw_h / overround
            df[f"{prefix}_impl_draw"] = raw_d / overround
            df[f"{prefix}_impl_away"] = raw_a / overround

    # ── Market Confidence (how decisive the sharp market is) ──
    if "pin_impl_home" in df.columns:
        impl_cols = ["pin_impl_home", "pin_impl_draw", "pin_impl_away"]
        df["market_confidence"] = df[impl_cols].max(axis=1)
        # idxmax fails on all-NaN rows; fill NaN before calling, then mask back
        has_odds = df[impl_cols].notna().any(axis=1)
        df["market_favourite"] = np.nan
        if has_odds.any():
            df.loc[has_odds, "market_favourite"] = (
                df.loc[has_odds, impl_cols]
                .idxmax(axis=1)
                .map({"pin_impl_home": 2, "pin_impl_draw": 1, "pin_impl_away": 0})
            )

    # ── Odds Movement: Pinnacle vs. Market Average ──
    if "PSCH" in df.columns and "AvgCH" in df.columns:
        df["odds_move_home"] = pd.to_numeric(df["PSCH"], errors="coerce") - pd.to_numeric(df["AvgCH"], errors="coerce")
        df["odds_move_draw"] = pd.to_numeric(df["PSCD"], errors="coerce") - pd.to_numeric(df["AvgCD"], errors="coerce")
        df["odds_move_away"] = pd.to_numeric(df["PSCA"], errors="coerce") - pd.to_numeric(df["AvgCA"], errors="coerce")

    logger.info("Odds features added")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  6. TARGETS
# ═══════════════════════════════════════════════════════════════════════════

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add all prediction target columns."""
    df = df.copy()

    # 1X2 target: H=2, D=1, A=0
    target_map = {"H": 2, "D": 1, "A": 0}
    df["target_1x2"] = df["FTR"].map(target_map)

    # Over/Under 2.5: 1 = Over, 0 = Under
    total_goals = df["FTHG"] + df["FTAG"]
    df["target_ou25"] = (total_goals >= 2.5).astype(int)

    # GG/NG: 1 = Both teams scored, 0 = at least one team didn't score
    df["target_ggng"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)

    df = df.dropna(subset=["target_1x2"]).reset_index(drop=True)

    logger.info("Targets added — 1X2: %s | O/U: %s | GG/NG: %s",
                df["target_1x2"].value_counts().to_dict(),
                df["target_ou25"].value_counts().to_dict(),
                df["target_ggng"].value_counts().to_dict())
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline in dependency order.
    Expects df from data_loader.load_raw_data() with Elo already computed.
    """
    logger.info("═══ Starting feature engineering pipeline ═══")
    df = add_geography_features(df)
    df = add_roster_features(df)
    df = add_rolling_features(df)
    df = add_context_features(df)
    df = add_odds_features(df)
    df = add_targets(df)

    # Drop rows with NaN in critical rolling features (first few matches per team)
    rolling_cols = [c for c in df.columns if "_r3" in c or "_r5" in c]
    if rolling_cols:
        before = len(df)
        df = df.dropna(subset=rolling_cols).reset_index(drop=True)
        logger.info("Dropped %d rows with NaN rolling features", before - len(df))

    logger.info("═══ Feature pipeline complete: %d matches, %d features ═══",
                len(df), len(df.columns))
    return df


def get_feature_columns() -> dict:
    """
    Return the feature column lists for each model target.
    Odds features are EXCLUDED from prediction (not available for future matches)
    but included during training when available.
    """
    # Base features available for all models
    base = [
        # Geography
        "travel_distance_km", "timezones_crossed", "venue_is_turf",
        "surface_mismatch", "venue_altitude_m", "high_altitude",
        "travel_fatigue_index",
        # Roster
        "during_fifa_window", "days_since_fifa_window", "near_fifa_window",
        "home_dp_quality", "away_dp_quality", "dp_quality_diff",
        # Context
        "home_rest_days", "away_rest_days", "rest_diff",
        "home_congestion", "away_congestion",
        "h2h_home_wins", "h2h_draws", "h2h_away_wins",
        "home_unbeaten_streak", "away_unbeaten_streak",
        "home_season_ppg", "away_season_ppg", "ppg_diff",
        "home_home_win_rate", "away_away_win_rate",
        "is_rivalry", "rivalry_intensity",
        "home_expansion", "away_expansion",
        # Elo
        "home_elo_before", "away_elo_before", "elo_diff",
    ]

    # Rolling features (multi-window)
    rolling = []
    for w in ROLLING_WINDOWS:
        rolling += [
            f"home_attack_r{w}", f"home_defense_r{w}",
            f"away_attack_r{w}", f"away_defense_r{w}",
            f"home_total_r{w}", f"away_total_r{w}",
            f"home_bts_r{w}", f"away_bts_r{w}",
            f"attack_diff_r{w}", f"defense_diff_r{w}",
        ]

    # Odds features (isolated for historical betting market benchmarking)
    odds = [
        "pin_impl_home", "pin_impl_draw", "pin_impl_away",
        "market_confidence",
        "odds_move_home", "odds_move_draw", "odds_move_away",
    ]

    production_features = base + rolling

    return {
        "base": base,
        "rolling": rolling,
        "odds": odds,
        "production": production_features,
        "all": production_features,  # Default training and prediction to leak-free pre-match features
        "prediction": production_features,
        "odds_augmented": production_features + odds,
    }
