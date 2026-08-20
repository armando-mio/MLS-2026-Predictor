"""
Elo rating system for MLS teams.
Provides a dynamic strength metric that updates after every match.
"""
import logging
import pandas as pd
import numpy as np

from mls_predictor.config import ELO_INITIAL, ELO_K, ELO_HOME_ADVANTAGE, PROCESSED_DIR

logger = logging.getLogger(__name__)


def expected_score(elo_a: float, elo_b: float) -> float:
    """Expected score for team A against team B."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def compute_elo_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Elo ratings for every team after every match.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, Season.
        Must be sorted chronologically.

    Returns
    -------
    pd.DataFrame
        Original df with added columns:
        - home_elo_before, away_elo_before: pre-match Elo ratings
        - home_elo_after, away_elo_after: post-match Elo ratings
    """
    logger.info("Computing Elo ratings …")
    ratings = {}  # team -> current Elo
    season_tracker = {}  # team -> last season seen (for regression to mean)

    home_elo_before = []
    away_elo_before = []
    home_elo_after = []
    away_elo_after = []

    for _, row in df.iterrows():
        ht = row["HomeTeam"]
        at = row["AwayTeam"]
        season = row["Season"]

        # Initialize or regress toward mean at season start
        for team in [ht, at]:
            if team not in ratings:
                ratings[team] = ELO_INITIAL
                season_tracker[team] = season
            elif season_tracker[team] != season:
                # Regress 1/3 toward the mean at season boundary
                ratings[team] = ratings[team] * 0.67 + ELO_INITIAL * 0.33
                season_tracker[team] = season

        # Pre-match Elo (home team gets advantage baked in for prediction)
        elo_h = ratings[ht] + ELO_HOME_ADVANTAGE
        elo_a = ratings[at]

        home_elo_before.append(ratings[ht])
        away_elo_before.append(ratings[at])

        # Expected scores
        exp_h = expected_score(elo_h, elo_a)
        exp_a = 1.0 - exp_h

        # Actual result
        ftr = row["FTR"]
        if ftr == "H":
            actual_h, actual_a = 1.0, 0.0
        elif ftr == "A":
            actual_h, actual_a = 0.0, 1.0
        else:  # Draw
            actual_h, actual_a = 0.5, 0.5

        # Goal difference multiplier (rewards dominant wins)
        goal_diff = abs(row["FTHG"] - row["FTAG"])
        if goal_diff <= 1:
            gd_mult = 1.0
        elif goal_diff == 2:
            gd_mult = 1.5
        else:
            gd_mult = (11.0 + goal_diff) / 8.0

        # Update
        ratings[ht] += ELO_K * gd_mult * (actual_h - exp_h)
        ratings[at] += ELO_K * gd_mult * (actual_a - exp_a)

        home_elo_after.append(ratings[ht])
        away_elo_after.append(ratings[at])

    df = df.copy()
    df["home_elo_before"] = home_elo_before
    df["away_elo_before"] = away_elo_before
    df["home_elo_after"] = home_elo_after
    df["away_elo_after"] = away_elo_after
    df["elo_diff"] = df["home_elo_before"] - df["away_elo_before"]

    logger.info("Elo computed for %d matches. Rating range: %.0f – %.0f",
                len(df),
                min(min(home_elo_before), min(away_elo_before)),
                max(max(home_elo_after), max(away_elo_after)))

    return df


def get_current_elo(df_with_elo: pd.DataFrame) -> dict:
    """
    Extract the most recent Elo for each team.

    Returns dict {team_name: elo_rating}.
    """
    teams = set(df_with_elo["HomeTeam"].unique()) | set(df_with_elo["AwayTeam"].unique())
    current = {}

    for team in teams:
        home_matches = df_with_elo[df_with_elo["HomeTeam"] == team]
        away_matches = df_with_elo[df_with_elo["AwayTeam"] == team]

        last_home = home_matches.iloc[-1]["home_elo_after"] if len(home_matches) > 0 else ELO_INITIAL
        last_away = away_matches.iloc[-1]["away_elo_after"] if len(away_matches) > 0 else ELO_INITIAL

        # Take the more recent one
        if len(home_matches) > 0 and len(away_matches) > 0:
            if home_matches.iloc[-1]["Date"] >= away_matches.iloc[-1]["Date"]:
                current[team] = last_home
            else:
                current[team] = last_away
        elif len(home_matches) > 0:
            current[team] = last_home
        else:
            current[team] = last_away

    return current


def save_elo_history(df_with_elo: pd.DataFrame) -> None:
    """Save Elo history to parquet."""
    path = PROCESSED_DIR / "elo_ratings.parquet"
    cols = ["Date", "HomeTeam", "AwayTeam", "Season",
            "home_elo_before", "away_elo_before",
            "home_elo_after", "away_elo_after", "elo_diff"]
    df_with_elo[cols].to_parquet(path, index=False)
    logger.info("Elo history saved → %s", path)
