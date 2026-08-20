"""
Data loading, downloading, ASA API integration, and preprocessing.
"""
import json
import logging
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from mls_predictor.config import (
    RAW_CSV, DATA_DIR, LOOKUPS_DIR, PROCESSED_DIR,
    FOOTBALL_DATA_URL, CSV_COLUMN_MAP, CORE_COLUMNS,
    ODDS_COLUMNS, MIN_SEASON, PREDICT_SEASON, standardize_team_name,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Raw Data Download
# ═══════════════════════════════════════════════════════════════════════════

def download_csv(force: bool = False) -> None:
    """Download the Football-Data.co.uk MLS CSV if absent or forced, with offline fallback."""
    if RAW_CSV.exists() and not force:
        logger.info("CSV already present at %s", RAW_CSV)
        return
    logger.info("Downloading MLS data from %s …", FOOTBALL_DATA_URL)
    try:
        resp = requests.get(FOOTBALL_DATA_URL, timeout=15)
        resp.raise_for_status()
        RAW_CSV.write_bytes(resp.content)
        logger.info("Saved %d bytes → %s", len(resp.content), RAW_CSV)
    except Exception as e:
        if RAW_CSV.exists():
            logger.warning("Download failed (%s); using existing local CSV at %s", e, RAW_CSV)
        else:
            logger.error("Download failed and no local CSV exists: %s", e)
            raise


# ═══════════════════════════════════════════════════════════════════════════
#  Load & Preprocess Raw CSV
# ═══════════════════════════════════════════════════════════════════════════

def load_raw_data(exclude_current_season: bool = True) -> pd.DataFrame:
    """Load, rename columns, filter to MIN_SEASON+, standardize names.

    Parameters
    ----------
    exclude_current_season : bool
        If True (default), remove PREDICT_SEASON (2026) from the dataset
        so the model only trains/tests on completed seasons.
    """
    download_csv()
    df = pd.read_csv(RAW_CSV)

    # Rename short-form columns
    df = df.rename(columns={k: v for k, v in CSV_COLUMN_MAP.items() if k in df.columns})

    # Keep core + odds columns that exist
    keep = [c for c in CORE_COLUMNS + ODDS_COLUMNS if c in df.columns]
    df = df[keep].copy()

    # Drop incomplete rows
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season"])

    # Season filter
    df = df[df["Season"] >= MIN_SEASON].copy()

    # Exclude current season if requested
    if exclude_current_season:
        df = df[df["Season"] < PREDICT_SEASON].copy()

    # Standardize team names
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip().apply(standardize_team_name)
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip().apply(standardize_team_name)
    df["FTR"] = df["FTR"].astype(str).str.strip()

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")

    # Cast goal columns
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")

    # Sort chronologically with stable index
    if "Time" in df.columns:
        df["Time"] = df["Time"].fillna("00:00").astype(str)
        df = df.sort_values(["Date", "Time"]).reset_index(drop=True)
    else:
        df = df.sort_values(["Date"]).reset_index(drop=True)

    logger.info("Loaded %d matches (%s → %s), exclude_current=%s",
                len(df), df["Date"].min().date(), df["Date"].max().date(),
                exclude_current_season)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  ASA API Client (itscalledsoccer)
# ═══════════════════════════════════════════════════════════════════════════

_ASA_CACHE_PATH = PROCESSED_DIR / "asa_cache.parquet"


def _try_import_asa():
    """Attempt to import the ASA client library."""
    try:
        from itscalledsoccer.client import AmericanSoccerAnalysis
        return AmericanSoccerAnalysis()
    except ImportError:
        logger.warning("itscalledsoccer not installed — ASA features unavailable. "
                       "Install with: pip install itscalledsoccer")
        return None
    except Exception as e:
        logger.warning("Failed to initialize ASA client: %s", e)
        return None


def fetch_asa_team_xg(seasons: list[int] | None = None) -> pd.DataFrame | None:
    """
    Fetch per-game team xG data from the ASA API.
    Returns DataFrame with columns: [date, home_team, away_team, home_xg, away_xg, ...]
    Falls back to cached parquet if API is unreachable.
    """
    # Try cache first
    if _ASA_CACHE_PATH.exists():
        logger.info("Loading ASA data from cache: %s", _ASA_CACHE_PATH)
        return pd.read_parquet(_ASA_CACHE_PATH)

    client = _try_import_asa()
    if client is None:
        return None

    try:
        logger.info("Fetching team xG from ASA API …")
        # Get team game-level xG data
        if seasons:
            games = client.get_team_xgoals(leagues="mls", season_name=seasons)
        else:
            games = client.get_team_xgoals(leagues="mls")

        if games is not None and len(games) > 0:
            # Cache locally
            games.to_parquet(_ASA_CACHE_PATH, index=False)
            logger.info("Cached %d ASA records → %s", len(games), _ASA_CACHE_PATH)
            return games
        else:
            logger.warning("ASA API returned empty data")
            return None
    except Exception as e:
        logger.warning("ASA API call failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Lookup Loaders
# ═══════════════════════════════════════════════════════════════════════════

def load_stadiums() -> dict:
    """Load stadium metadata as {team_name: {lat, lon, altitude_m, surface, timezone}}."""
    path = LOOKUPS_DIR / "stadiums.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {s["team"]: s for s in data["stadiums"]}


def load_fifa_windows() -> list[dict]:
    """Load FIFA international window date ranges."""
    path = LOOKUPS_DIR / "fifa_windows.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    windows = []
    for w in data["windows"]:
        windows.append({
            "start": pd.Timestamp(w["start"]),
            "end": pd.Timestamp(w["end"]),
            "name": w["name"],
        })
    return windows


def load_rivalries() -> list[dict]:
    """Load rivalry pairings."""
    path = LOOKUPS_DIR / "rivalries.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["rivalries"]


def load_designated_players() -> dict:
    """Load DP lookup as {season_str: {team: [players]}}."""
    path = LOOKUPS_DIR / "designated_players.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Remove the _note key
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ═══════════════════════════════════════════════════════════════════════════
#  Utility: Haversine distance
# ═══════════════════════════════════════════════════════════════════════════

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))
