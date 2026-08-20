"""
Central configuration for the MLS 2026 Predictor.
All constants, paths, and team-name mappings live here.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "USA.csv"
LOOKUPS_DIR = DATA_DIR / "lookups"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CALIBRATORS_DIR = MODELS_DIR / "calibrators"

# Ensure directories exist
for d in [DATA_DIR, LOOKUPS_DIR, PROCESSED_DIR, MODELS_DIR, CALIBRATORS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data source URLs ──────────────────────────────────────────────────────
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/new/USA.csv"

# ── Season filter — we only use 2022 onward ──────────────────────────────
MIN_SEASON = 2022
TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASON = 2025
PREDICT_SEASON = 2026

# ── CSV column mapping (football-data.co.uk uses short names) ─────────────
CSV_COLUMN_MAP = {
    "Home": "HomeTeam",
    "Away": "AwayTeam",
    "HG": "FTHG",
    "AG": "FTAG",
    "Res": "FTR",
}

CORE_COLUMNS = [
    "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "Season",
]

ODDS_COLUMNS = [
    "PSCH", "PSCD", "PSCA",       # Pinnacle closing
    "MaxCH", "MaxCD", "MaxCA",     # Market max
    "AvgCH", "AvgCD", "AvgCA",     # Market average
    "B365CH", "B365CD", "B365CA",  # Bet365 closing
]

# ── Rolling window sizes ─────────────────────────────────────────────────
ROLLING_WINDOWS = [3, 5, 10]

# ── Elo defaults ─────────────────────────────────────────────────────────
ELO_INITIAL = 1500
ELO_K = 32
ELO_HOME_ADVANTAGE = 65

# ── Monte Carlo ──────────────────────────────────────────────────────────
MC_SIMULATIONS = 10_000

# ── Team name standardization ────────────────────────────────────────────
TEAM_NAME_MAPPING = {
    "la galaxy": "Los Angeles Galaxy",
    "los angeles galaxy": "Los Angeles Galaxy",
    "atlanta united": "Atlanta Utd",
    "atlanta united fc": "Atlanta Utd",
    "atlanta utd": "Atlanta Utd",
    "charlotte fc": "Charlotte",
    "charlotte": "Charlotte",
    "montreal impact": "CF Montreal",
    "cf montreal": "CF Montreal",
    "club de foot montreal": "CF Montreal",
    "new york red bulls": "New York Red Bulls",
    "ny red bulls": "New York Red Bulls",
    "minnesota united": "Minnesota United",
    "minnesota united fc": "Minnesota United",
    "minnesota utd": "Minnesota United",
    "nashville sc": "Nashville SC",
    "nashville": "Nashville SC",
    "inter miami": "Inter Miami",
    "inter miami cf": "Inter Miami",
    "miami": "Inter Miami",
    "st. louis city": "St. Louis City",
    "st. louis city sc": "St. Louis City",
    "st louis city": "St. Louis City",
    "st louis city sc": "St. Louis City",
    "orlando city": "Orlando City",
    "orlando city sc": "Orlando City",
    "orlando": "Orlando City",
    "colorado": "Colorado Rapids",
    "colorado rapids": "Colorado Rapids",
    "columbus": "Columbus Crew",
    "columbus crew": "Columbus Crew",
    "columbus crew sc": "Columbus Crew",
    "dc united": "DC United",
    "d.c. united": "DC United",
    "fc cincinnati": "FC Cincinnati",
    "cincinnati": "FC Cincinnati",
    "fc dallas": "FC Dallas",
    "dallas": "FC Dallas",
    "houston": "Houston Dynamo",
    "houston dynamo": "Houston Dynamo",
    "houston dynamo fc": "Houston Dynamo",
    "lafc": "Los Angeles FC",
    "los angeles fc": "Los Angeles FC",
    "new england": "New England Revolution",
    "new england revolution": "New England Revolution",
    "new york city": "New York City",
    "new york city fc": "New York City",
    "nycfc": "New York City",
    "philadelphia": "Philadelphia Union",
    "philadelphia union": "Philadelphia Union",
    "portland": "Portland Timbers",
    "portland timbers": "Portland Timbers",
    "portland timbers fc": "Portland Timbers",
    "real salt lake": "Real Salt Lake",
    "rsl": "Real Salt Lake",
    "san jose": "San Jose Earthquakes",
    "san jose earthquakes": "San Jose Earthquakes",
    "seattle": "Seattle Sounders",
    "seattle sounders": "Seattle Sounders",
    "seattle sounders fc": "Seattle Sounders",
    "sporting kansas city": "Sporting Kansas City",
    "sporting kc": "Sporting Kansas City",
    "st. louis": "St. Louis City",
    "vancouver": "Vancouver Whitecaps",
    "vancouver whitecaps": "Vancouver Whitecaps",
    "vancouver whitecaps fc": "Vancouver Whitecaps",
    "san diego": "San Diego FC",
    "san diego fc": "San Diego FC",
    "austin fc": "Austin FC",
    "austin": "Austin FC",
    "chicago fire": "Chicago Fire",
    "chicago fire fc": "Chicago Fire",
    "chicago": "Chicago Fire",
    "toronto fc": "Toronto FC",
    "toronto": "Toronto FC",
}


def standardize_team_name(name: str) -> str:
    """Map team name variations to the canonical name used in the dataset."""
    if not isinstance(name, str):
        return name
    return TEAM_NAME_MAPPING.get(name.strip().lower(), name.strip())


# ── Conference assignments (2024-2026) ───────────────────────────────────
EASTERN_CONFERENCE = [
    "Atlanta Utd", "Charlotte", "CF Montreal", "Chicago Fire",
    "Columbus Crew", "DC United", "FC Cincinnati", "Inter Miami",
    "Nashville SC", "New England Revolution", "New York City",
    "New York Red Bulls", "Orlando City", "Philadelphia Union",
    "Toronto FC",
]

WESTERN_CONFERENCE = [
    "Austin FC", "Colorado Rapids", "FC Dallas", "Houston Dynamo",
    "Los Angeles FC", "Los Angeles Galaxy", "Minnesota United",
    "Portland Timbers", "Real Salt Lake", "San Diego FC",
    "San Jose Earthquakes", "Seattle Sounders",
    "Sporting Kansas City", "St. Louis City", "Vancouver Whitecaps",
]

# ── Canonical MLS franchise inaugural seasons ────────────────────────────
MLS_EXPANSION_SEASONS = {
    # 1996 Originals
    "Colorado Rapids": 1996, "Columbus Crew": 1996, "DC United": 1996,
    "FC Dallas": 1996, "Sporting Kansas City": 1996, "Los Angeles Galaxy": 1996,
    "New England Revolution": 1996, "New York Red Bulls": 1996,
    "San Jose Earthquakes": 1996,
    # Expansion Eras
    "Chicago Fire": 1998, "Real Salt Lake": 2005, "Houston Dynamo": 2006,
    "Toronto FC": 2007, "Seattle Sounders": 2009, "Philadelphia Union": 2010,
    "Portland Timbers": 2011, "Vancouver Whitecaps": 2011, "CF Montreal": 2012,
    "New York City": 2015, "Orlando City": 2015, "Atlanta Utd": 2017,
    "Minnesota United": 2017, "Los Angeles FC": 2018, "FC Cincinnati": 2019,
    "Inter Miami": 2020, "Nashville SC": 2020, "Austin FC": 2021,
    "Charlotte": 2022, "St. Louis City": 2023, "San Diego FC": 2025,
}

