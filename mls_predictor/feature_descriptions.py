"""
Feature descriptions and human-readable metadata for MLS match predictors.
Provides user-friendly labels, metric explanations, and value formatters
for SHAP explainability and feature importance visualizations.
"""
import re


def wrap_text_html(text: str, max_chars: int = 50) -> str:
    """Wraps text with <br> tags every max_chars for clean, readable tooltip rendering."""
    if not text:
        return ""
    words = text.split(" ")
    lines = []
    current_line = []
    current_len = 0
    for w in words:
        if current_len + len(w) + 1 > max_chars and current_line:
            lines.append(" ".join(current_line))
            current_line = [w]
            current_len = len(w)
        else:
            current_line.append(w)
            current_len += len(w) + 1
    if current_line:
        lines.append(" ".join(current_line))
    return "<br>".join(lines)


def get_feature_label(feat_name: str, home_team: str = "Home", away_team: str = "Away") -> str:
    """
    Returns a clean, human-readable display label for a feature,
    optionally customized with specific team names.
    """
    # Regex matchers for rolling windows
    m = re.match(r"^home_attack_r(\d+)$", feat_name)
    if m:
        return f"{home_team} Attack (Last {m.group(1)})"
    m = re.match(r"^home_defense_r(\d+)$", feat_name)
    if m:
        return f"{home_team} Defense (Last {m.group(1)})"
    m = re.match(r"^away_attack_r(\d+)$", feat_name)
    if m:
        return f"{away_team} Attack (Last {m.group(1)})"
    m = re.match(r"^away_defense_r(\d+)$", feat_name)
    if m:
        return f"{away_team} Defense (Last {m.group(1)})"
    m = re.match(r"^home_total_r(\d+)$", feat_name)
    if m:
        return f"{home_team} Total Goals (Last {m.group(1)})"
    m = re.match(r"^away_total_r(\d+)$", feat_name)
    if m:
        return f"{away_team} Total Goals (Last {m.group(1)})"
    m = re.match(r"^home_bts_r(\d+)$", feat_name)
    if m:
        return f"{home_team} BTTS Rate (Last {m.group(1)})"
    m = re.match(r"^away_bts_r(\d+)$", feat_name)
    if m:
        return f"{away_team} BTTS Rate (Last {m.group(1)})"
    m = re.match(r"^attack_diff_r(\d+)$", feat_name)
    if m:
        return f"Attack Diff (Last {m.group(1)})"
    m = re.match(r"^defense_diff_r(\d+)$", feat_name)
    if m:
        return f"Defense Diff (Last {m.group(1)})"

    static_labels = {
        # Geography
        "travel_distance_km": f"{away_team} Travel Distance (km)",
        "timezones_crossed": f"Time Zones Crossed ({away_team})",
        "venue_is_turf": f"Artificial Turf Venue ({home_team})",
        "surface_mismatch": f"Surface Mismatch ({away_team})",
        "venue_altitude_m": f"Stadium Altitude ({home_team})",
        "high_altitude": f"High Altitude Venue (>=1200m)",
        "travel_fatigue_index": f"{away_team} Travel Fatigue Index",
        # Roster
        "during_fifa_window": "Active FIFA Window",
        "days_since_fifa_window": "Days Since FIFA Window",
        "near_fifa_window": "Near FIFA International Break",
        "home_dp_quality": f"{home_team} DP Star Quality",
        "away_dp_quality": f"{away_team} DP Star Quality",
        "dp_quality_diff": f"DP Quality Differential ({home_team} - {away_team})",
        # Context
        "home_rest_days": f"{home_team} Rest Days",
        "away_rest_days": f"{away_team} Rest Days",
        "rest_diff": f"Rest Advantage ({home_team} - {away_team})",
        "home_congestion": f"{home_team} Schedule Congestion",
        "away_congestion": f"{away_team} Schedule Congestion",
        "h2h_home_wins": f"H2H {home_team} Wins",
        "h2h_draws": "H2H Draws",
        "h2h_away_wins": f"H2H {away_team} Wins",
        "home_unbeaten_streak": f"{home_team} Unbeaten Streak",
        "away_unbeaten_streak": f"{away_team} Unbeaten Streak",
        "home_season_ppg": f"{home_team} Season PPG",
        "away_season_ppg": f"{away_team} Season PPG",
        "ppg_diff": f"Season PPG Diff ({home_team} - {away_team})",
        "home_home_win_rate": f"{home_team} Home Win Rate",
        "away_away_win_rate": f"{away_team} Away Win Rate",
        "is_rivalry": "Rivalry / Derby Match",
        "rivalry_intensity": "Rivalry Intensity Index",
        "home_expansion": f"{home_team} Expansion Team",
        "away_expansion": f"{away_team} Expansion Team",
        # Elo
        "home_elo_before": f"{home_team} Pre-Match Elo",
        "away_elo_before": f"{away_team} Pre-Match Elo",
        "elo_diff": f"Elo Rating Differential ({home_team} - {away_team})",
        # Betting Odds
        "pin_impl_home": "Market Implied Home Win %",
        "pin_impl_draw": "Market Implied Draw %",
        "pin_impl_away": "Market Implied Away Win %",
        "market_confidence": "Market Confidence Index",
        "odds_move_home": "Home Odds Movement",
        "odds_move_draw": "Draw Odds Movement",
        "odds_move_away": "Away Odds Movement",
    }

    if feat_name in static_labels:
        return static_labels[feat_name]

    # Fallback: clean title case
    clean = feat_name.replace("_", " ").title()
    return clean


def get_feature_description(feat_name: str, home_team: str = "Home Team", away_team: str = "Away Team") -> str:
    """
    Returns an informative, comprehensive explanation of the metric and what it measures.
    """
    # Regex matchers for rolling windows
    m = re.match(r"^home_attack_r(\d+)$", feat_name)
    if m:
        w = m.group(1)
        return f"Average goals scored per match by {home_team} over their last {w} games. Indicates current offensive form."
    m = re.match(r"^home_defense_r(\d+)$", feat_name)
    if m:
        w = m.group(1)
        return f"Average goals conceded per match by {home_team} over their last {w} games. Lower values indicate strong defensive stability."
    m = re.match(r"^away_attack_r(\d+)$", feat_name)
    if m:
        w = m.group(1)
        return f"Average goals scored per match by {away_team} over their last {w} games. Measures away offensive potency."
    m = re.match(r"^away_defense_r(\d+)$", feat_name)
    if m:
        w = m.group(1)
        return f"Average goals conceded per match by {away_team} over their last {w} games. Measures away defensive resilience."
    m = re.match(r"^home_total_r(\d+)$", feat_name)
    if m:
        w = m.group(1)
        return f"Average combined match goals in games involving {home_team} across the last {w} matches."
    m = re.match(r"^away_total_r(\d+)$", feat_name)
    if m:
        w = m.group(1)
        return f"Average combined match goals in games involving {away_team} across the last {w} matches."
    m = re.match(r"^home_bts_r(\d+)$", feat_name)
    if m:
        w = m.group(1)
        return f"Both Teams To Score (BTTS) rate in {home_team}'s last {w} matches (frequency of both teams scoring)."
    m = re.match(r"^away_bts_r(\d+)$", feat_name)
    if m:
        w = m.group(1)
        return f"Both Teams To Score (BTTS) rate in {away_team}'s last {w} matches (frequency of both teams scoring)."
    m = re.match(r"^attack_diff_r(\d+)$", feat_name)
    if m:
        w = m.group(1)
        return f"Attack differential over the last {w} matches ({home_team} goals scored - {away_team} goals scored)."
    m = re.match(r"^defense_diff_r(\d+)$", feat_name)
    if m:
        w = m.group(1)
        return f"Defense differential over the last {w} matches ({home_team} goals conceded - {away_team} goals conceded)."

    static_descriptions = {
        # Geography
        "travel_distance_km": f"Direct flight distance (km) traveled by {away_team} to reach {home_team}'s stadium. Longer journeys contribute to away fatigue.",
        "timezones_crossed": f"Number of time zones crossed by {away_team} for this fixture, affecting circadian rhythm and recovery.",
        "venue_is_turf": f"Indicates if {home_team}'s stadium uses artificial turf instead of natural grass, altering ball speed and physical demands.",
        "surface_mismatch": f"Indicates {away_team} plays home games on grass/turf and must adapt to the opposite playing surface at {home_team}'s venue.",
        "venue_altitude_m": f"Elevation of {home_team}'s stadium in meters above sea level. High altitude gives the home side a physical stamina advantage.",
        "high_altitude": f"Flag indicating extreme stadium altitude (>=1,200m, e.g. Colorado or Salt Lake) causing rapid visiting fatigue.",
        "travel_fatigue_index": f"Cumulative travel distance (km) covered by {away_team} across their recent consecutive away matches.",
        # Roster
        "during_fifa_window": "Match takes place during an active FIFA international window when key national team players may be absent.",
        "days_since_fifa_window": "Days elapsed since the previous international break, reflecting international squad recovery time.",
        "near_fifa_window": "Match scheduled near or during an international window, risking absence or travel fatigue of star Designated Players.",
        "home_dp_quality": f"Average quality and talent rating of {home_team}'s Designated Players (marquee star players).",
        "away_dp_quality": f"Average quality and talent rating of {away_team}'s Designated Players (marquee star players).",
        "dp_quality_diff": f"Star player quality differential ({home_team} DP Rating - {away_team} DP Rating). Higher values favor {home_team}.",
        # Context
        "home_rest_days": f"Days of physical recovery and preparation {home_team} had since their previous competitive match.",
        "away_rest_days": f"Days of physical recovery and preparation {away_team} had since their previous competitive match.",
        "rest_diff": f"Rest advantage in days ({home_team} rest - {away_team} rest). Positive values mean {home_team} is fresher.",
        "home_congestion": f"Schedule congestion: number of competitive matches played by {home_team} in the prior 14 days.",
        "away_congestion": f"Schedule congestion: number of competitive matches played by {away_team} in the prior 14 days.",
        "h2h_home_wins": f"Number of victories by {home_team} in the last 5 head-to-head matches between these two clubs.",
        "h2h_draws": f"Number of drawn matches in the last 5 head-to-head meetings between {home_team} and {away_team}.",
        "h2h_away_wins": f"Number of victories by {away_team} in the last 5 head-to-head matches between these two clubs.",
        "home_unbeaten_streak": f"Current consecutive matches without defeat for {home_team} entering this fixture.",
        "away_unbeaten_streak": f"Current consecutive matches without defeat for {away_team} entering this fixture.",
        "home_season_ppg": f"Average points per game (PPG) earned by {home_team} throughout the current season.",
        "away_season_ppg": f"Average points per game (PPG) earned by {away_team} throughout the current season.",
        "ppg_diff": f"Season PPG differential ({home_team} PPG - {away_team} PPG). Compares overall regular-season caliber.",
        "home_home_win_rate": f"Historical winning percentage for {home_team} when playing at their home stadium.",
        "away_away_win_rate": f"Historical winning percentage for {away_team} when playing away on the road.",
        "is_rivalry": "Indicates a historic MLS rivalry or geographic derby matchup with elevated competitive intensity.",
        "rivalry_intensity": "Historical intensity rating based on disciplinary cards and fouls in previous derby clashes.",
        "home_expansion": f"Indicates {home_team} is in an inaugural/expansion season phase with developing squad chemistry.",
        "away_expansion": f"Indicates {away_team} is in an inaugural/expansion season phase with developing squad chemistry.",
        # Elo
        "home_elo_before": f"Pre-match Elo rating of {home_team} (1500 baseline). Measures long-term relative strength.",
        "away_elo_before": f"Pre-match Elo rating of {away_team} (1500 baseline). Measures long-term relative strength.",
        "elo_diff": f"Pre-match Elo rating difference ({home_team} Elo - {away_team} Elo). Positive values favor {home_team}.",
        # Betting Odds
        "pin_impl_home": "Vig-removed Pinnacle market implied probability for a Home Win based on pre-match closing odds.",
        "pin_impl_draw": "Vig-removed Pinnacle market implied probability for a Draw based on pre-match closing odds.",
        "pin_impl_away": "Vig-removed Pinnacle market implied probability for an Away Win based on pre-match closing odds.",
        "market_confidence": "Consensus market confidence level and liquidity certainty behind pre-match betting odds.",
        "odds_move_home": "Pre-match betting market price movement towards or against a Home Win.",
        "odds_move_draw": "Pre-match betting market price movement for a Draw outcome.",
        "odds_move_away": "Pre-match betting market price movement towards or against an Away Win.",
    }

    if feat_name in static_descriptions:
        return static_descriptions[feat_name]

    return f"Statistical feature representing '{feat_name.replace('_', ' ')}' used in the predictive model."


def format_feature_value(feat_name: str, val: any) -> str:
    """
    Formats the raw feature value for intuitive display in tooltips and tables.
    """
    if val is None:
        return "N/A"
    
    try:
        fval = float(val)
    except (ValueError, TypeError):
        return str(val)

    # Boolean / Binary indicators
    if feat_name in ["venue_is_turf"]:
        return "Turf (Artificial)" if fval >= 0.5 else "Grass (Natural)"
    if feat_name in ["surface_mismatch"]:
        return "Mismatch (Yes)" if fval >= 0.5 else "Familiar Surface (No)"
    if feat_name in ["high_altitude"]:
        return "High Altitude (Yes)" if fval >= 0.5 else "Standard/Sea Level (No)"
    if feat_name in ["during_fifa_window"]:
        return "Yes (Active Break)" if fval >= 0.5 else "No (Clear)"
    if feat_name in ["near_fifa_window"]:
        return "Yes (Within 7 Days)" if fval >= 0.5 else "No (Clear)"
    if feat_name in ["is_rivalry"]:
        return "Yes (Derby)" if fval >= 0.5 else "No (Standard Match)"
    if feat_name in ["home_expansion", "away_expansion"]:
        return "Expansion Club (Yes)" if fval >= 0.5 else "Established (No)"

    # Specific units
    if feat_name in ["travel_distance_km", "travel_fatigue_index"]:
        return f"{fval:,.0f} km"
    if feat_name in ["venue_altitude_m"]:
        return f"{fval:,.0f} m"
    if feat_name in ["timezones_crossed"]:
        tz_suffix = "s" if int(fval) != 1 else ""
        return f"{int(fval)} time zone{tz_suffix}"
    if feat_name in ["home_rest_days", "away_rest_days"]:
        return f"{fval:.0f} days"
    if feat_name in ["rest_diff"]:
        return f"{fval:+.0f} days"
    if feat_name in ["days_since_fifa_window"]:
        return f"{fval:.0f} days"
    if feat_name in ["home_congestion", "away_congestion"]:
        return f"{fval:.0f} matches in 14d"
    if feat_name in ["h2h_home_wins", "h2h_draws", "h2h_away_wins"]:
        return f"{fval:.0f} / 5 matches"
    if feat_name in ["home_unbeaten_streak", "away_unbeaten_streak"]:
        return f"{fval:.0f} matches"
    if feat_name in ["home_season_ppg", "away_season_ppg"]:
        return f"{fval:.2f} PPG"
    if feat_name in ["ppg_diff"]:
        return f"{fval:+.2f} PPG"
    if feat_name in ["home_home_win_rate", "away_away_win_rate"]:
        return f"{fval * 100:.1f}% win rate"
    if feat_name in ["home_dp_quality", "away_dp_quality"]:
        return f"{fval:.1f} / 10"
    if feat_name in ["dp_quality_diff"]:
        return f"{fval:+.1f} rating diff"
    if feat_name in ["home_elo_before", "away_elo_before"]:
        return f"{fval:.0f} Elo"
    if feat_name in ["elo_diff"]:
        return f"{fval:+.0f} Elo diff"
    if feat_name in ["rivalry_intensity"]:
        return f"{fval:.1f} / 10"
    if feat_name.startswith("pin_impl_"):
        return f"{fval * 100:.1f}% implied prob"

    m_bts = re.match(r"^(?:home|away)_bts_r(\d+)$", feat_name)
    if m_bts:
        w = int(m_bts.group(1))
        games_bts = round(fval * w)
        return f"{games_bts} of {w} matches ({fval * 100:.0f}%)"

    if "_attack_r" in feat_name or "_defense_r" in feat_name or "_total_r" in feat_name:
        return f"{fval:.2f} goals/match"
    if "diff_r" in feat_name:
        return f"{fval:+.2f} goals/match"

    if abs(fval) < 0.001 and fval != 0:
        return f"{fval:.4f}"
    if abs(fval) >= 100:
        return f"{fval:.1f}"
    return f"{fval:.2f}"
