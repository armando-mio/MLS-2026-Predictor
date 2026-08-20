"""
Monte Carlo simulation engine for MLS playoff predictions.

Simulates the remainder of the regular season and the full playoff bracket
using model-generated match probabilities.
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from mls_predictor.config import (
    MC_SIMULATIONS,
    EASTERN_CONFERENCE, WESTERN_CONFERENCE,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Regular Season Simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_remaining_season(
    standings: dict,
    remaining_matches: list[dict],
    n_sims: int = MC_SIMULATIONS,
) -> pd.DataFrame:
    """
    Simulate the remaining regular season matches.

    Parameters
    ----------
    standings : dict
        {team_name: {"points": int, "gd": int, "gf": int, "played": int}}
    remaining_matches : list[dict]
        Each dict: {"home": str, "away": str, "prob_h": float, "prob_d": float, "prob_a": float}
    n_sims : int
        Number of Monte Carlo simulations.

    Returns
    -------
    pd.DataFrame with columns: team, avg_points, playoff_pct, avg_position
    """
    logger.info("Running %d season simulations with %d remaining matches …",
                n_sims, len(remaining_matches))

    team_points = defaultdict(list)
    team_positions = defaultdict(list)
    playoff_counts = defaultdict(int)

    for sim in range(n_sims):
        # Copy standings
        sim_standings = {t: dict(s) for t, s in standings.items()}

        # Simulate each remaining match
        for match in remaining_matches:
            rand = np.random.random()
            if rand < match["prob_h"]:
                sim_standings[match["home"]]["points"] += 3
                sim_standings[match["home"]]["gf"] += 2
                sim_standings[match["away"]]["gf"] += 1
                sim_standings[match["home"]]["gd"] += 1
                sim_standings[match["away"]]["gd"] -= 1
            elif rand < match["prob_h"] + match["prob_d"]:
                sim_standings[match["home"]]["points"] += 1
                sim_standings[match["away"]]["points"] += 1
                sim_standings[match["home"]]["gf"] += 1
                sim_standings[match["away"]]["gf"] += 1
            else:
                sim_standings[match["away"]]["points"] += 3
                sim_standings[match["away"]]["gf"] += 2
                sim_standings[match["home"]]["gf"] += 1
                sim_standings[match["home"]]["gd"] -= 1
                sim_standings[match["away"]]["gd"] += 1

        # Record final points
        for team, stats in sim_standings.items():
            team_points[team].append(stats["points"])

        # Determine playoff qualifiers per conference (top 9)
        for conf_teams in [EASTERN_CONFERENCE, WESTERN_CONFERENCE]:
            conf_standings = [
                (t, sim_standings.get(t, {"points": 0, "gd": 0}))
                for t in conf_teams if t in sim_standings
            ]
            conf_standings.sort(key=lambda x: (x[1]["points"], x[1]["gd"]), reverse=True)

            for rank, (team, _) in enumerate(conf_standings, 1):
                team_positions[team].append(rank)
                if rank <= 9:
                    playoff_counts[team] += 1

    # Build results
    results = []
    for team in standings.keys():
        pts_list = team_points.get(team, [0])
        pos_list = team_positions.get(team, [15])
        results.append({
            "team": team,
            "avg_points": np.mean(pts_list),
            "std_points": np.std(pts_list),
            "playoff_pct": playoff_counts.get(team, 0) / n_sims * 100,
            "avg_position": np.mean(pos_list),
        })

    result_df = pd.DataFrame(results).sort_values("avg_points", ascending=False).reset_index(drop=True)
    logger.info("Season simulation complete. Top playoff team: %s (%.1f%%)",
                result_df.iloc[0]["team"], result_df.iloc[0]["playoff_pct"])
    return result_df


# ═══════════════════════════════════════════════════════════════════════════
#  Detailed Season Simulation (for Season Rankings tab)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_season_detailed(
    standings: dict,
    remaining_matches: list[dict],
    n_sims: int = MC_SIMULATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enhanced season simulation with per-team stats and position distributions.

    Parameters
    ----------
    standings : dict
        {team_name: {"points": int, "gd": int, "gf": int, "ga": int,
                      "played": int, "w": int, "d": int, "l": int}}
    remaining_matches : list[dict]
        Each dict: {"home": str, "away": str, "prob_h": float,
                     "prob_d": float, "prob_a": float,
                     "lambda_h": float, "lambda_a": float}
    n_sims : int
        Number of Monte Carlo simulations.

    Returns
    -------
    summary_df : pd.DataFrame
        Per-team averages: avg_pts, avg_gf, avg_ga, avg_gd, avg_w, avg_d, avg_l,
        playoff_pct, bye_pct, top4_pct, eliminated_pct
    position_dist_df : pd.DataFrame
        Team × position matrix showing frequency (%) of finishing in each rank.
    """
    logger.info("Running %d detailed season simulations with %d remaining matches …",
                n_sims, len(remaining_matches))

    # Accumulators
    team_stats = defaultdict(lambda: {
        "points": [], "gf": [], "ga": [], "gd": [],
        "w": [], "d": [], "l": [],
    })
    # Position tracking: team -> {position -> count}
    team_position_counts = defaultdict(lambda: defaultdict(int))
    # Bracket tracking
    bracket_counts = defaultdict(lambda: {
        "bye": 0, "top4": 0, "playoff": 0, "eliminated": 0,
    })

    for sim in range(n_sims):
        # Deep-copy standings with full stats
        sim_standings = {}
        for t, s in standings.items():
            sim_standings[t] = {
                "points": s.get("points", 0),
                "gf": s.get("gf", 0),
                "ga": s.get("ga", 0),
                "gd": s.get("gd", 0),
                "w": s.get("w", 0),
                "d": s.get("d", 0),
                "l": s.get("l", 0),
            }

        # Simulate each remaining match using Poisson goals
        for match in remaining_matches:
            lambda_h = match.get("lambda_h", 1.4)
            lambda_a = match.get("lambda_a", 1.1)

            goals_h = np.random.poisson(lambda_h)
            goals_a = np.random.poisson(lambda_a)

            home_t = match["home"]
            away_t = match["away"]

            # Update goals
            if home_t in sim_standings:
                sim_standings[home_t]["gf"] += goals_h
                sim_standings[home_t]["ga"] += goals_a
                sim_standings[home_t]["gd"] += (goals_h - goals_a)
            if away_t in sim_standings:
                sim_standings[away_t]["gf"] += goals_a
                sim_standings[away_t]["ga"] += goals_h
                sim_standings[away_t]["gd"] += (goals_a - goals_h)

            # Determine result and update points/W/D/L
            if goals_h > goals_a:
                if home_t in sim_standings:
                    sim_standings[home_t]["points"] += 3
                    sim_standings[home_t]["w"] += 1
                if away_t in sim_standings:
                    sim_standings[away_t]["l"] += 1
            elif goals_h == goals_a:
                if home_t in sim_standings:
                    sim_standings[home_t]["points"] += 1
                    sim_standings[home_t]["d"] += 1
                if away_t in sim_standings:
                    sim_standings[away_t]["points"] += 1
                    sim_standings[away_t]["d"] += 1
            else:
                if away_t in sim_standings:
                    sim_standings[away_t]["points"] += 3
                    sim_standings[away_t]["w"] += 1
                if home_t in sim_standings:
                    sim_standings[home_t]["l"] += 1

        # Record final stats for each team
        for team, stats in sim_standings.items():
            team_stats[team]["points"].append(stats["points"])
            team_stats[team]["gf"].append(stats["gf"])
            team_stats[team]["ga"].append(stats["ga"])
            team_stats[team]["gd"].append(stats["gd"])
            team_stats[team]["w"].append(stats["w"])
            team_stats[team]["d"].append(stats["d"])
            team_stats[team]["l"].append(stats["l"])

        # Rank per conference and track positions + brackets
        # Official MLS Tiebreakers: 1. Points, 2. Total Wins (w), 3. Goal Differential (gd), 4. Goals For (gf)
        for conf_teams in [EASTERN_CONFERENCE, WESTERN_CONFERENCE]:
            conf_standings = [
                (t, sim_standings.get(t, {"points": 0, "w": 0, "gd": 0, "gf": 0}))
                for t in conf_teams if t in sim_standings
            ]
            conf_standings.sort(
                key=lambda x: (x[1]["points"], x[1]["w"], x[1]["gd"], x[1]["gf"]),
                reverse=True,
            )

            for rank, (team, _) in enumerate(conf_standings, 1):
                team_position_counts[team][rank] += 1

                if rank == 1:
                    bracket_counts[team]["bye"] += 1  # 1st place in conference / shield contender
                if rank <= 4:
                    bracket_counts[team]["top4"] += 1
                if rank <= 9:
                    bracket_counts[team]["playoff"] += 1
                else:
                    bracket_counts[team]["eliminated"] += 1

    # Build summary DataFrame
    summary_rows = []
    for team in standings.keys():
        stats = team_stats[team]
        brackets = bracket_counts[team]
        summary_rows.append({
            "team": team,
            "avg_pts": np.mean(stats["points"]) if stats["points"] else 0,
            "std_pts": np.std(stats["points"]) if stats["points"] else 0,
            "avg_gf": np.mean(stats["gf"]) if stats["gf"] else 0,
            "avg_ga": np.mean(stats["ga"]) if stats["ga"] else 0,
            "avg_gd": np.mean(stats["gd"]) if stats["gd"] else 0,
            "avg_w": np.mean(stats["w"]) if stats["w"] else 0,
            "avg_d": np.mean(stats["d"]) if stats["d"] else 0,
            "avg_l": np.mean(stats["l"]) if stats["l"] else 0,
            "bye_pct": brackets["bye"] / n_sims * 100,
            "top4_pct": brackets["top4"] / n_sims * 100,
            "playoff_pct": brackets["playoff"] / n_sims * 100,
            "eliminated_pct": brackets["eliminated"] / n_sims * 100,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("avg_pts", ascending=False).reset_index(drop=True)

    # Build position distribution DataFrame
    all_teams = list(standings.keys())
    max_pos = 15  # max teams in a conference
    pos_rows = []
    for team in all_teams:
        row = {"team": team}
        for pos in range(1, max_pos + 1):
            row[f"pos_{pos}"] = team_position_counts[team].get(pos, 0) / n_sims * 100
        pos_rows.append(row)

    position_dist_df = pd.DataFrame(pos_rows)

    logger.info("Detailed season simulation complete. %d teams processed.", len(summary_df))
    return summary_df, position_dist_df


# ═══════════════════════════════════════════════════════════════════════════
#  Playoff Bracket Simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_best_of_three(
    prob_higher_seed_wins: float,
    home_advantage: float = 0.05,
) -> bool:
    """
    Simulate an MLS Round 1 best-of-3 series where the higher seed hosts Games 1 & 3.
    Returns True if higher seed wins the series.
    """
    wins_higher = 0
    wins_lower = 0

    for game in range(3):
        if wins_higher == 2 or wins_lower == 2:
            break
        # Games 0 and 2 are at higher seed's home venue
        if game in [0, 2]:
            p = prob_higher_seed_wins + home_advantage
        else:
            p = prob_higher_seed_wins - home_advantage
        p = np.clip(p, 0.05, 0.95)

        if np.random.random() < p:
            wins_higher += 1
        else:
            wins_lower += 1

    return wins_higher == 2


def simulate_single_knockout(
    elo_higher: float,
    elo_lower: float,
    home_advantage_elo: float = 65.0,
) -> bool:
    """
    Simulate a single-elimination knockout match (Wild Card, Conf Semis, Conf Finals).
    Returns True if higher seed wins.
    """
    from mls_predictor.elo import expected_score
    p_higher = expected_score(elo_higher + home_advantage_elo, elo_lower)
    return np.random.random() < p_higher


def simulate_playoff_bracket(
    seeded_teams: list[str],
    elo_ratings: dict[str, float],
    n_sims: int = MC_SIMULATIONS,
) -> pd.DataFrame:
    """
    Simulate the MLS Cup Playoffs for one conference under official regulations:
    - Wild Card (8 vs 9): Single Elimination (8 hosts 9)
    - Round 1 (1 vs WC, 2 vs 7, 3 vs 6, 4 vs 5): Best-of-3 Series
    - Conf. Semifinals: Single Elimination (Higher seed hosts)
    - Conf. Final: Single Elimination (Higher seed hosts)

    Parameters
    ----------
    seeded_teams : list[str]
        Teams ranked 1 through 9 by conference standing.
    elo_ratings : dict[str, float]
        Current Elo for each team.
    n_sims : int
        Number of simulations.

    Returns
    -------
    pd.DataFrame with columns: team, round1_pct, conf_semi_pct, conf_final_pct, champion_pct
    """
    from mls_predictor.elo import expected_score

    champion_counts = defaultdict(int)
    conf_final_counts = defaultdict(int)
    conf_semi_counts = defaultdict(int)
    round1_counts = defaultdict(int)

    for _ in range(n_sims):
        teams = list(seeded_teams[:9])  # top 9 qualify

        if len(teams) < 9:
            continue

        # ── 1. Wild Card: Seed 8 vs Seed 9 (single game, 8 hosts) ──
        wc_higher = teams[7]  # seed 8
        wc_lower = teams[8]   # seed 9
        elo_8 = elo_ratings.get(wc_higher, 1500)
        elo_9 = elo_ratings.get(wc_lower, 1500)
        wc_winner = wc_higher if simulate_single_knockout(elo_8, elo_9) else wc_lower

        # ── 2. Round 1 (Best of 3 series): 4 matchups ──
        # 1 vs WC winner, 2 vs 7, 3 vs 6, 4 vs 5
        r1_matchup_teams = [
            (teams[0], wc_winner),     # seed 1 vs WC winner
            (teams[1], teams[6]),       # seed 2 vs seed 7
            (teams[2], teams[5]),       # seed 3 vs seed 6
            (teams[3], teams[4]),       # seed 4 vs seed 5
        ]

        r1_winners = []
        for higher, lower in r1_matchup_teams:
            round1_counts[higher] += 1
            round1_counts[lower] += 1

            elo_h = elo_ratings.get(higher, 1500)
            elo_l = elo_ratings.get(lower, 1500)
            p_higher = expected_score(elo_h, elo_l)

            if simulate_best_of_three(p_higher):
                r1_winners.append(higher)
            else:
                r1_winners.append(lower)

        # ── 3. Conference Semifinals: Single Elimination (Higher seed hosts) ──
        if len(r1_winners) >= 4:
            # Fixed bracket: Winner of (1/WC) plays Winner of (4/5); Winner of (2/7) plays Winner of (3/6)
            semi_pairs = [(r1_winners[0], r1_winners[3]), (r1_winners[1], r1_winners[2])]
            semi_winners = []
            for t_a, t_b in semi_pairs:
                conf_semi_counts[t_a] += 1
                conf_semi_counts[t_b] += 1

                # Higher seed from regular season hosts
                seed_a = seeded_teams.index(t_a) if t_a in seeded_teams else 99
                seed_b = seeded_teams.index(t_b) if t_b in seeded_teams else 99
                higher = t_a if seed_a <= seed_b else t_b
                lower = t_b if higher == t_a else t_a

                elo_h = elo_ratings.get(higher, 1500)
                elo_l = elo_ratings.get(lower, 1500)

                winner = higher if simulate_single_knockout(elo_h, elo_l) else lower
                semi_winners.append(winner)

            # ── 4. Conference Final: Single Elimination (Higher seed hosts) ──
            if len(semi_winners) >= 2:
                t1, t2 = semi_winners[0], semi_winners[1]
                conf_final_counts[t1] += 1
                conf_final_counts[t2] += 1

                seed_1 = seeded_teams.index(t1) if t1 in seeded_teams else 99
                seed_2 = seeded_teams.index(t2) if t2 in seeded_teams else 99
                higher = t1 if seed_1 <= seed_2 else t2
                lower = t2 if higher == t1 else t1

                elo_h = elo_ratings.get(higher, 1500)
                elo_l = elo_ratings.get(lower, 1500)

                champ = higher if simulate_single_knockout(elo_h, elo_l) else lower
                champion_counts[champ] += 1

    results = []
    for team in seeded_teams:
        results.append({
            "team": team,
            "round1_pct": round1_counts.get(team, 0) / n_sims * 100,
            "conf_semi_pct": conf_semi_counts.get(team, 0) / n_sims * 100,
            "conf_final_pct": conf_final_counts.get(team, 0) / n_sims * 100,
            "champion_pct": champion_counts.get(team, 0) / n_sims * 100,
        })

    return pd.DataFrame(results).sort_values("champion_pct", ascending=False).reset_index(drop=True)


def compute_current_standings(df: pd.DataFrame, season: int) -> dict:
    """
    Compute current standings from match data for a given season.
    Returns {team: {"points": int, "gd": int, "gf": int, "played": int, "w": int, "d": int, "l": int}}
    """
    season_df = df[df["Season"] == season].copy()
    teams = set(season_df["HomeTeam"].unique()) | set(season_df["AwayTeam"].unique())

    standings = {}
    for team in teams:
        home = season_df[season_df["HomeTeam"] == team]
        away = season_df[season_df["AwayTeam"] == team]

        hw = len(home[home["FTR"] == "H"])
        hd = len(home[home["FTR"] == "D"])
        hl = len(home[home["FTR"] == "A"])

        aw = len(away[away["FTR"] == "A"])
        ad = len(away[away["FTR"] == "D"])
        al = len(away[away["FTR"] == "H"])

        gf = home["FTHG"].sum() + away["FTAG"].sum()
        ga = home["FTAG"].sum() + away["FTHG"].sum()

        standings[team] = {
            "points": (hw + aw) * 3 + (hd + ad),
            "gd": int(gf - ga),
            "gf": int(gf),
            "ga": int(ga),
            "played": len(home) + len(away),
            "w": hw + aw,
            "d": hd + ad,
            "l": hl + al,
        }

    return standings
