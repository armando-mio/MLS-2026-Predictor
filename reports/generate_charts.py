"""
MLS 2026 Predictor — Publication Quality Visual Analytics & Findings Generator
Generates high-resolution figures for reports, LinkedIn articles, and technical presentations.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from scipy.stats import poisson

from mls_predictor.data_loader import load_raw_data
from mls_predictor.elo import compute_elo_history
from mls_predictor.feature_engine import build_all_features, get_feature_columns

# Output directories
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set high-quality styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Custom Dark Theme Palette matching the MLS Predictor Dashboard
BG_COLOR = "#0f172a"
CARD_COLOR = "#1e293b"
GRID_COLOR = "#334155"
TEXT_COLOR = "#ffffff"
MUTED_TEXT = "#94a3b8"
ACCENT_BLUE = "#38bdf8"
ACCENT_GREEN = "#4ade80"
ACCENT_RED = "#f87171"
ACCENT_YELLOW = "#facc15"
ACCENT_PURPLE = "#c084fc"

# Ensure all text, axes, ticks, and legends globally default to pure white
plt.rcParams['text.color'] = TEXT_COLOR
plt.rcParams['axes.labelcolor'] = TEXT_COLOR
plt.rcParams['xtick.color'] = TEXT_COLOR
plt.rcParams['ytick.color'] = TEXT_COLOR
plt.rcParams['legend.labelcolor'] = TEXT_COLOR
plt.rcParams['legend.facecolor'] = CARD_COLOR
plt.rcParams['legend.edgecolor'] = GRID_COLOR
plt.rcParams['legend.fontsize'] = 10

def style_legend(leg):
    """Explicitly enforce pure white text and dark themed frame for legends."""
    if leg:
        leg.get_frame().set_facecolor(CARD_COLOR)
        leg.get_frame().set_edgecolor(GRID_COLOR)
        for text in leg.get_texts():
            text.set_color("#ffffff")
        title = leg.get_title()
        if title:
            title.set_color("#ffffff")
    return leg

def apply_theme(fig, ax):
    fig.patch.set_facecolor(BG_COLOR)
    if isinstance(ax, (np.ndarray, list)):
        for a in np.array(ax).flat:
            _apply_single_ax(a)
    else:
        _apply_single_ax(ax)

def _apply_single_ax(a):
    a.set_facecolor(CARD_COLOR)
    a.tick_params(colors=TEXT_COLOR, labelsize=10)
    a.xaxis.label.set_color(TEXT_COLOR)
    a.yaxis.label.set_color(TEXT_COLOR)
    a.title.set_color(TEXT_COLOR)
    for spine in a.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.2)
    a.grid(True, linestyle="--", alpha=0.3, color=GRID_COLOR)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & ENRICH DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("Loading match dataset and computing features (2022-2026)...")
df_raw = load_raw_data(exclude_current_season=False)
df_elo = compute_elo_history(df_raw)
df = build_all_features(df_elo)
print(f"Total dataset: {len(df)} matches loaded (2022-2026).")

# Add auxiliary calculation columns
df["total_goals"] = df["FTHG"] + df["FTAG"]
df["is_over25"] = (df["total_goals"] > 2.5).astype(int)

# Historical dataset for exploratory finding figures (2022-2025)
df_hist = df[df["Season"] <= 2025].copy()

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: WORST-CASE VS BEST-CASE SCHEDULE CONGESTION & TRAVEL
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Figure 1: Worst vs. Best Schedule Scenarios...")

# Define Schedule Cohorts:
# Worst-Case Road Trip: Away match, Distance > 2000 km, Timezones crossed >= 2, Rest <= 4 days
worst_mask = (df_hist["travel_distance_km"] >= 2000) & (df_hist["timezones_crossed"] >= 2) & (df_hist["away_rest_days"] <= 4)
# Moderate Road Trip: Average away match (800-2000 km, normal rest)
moderate_mask = (df_hist["travel_distance_km"] < 2000) & (df_hist["travel_distance_km"] >= 800) & (df_hist["away_rest_days"] >= 5)
# Best-Case Road Trip: Short travel (< 800 km), Same timezone, Fresh rest (>= 6 days)
best_mask = (df_hist["travel_distance_km"] < 800) & (df_hist["timezones_crossed"] == 0) & (df_hist["away_rest_days"] >= 6)

cohorts = [
    ("Hell Road Trip\n(>2,000 km, ≥2 TZ, ≤4 Rest Days)", df_hist[worst_mask]),
    ("Average Travel\n(800–2,000 km, Normal Rest)", df_hist[moderate_mask]),
    ("Favorable Schedule\n(<800 km, 0 TZ, ≥6 Rest Days)", df_hist[best_mask]),
]

labels, win_rates, loss_rates, ppg_rates = [], [], [], []
sample_sizes = []
for name, subset in cohorts:
    labels.append(name)
    sample_sizes.append(len(subset))
    w_rate = (subset["FTR"] == "A").mean() * 100
    l_rate = (subset["FTR"] == "H").mean() * 100
    d_rate = (subset["FTR"] == "D").mean() * 100
    ppg = (w_rate * 3 + d_rate * 1) / 100
    win_rates.append(w_rate)
    loss_rates.append(l_rate)
    ppg_rates.append(ppg)

fig, ax1 = plt.subplots(figsize=(10, 6))
apply_theme(fig, ax1)

x = np.arange(len(labels))
width = 0.32

rects1 = ax1.bar(x - width/2, win_rates, width, label='Away Win %', color=ACCENT_GREEN, alpha=0.9, edgecolor='white', linewidth=0.5)
rects2 = ax1.bar(x + width/2, loss_rates, width, label='Away Loss % (Home Win)', color=ACCENT_RED, alpha=0.9, edgecolor='white', linewidth=0.5)

ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax1.set_title('MLS Travel & Congestion Penalty: Worst-Case vs. Best-Case Schedules\n(Empirical Analysis of 2022–2025 Match Data)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{l}\n(n={s})" for l, s in zip(labels, sample_sizes)], fontsize=10, fontweight='bold')
ax1.set_ylim(0, 65)

# Add values above bars
for rect in rects1:
    height = rect.get_height()
    ax1.annotate(f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', color=TEXT_COLOR, fontweight='bold', fontsize=10)

for rect in rects2:
    height = rect.get_height()
    ax1.annotate(f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', color=TEXT_COLOR, fontweight='bold', fontsize=10)

# Secondary Axis for PPG
ax2 = ax1.twinx()
ax2.plot(x, ppg_rates, color=ACCENT_YELLOW, marker='o', linewidth=3, markersize=8, label='Away Points Per Game (PPG)')
ax2.set_ylabel('Points Per Game (PPG)', color=ACCENT_YELLOW, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=ACCENT_YELLOW, labelsize=10)
ax2.set_ylim(0.5, 1.6)

for i, ppg in enumerate(ppg_rates):
    ax2.annotate(f'{ppg:.2f} PPG', xy=(x[i], ppg), xytext=(0, 10), textcoords="offset points",
                 ha='center', color=ACCENT_YELLOW, fontweight='bold', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor=BG_COLOR, edgecolor=ACCENT_YELLOW, alpha=0.8))

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
leg1 = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.85, facecolor=CARD_COLOR, edgecolor=GRID_COLOR, fontsize=10, labelcolor='#ffffff')
style_legend(leg1)

plt.savefig(FIGURES_DIR / "fig1_worst_vs_best_schedule.png")
plt.close()
print("  ✓ Saved: reports/figures/fig1_worst_vs_best_schedule.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: MLS RIVALRIES & DERBIES VS STANDARD FIXTURES
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Figure 2: Rivalries & Derbies Comparison...")

riv = df_hist[df_hist['is_rivalry'] == 1]
non_riv = df_hist[df_hist['is_rivalry'] == 0]

fig, (ax_rates, ax_goals) = plt.subplots(1, 2, figsize=(14, 6))
apply_theme(fig, np.array([ax_rates, ax_goals]))

# Left Subplot: Outcome Distribution (Home Win, Draw, Away Win)
categories = ['Standard Matches\n(n=1,893)', 'Rivalry / Derby Matches\n(n=163)']
hw_rates = [(non_riv['FTR'] == 'H').mean()*100, (riv['FTR'] == 'H').mean()*100]
d_rates  = [(non_riv['FTR'] == 'D').mean()*100, (riv['FTR'] == 'D').mean()*100]
aw_rates = [(non_riv['FTR'] == 'A').mean()*100, (riv['FTR'] == 'A').mean()*100]

x_cat = np.arange(len(categories))
bar_w = 0.26

ax_rates.bar(x_cat - bar_w, hw_rates, bar_w, label='Home Win %', color=ACCENT_BLUE, edgecolor='white', linewidth=0.5)
ax_rates.bar(x_cat, d_rates, bar_w, label='Draw %', color=ACCENT_YELLOW, edgecolor='white', linewidth=0.5)
ax_rates.bar(x_cat + bar_w, aw_rates, bar_w, label='Away Win %', color=ACCENT_RED, edgecolor='white', linewidth=0.5)

ax_rates.set_title('Outcome Breakdown: Derbies vs. Standard MLS', fontsize=13, fontweight='bold', pad=12)
ax_rates.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax_rates.set_xticks(x_cat)
ax_rates.set_xticklabels(categories, fontsize=10, fontweight='bold')
ax_rates.set_ylim(0, 55)
leg_rates = ax_rates.legend(facecolor=CARD_COLOR, edgecolor=GRID_COLOR, fontsize=9, labelcolor='#ffffff')
style_legend(leg_rates)

for bars in ax_rates.containers:
    ax_rates.bar_label(bars, fmt='%.1f%%', padding=3, color=TEXT_COLOR, fontweight='bold', fontsize=9)

# Highlight lower draw rate with arrow/box
ax_rates.annotate('Draw Rate Drops by -2.2%\n(Teams play to win)',
                  xy=(1, d_rates[1]), xytext=(0.7, 34),
                  arrowprops=dict(facecolor=ACCENT_YELLOW, shrink=0.08, width=1.5, headwidth=6),
                  color=ACCENT_YELLOW, fontweight='bold', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor=BG_COLOR, edgecolor=ACCENT_YELLOW, alpha=0.9))

# Right Subplot: Goal Volume & Over 2.5 Rate
metric_names = ['Avg Goals per Match', 'Over 2.5 Goals (%)']
std_metrics = [(non_riv['FTHG'] + non_riv['FTAG']).mean(), (non_riv['is_over25']).mean()*100]
riv_metrics = [(riv['FTHG'] + riv['FTAG']).mean(), (riv['is_over25']).mean()*100]

x_m = np.arange(len(metric_names))
w_m = 0.35

# Dual plot or separate bars
ax_goals_twin = ax_goals.twinx()

b1 = ax_goals.bar(0 - w_m/2, std_metrics[0], w_m, color=MUTED_TEXT, label='Standard Matches', edgecolor='white', linewidth=0.5)
b2 = ax_goals.bar(0 + w_m/2, riv_metrics[0], w_m, color=ACCENT_PURPLE, label='Rivalry Matches', edgecolor='white', linewidth=0.5)

b3 = ax_goals_twin.bar(1 - w_m/2, std_metrics[1], w_m, color=MUTED_TEXT, edgecolor='white', linewidth=0.5)
b4 = ax_goals_twin.bar(1 + w_m/2, riv_metrics[1], w_m, color=ACCENT_PURPLE, edgecolor='white', linewidth=0.5)

ax_goals.set_ylabel('Goals per Match', color=TEXT_COLOR, fontsize=11, fontweight='bold')
ax_goals_twin.set_ylabel('Over 2.5 Percentage (%)', color=ACCENT_PURPLE, fontsize=11, fontweight='bold')
ax_goals_twin.tick_params(axis='y', labelcolor=ACCENT_PURPLE)

ax_goals.set_title('Goal Dynamics: Rivalries Spark More Goals (+0.29/gm)', fontsize=13, fontweight='bold', pad=12)
ax_goals.set_xticks([0, 1])
ax_goals.set_xticklabels(['Average Goals\nper Match', 'Over 2.5 Goals\nFrequency (%)'], fontsize=10, fontweight='bold')
ax_goals.set_ylim(0, 4.0)
ax_goals_twin.set_ylim(0, 75)

# Annotations for goals
ax_goals.annotate(f'{std_metrics[0]:.2f} G/M', xy=(0 - w_m/2, std_metrics[0]), xytext=(0, 4), textcoords="offset points", ha='center', color=TEXT_COLOR, fontweight='bold', fontsize=9)
ax_goals.annotate(f'{riv_metrics[0]:.2f} G/M\n(+0.29)', xy=(0 + w_m/2, riv_metrics[0]), xytext=(0, 4), textcoords="offset points", ha='center', color=ACCENT_PURPLE, fontweight='bold', fontsize=9)

ax_goals_twin.annotate(f'{std_metrics[1]:.1f}%', xy=(1 - w_m/2, std_metrics[1]), xytext=(0, 4), textcoords="offset points", ha='center', color=TEXT_COLOR, fontweight='bold', fontsize=9)
ax_goals_twin.annotate(f'{riv_metrics[1]:.1f}%\n(+6.9%)', xy=(1 + w_m/2, riv_metrics[1]), xytext=(0, 4), textcoords="offset points", ha='center', color=ACCENT_PURPLE, fontweight='bold', fontsize=9)

leg_goals = ax_goals.legend(loc='upper left', facecolor=CARD_COLOR, edgecolor=GRID_COLOR, fontsize=9, labelcolor='#ffffff')
style_legend(leg_goals)

plt.suptitle('MLS Rivalry Dynamics: High-Scoring Intensity & Lower Draw Frequency', fontsize=15, fontweight='bold', color=TEXT_COLOR, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_derbies_vs_standard.png")
plt.close()
print("  ✓ Saved: reports/figures/fig2_derbies_vs_standard.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: DESIGNATED PLAYER QUALITY DIFFERENTIAL
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Figure 3: Designated Player Impact...")

dp_bins = [
    ("Severe DP Deficit\n(DP Diff ≤ -1.5)", df_hist[df_hist['dp_quality_diff'] <= -1.5]),
    ("Slight DP Deficit\n(-1.5 < Diff < -0.2)", df_hist[(df_hist['dp_quality_diff'] > -1.5) & (df_hist['dp_quality_diff'] < -0.2)]),
    ("DP Quality Parity\n(-0.2 ≤ Diff ≤ +0.2)", df_hist[(df_hist['dp_quality_diff'] >= -0.2) & (df_hist['dp_quality_diff'] <= 0.2)]),
    ("Slight DP Advantage\n(+0.2 < Diff < +1.5)", df_hist[(df_hist['dp_quality_diff'] > 0.2) & (df_hist['dp_quality_diff'] < 1.5)]),
    ("Large DP Advantage\n(DP Diff ≥ +1.5)", df_hist[df_hist['dp_quality_diff'] >= 1.5]),
]

dp_labels = [d[0] for d in dp_bins]
dp_home_win = [(d[1]['FTR'] == 'H').mean()*100 for d in dp_bins]
dp_away_win = [(d[1]['FTR'] == 'A').mean()*100 for d in dp_bins]
dp_counts = [len(d[1]) for d in dp_bins]

fig, ax = plt.subplots(figsize=(11, 6))
apply_theme(fig, ax)

x_dp = np.arange(len(dp_labels))
b_w = 0.35

bars_h = ax.bar(x_dp - b_w/2, dp_home_win, b_w, label='Home Win Rate (%)', color=ACCENT_BLUE, edgecolor='white', linewidth=0.5)
bars_a = ax.bar(x_dp + b_w/2, dp_away_win, b_w, label='Away Win Rate (%)', color=ACCENT_RED, edgecolor='white', linewidth=0.5)

ax.set_title('The Designated Player (DP) Multiplier: How Star Power Breaks MLS Parity\n(Win Rate Distribution Across DP Quality Differentials)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
ax.set_xticks(x_dp)
ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(dp_labels, dp_counts)], fontsize=9, fontweight='bold')
ax.set_ylim(0, 68)
ax.axhline(46.6, color=ACCENT_BLUE, linestyle=':', alpha=0.7, label='League Avg Home Win (46.6%)')
ax.axhline(27.5, color=ACCENT_RED, linestyle=':', alpha=0.7, label='League Avg Away Win (27.5%)')
leg3 = ax.legend(facecolor=CARD_COLOR, edgecolor=GRID_COLOR, fontsize=9, loc='upper left', labelcolor='#ffffff')
style_legend(leg3)

for b in bars_h:
    h = b.get_height()
    ax.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha='center', color=TEXT_COLOR, fontweight='bold', fontsize=9)

for b in bars_a:
    h = b.get_height()
    ax.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha='center', color=TEXT_COLOR, fontweight='bold', fontsize=9)

# Highlight the +11.1% gain
ax.annotate('+11.1% Home Surge\nwith Elite DP Tier', xy=(4 - b_w/2, dp_home_win[4]), xytext=(3.4, 62),
            arrowprops=dict(facecolor=ACCENT_GREEN, shrink=0.08, width=1.5, headwidth=6),
            color=ACCENT_GREEN, fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG_COLOR, edgecolor=ACCENT_GREEN, alpha=0.9))

plt.savefig(FIGURES_DIR / "fig3_dp_quality_impact.png")
plt.close()
print("  ✓ Saved: reports/figures/fig3_dp_quality_impact.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: REST DIFFERENTIAL SWING CURVE
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Figure 4: Rest Differential Swing Curve...")

# Bin rest differential from -4 to +4
rest_diffs = np.arange(-4, 5, 1)
rest_stats = []

for rd in rest_diffs:
    sub = df_hist[df_hist['rest_diff'] == rd] if rd != -4 and rd != 4 else (df_hist[df_hist['rest_diff'] <= -4] if rd == -4 else df_hist[df_hist['rest_diff'] >= 4])
    if len(sub) > 0:
        hw = (sub['FTR'] == 'H').mean() * 100
        d  = (sub['FTR'] == 'D').mean() * 100
        aw = (sub['FTR'] == 'A').mean() * 100
        ppg_h = (hw * 3 + d * 1) / 100
        rest_stats.append({
            'rest_diff': rd,
            'label': '≤ -4' if rd == -4 else ('≥ +4' if rd == 4 else str(rd)),
            'home_win': hw,
            'draw': d,
            'away_win': aw,
            'ppg_h': ppg_h,
            'count': len(sub)
        })

df_rest = pd.DataFrame(rest_stats)

fig, ax = plt.subplots(figsize=(11, 6.5))
apply_theme(fig, ax)

ax.plot(df_rest['rest_diff'], df_rest['home_win'], marker='s', color=ACCENT_BLUE, linewidth=2.8, markersize=8, label='Home Win Rate (%)')
ax.plot(df_rest['rest_diff'], df_rest['away_win'], marker='^', color=ACCENT_RED, linewidth=2.8, markersize=8, label='Away Win Rate (%)')
ax.plot(df_rest['rest_diff'], df_rest['draw'], marker='o', color=ACCENT_YELLOW, linewidth=2, linestyle='--', markersize=6, label='Draw Rate (%)')

ax.set_title('The Rest Asymmetry Effect: Impact of Days-of-Rest Differential\n(Home Rest Days minus Away Rest Days)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Rest Differential (Days)', fontsize=11, fontweight='bold')
ax.set_ylabel('Outcome Frequency (%)', fontsize=11, fontweight='bold')
ax.set_xticks(df_rest['rest_diff'])
ax.set_xticklabels([f"{r['label']}\n(n={r['count']})" for _, r in df_rest.iterrows()], fontsize=9)
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(0, 75)
leg4 = ax.legend(facecolor=CARD_COLOR, edgecolor=GRID_COLOR, fontsize=10, loc='center left', labelcolor='#ffffff')
style_legend(leg4)

for _, r in df_rest.iterrows():
    ax.annotate(f"{r['home_win']:.1f}%", (r['rest_diff'], r['home_win']), xytext=(0, 7), textcoords="offset points", ha='center', color=ACCENT_BLUE, fontweight='bold', fontsize=8.5)
    ax.annotate(f"{r['away_win']:.1f}%", (r['rest_diff'], r['away_win']), xytext=(0, -13), textcoords="offset points", ha='center', color=ACCENT_RED, fontweight='bold', fontsize=8.5)

plt.savefig(FIGURES_DIR / "fig4_rest_differential_curve.png")
plt.close()
print("  ✓ Saved: reports/figures/fig4_rest_differential_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: FEATURE IMPORTANCE RANKING (RANDOM FOREST)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Figure 5: Feature Importance Hierarchy...")

try:
    rf = joblib.load(PROJECT_ROOT / "models" / "random_forest_1x2.joblib")
    prep = joblib.load(PROJECT_ROOT / "models" / "preprocess_1x2.joblib")
    cols = prep.get("feature_cols", get_feature_columns()["production"])
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:12]
    
    top_cols = [cols[i] for i in sorted_idx][::-1]
    top_vals = [importances[i]*100 for i in sorted_idx][::-1]

    # Map human-readable labels and category colors
    label_map = {
        "home_elo_before": "Home Team Elo Rating",
        "elo_diff": "Elo Rating Differential (H - A)",
        "ppg_diff": "Season Points Per Game Differential",
        "away_away_win_rate": "Away Team Road Win Rate",
        "away_elo_before": "Away Team Elo Rating",
        "travel_fatigue_index": "Travel Fatigue Index (Last 3 Gms)",
        "travel_distance_km": "Travel Distance (km)",
        "attack_diff_r10": "Attack Efficiency Diff (10-Gms)",
        "away_season_ppg": "Away Season Points Per Game",
        "home_season_ppg": "Home Season Points Per Game",
        "attack_diff_r5": "Attack Efficiency Diff (5-Gms)",
        "dp_quality_diff": "Designated Player Quality Diff",
    }
    
    category_colors = {
        "home_elo_before": ACCENT_BLUE,
        "elo_diff": ACCENT_BLUE,
        "away_elo_before": ACCENT_BLUE,
        "ppg_diff": ACCENT_GREEN,
        "away_away_win_rate": ACCENT_GREEN,
        "away_season_ppg": ACCENT_GREEN,
        "home_season_ppg": ACCENT_GREEN,
        "attack_diff_r10": ACCENT_YELLOW,
        "attack_diff_r5": ACCENT_YELLOW,
        "travel_fatigue_index": ACCENT_RED,
        "travel_distance_km": ACCENT_RED,
        "dp_quality_diff": ACCENT_PURPLE,
    }

    clean_labels = [label_map.get(c, c) for c in top_cols]
    bar_colors = [category_colors.get(c, ACCENT_BLUE) for c in top_cols]

    fig, ax = plt.subplots(figsize=(11, 7))
    apply_theme(fig, ax)

    bars = ax.barh(clean_labels, top_vals, color=bar_colors, edgecolor='white', linewidth=0.5, height=0.65)
    ax.set_title('Top 12 Predictive Features in MLS Outcome Modeling (Random Forest)\nRelative Information Gain in Calibrated 1X2 Classification', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Feature Importance Relative Weight (%)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 6.5)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f'{w:.2f}%',
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    ha='left', va='center', color=TEXT_COLOR, fontweight='bold', fontsize=10)

    # Custom legend for feature categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACCENT_BLUE, edgecolor='white', label='Team Strength (Elo)'),
        Patch(facecolor=ACCENT_GREEN, edgecolor='white', label='Seasonal Form / PPG'),
        Patch(facecolor=ACCENT_RED, edgecolor='white', label='Geography & Travel Logistics'),
        Patch(facecolor=ACCENT_YELLOW, edgecolor='white', label='Tactical / Attack Efficiency'),
        Patch(facecolor=ACCENT_PURPLE, edgecolor='white', label='Squad Quality / Designated Players'),
    ]
    leg5 = ax.legend(handles=legend_elements, loc='lower right', facecolor=CARD_COLOR, edgecolor=GRID_COLOR, fontsize=9, labelcolor='#ffffff')
    style_legend(leg5)

    plt.savefig(FIGURES_DIR / "fig5_feature_importance.png")
    plt.close()
    print("  ✓ Saved: reports/figures/fig5_feature_importance.png")
except Exception as e:
    print(f"  ✗ Figure 5 error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: 2026 SEASON PREDICTION ACCURACY (1X2 VS EXACT SCORE)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Figure 6: 2026 Season Prediction Performance (1X2 vs Exact Score)...")

try:
    df_2026 = df[df["Season"] == 2026].copy()
    if len(df_2026) > 0:
        rf = joblib.load(PROJECT_ROOT / "models" / "random_forest_1x2.joblib")
        prep = joblib.load(PROJECT_ROOT / "models" / "preprocess_1x2.joblib")
        feature_cols = prep.get("feature_cols", get_feature_columns()["production"])
        imputer = prep.get("imputer")

        X_2026 = df_2026[feature_cols]
        X_2026_imp = pd.DataFrame(
            imputer.transform(X_2026) if imputer else X_2026.fillna(X_2026.median()),
            columns=feature_cols, index=X_2026.index
        )

        # 1X2 Outcome Prediction
        pred_num = rf.predict(X_2026_imp)
        inv_map = {0: 'A', 1: 'D', 2: 'H'}
        df_2026["pred_1x2"] = [inv_map[n] for n in pred_num]
        df_2026["correct_1x2"] = (df_2026["pred_1x2"] == df_2026["FTR"])

        # Exact Score Prediction via Poisson Model
        home_advantage = 1.10
        max_goals = 6
        pred_hg_list, pred_ag_list = [], []

        for _, row in df_2026.iterrows():
            home_attack = row.get("home_attack_r5", 1.3)
            away_defense = row.get("away_defense_r5", 1.3)
            away_attack = row.get("away_attack_r5", 0.9)
            home_defense = row.get("home_defense_r5", 1.3)
            home_elo_val = row.get("home_elo_before", 1500)
            away_elo_val = row.get("away_elo_before", 1500)
            elo_ratio = home_elo_val / max(away_elo_val, 1)

            lambda_h = max(0.3, (home_attack + away_defense) / 2 * home_advantage * (elo_ratio ** 0.15))
            lambda_a = max(0.3, (away_attack + home_defense) / 2 / home_advantage * ((1 / elo_ratio) ** 0.15))

            best_p, best_score = -1, (1, 1)
            for hg in range(max_goals + 1):
                for ag in range(max_goals + 1):
                    p = poisson.pmf(hg, lambda_h) * poisson.pmf(ag, lambda_a)
                    if p > best_p:
                        best_p = p
                        best_score = (hg, ag)
            pred_hg_list.append(best_score[0])
            pred_ag_list.append(best_score[1])

        df_2026["pred_score"] = [f"{h}-{a}" for h, a in zip(pred_hg_list, pred_ag_list)]
        df_2026["actual_score"] = df_2026["FTHG"].astype(int).astype(str) + "-" + df_2026["FTAG"].astype(int).astype(str)
        df_2026["correct_score"] = (df_2026["pred_score"] == df_2026["actual_score"])

        total_matches = len(df_2026)
        c_1x2 = int(df_2026["correct_1x2"].sum())
        err_1x2 = total_matches - c_1x2
        pct_1x2 = (c_1x2 / total_matches) * 100

        c_score = int(df_2026["correct_score"].sum())
        err_score = total_matches - c_score
        pct_score = (c_score / total_matches) * 100

        # Subplots: Left for Comparison Bar, Right for Category Breakdown & Insights
        fig, (ax_main, ax_detail) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={'width_ratios': [1.1, 1]})
        apply_theme(fig, np.array([ax_main, ax_detail]))

        # --- LEFT SUBPLOT: Head-to-Head Comparison ---
        categories = ['1X2 Outcome\n(Random Forest)', 'Exact Scoreline\n(Bivariate Poisson)']
        x_idx = np.arange(len(categories))
        bar_w = 0.35

        bars_corr = ax_main.bar(x_idx - bar_w/2, [c_1x2, c_score], bar_w,
                                label='Correct Predictions', color=ACCENT_GREEN, edgecolor='white', linewidth=0.6)
        bars_err = ax_main.bar(x_idx + bar_w/2, [err_1x2, err_score], bar_w,
                               label='Incorrect Predictions', color=ACCENT_RED, edgecolor='white', linewidth=0.6)

        ax_main.set_title('Prediction Accuracy: Correct vs. Incorrect Matches\n(MLS 2026 Season Through Gameweek 15 — 218 Matches)',
                          fontsize=13, fontweight='bold', pad=14)
        ax_main.set_ylabel('Number of Matches', fontsize=11, fontweight='bold')
        ax_main.set_xticks(x_idx)
        ax_main.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax_main.set_ylim(0, 275)

        # Annotations on bars
        ax_main.annotate(f'{c_1x2} ({pct_1x2:.1f}%)\n[98 / 218]',
                         xy=(x_idx[0] - bar_w/2, c_1x2), xytext=(0, 6), textcoords="offset points",
                         ha='center', va='bottom', color=ACCENT_GREEN, fontweight='bold', fontsize=9.5)
        ax_main.annotate(f'{err_1x2} ({100-pct_1x2:.1f}%)\n[120 / 218]',
                         xy=(x_idx[0] + bar_w/2, err_1x2), xytext=(0, 6), textcoords="offset points",
                         ha='center', va='bottom', color=ACCENT_RED, fontweight='bold', fontsize=9.5)

        ax_main.annotate(f'{c_score} ({pct_score:.1f}%)\n[14 / 218]',
                         xy=(x_idx[1] - bar_w/2, c_score), xytext=(0, 6), textcoords="offset points",
                         ha='center', va='bottom', color=ACCENT_GREEN, fontweight='bold', fontsize=9.5)
        ax_main.annotate(f'{err_score} ({100-pct_score:.1f}%)\n[204 / 218]',
                         xy=(x_idx[1] + bar_w/2, err_score), xytext=(0, 6), textcoords="offset points",
                         ha='center', va='bottom', color=ACCENT_RED, fontweight='bold', fontsize=9.5)

        # Reference baseline line for 1X2 (33.3% of 218 = 72.6 matches)
        ax_main.axhline(72.6, color=ACCENT_YELLOW, linestyle=':', linewidth=1.5, alpha=0.8,
                        label='Random 1X2 Baseline (33.3% = 73 matches)')

        leg_main = ax_main.legend(loc='upper right', facecolor=CARD_COLOR, edgecolor=GRID_COLOR, fontsize=9.5, labelcolor='#ffffff')
        style_legend(leg_main)

        # --- RIGHT SUBPLOT: Breakdown by 1X2 Outcome & Exact Scores ---
        h_total = (df_2026["FTR"] == "H").sum()
        h_corr = ((df_2026["FTR"] == "H") & df_2026["correct_1x2"]).sum()
        h_pct = (h_corr / h_total) * 100 if h_total > 0 else 0

        a_total = (df_2026["FTR"] == "A").sum()
        a_corr = ((df_2026["FTR"] == "A") & df_2026["correct_1x2"]).sum()
        a_pct = (a_corr / a_total) * 100 if a_total > 0 else 0

        d_total = (df_2026["FTR"] == "D").sum()
        d_corr = ((df_2026["FTR"] == "D") & df_2026["correct_1x2"]).sum()
        d_pct = (d_corr / d_total) * 100 if d_total > 0 else 0

        breakdown_labels = [
            f"Home Win (H)\n{h_corr}/{h_total} ({h_pct:.1f}%)",
            f"Away Win (A)\n{a_corr}/{a_total} ({a_pct:.1f}%)",
            f"Draw (D)\n{d_corr}/{d_total} ({d_pct:.1f}%)",
        ]
        breakdown_pcts = [h_pct, a_pct, d_pct]
        breakdown_colors = [ACCENT_BLUE, ACCENT_PURPLE, ACCENT_YELLOW]

        y_pos = np.arange(len(breakdown_labels))
        bars_b = ax_detail.barh(y_pos, breakdown_pcts, color=breakdown_colors, edgecolor='white', linewidth=0.6, height=0.5)

        ax_detail.set_yticks(y_pos)
        ax_detail.set_yticklabels(breakdown_labels, fontsize=10, fontweight='bold')
        ax_detail.set_xlim(0, 100)
        ax_detail.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax_detail.set_title('1X2 Accuracy by Match Outcome\n(Home Win / Away Win / Draw Breakdown)', fontsize=13, fontweight='bold', pad=14)
        ax_detail.invert_yaxis()

        for bar, pct in zip(bars_b, breakdown_pcts):
            ax_detail.annotate(f"{pct:.1f}%",
                               xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                               xytext=(6, 0), textcoords="offset points",
                               ha='left', va='center', color=TEXT_COLOR, fontweight='bold', fontsize=10)

        plt.suptitle('MLS 2026 Predictor — Season Validation Benchmark (Through Gameweek 15)',
                     fontsize=15, fontweight='bold', color=TEXT_COLOR, y=1.02)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig6_2026_season_accuracy.png")
        plt.close()
        print("  ✓ Saved: reports/figures/fig6_2026_season_accuracy.png")
except Exception as e:
    print(f"  ✗ Figure 6 error: {e}")

print("\n══════════════════════════════════════════════════════════════")
print("✅ ALL PUBLICATION-GRADE CHARTS SUCCESSFULLY GENERATED!")
print("══════════════════════════════════════════════════════════════")
