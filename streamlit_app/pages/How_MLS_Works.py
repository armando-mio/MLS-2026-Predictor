"""
How MLS Works: Comprehensive guide to Major League Soccer
Accessible from the sidebar via Streamlit multipage convention.
"""
import streamlit as st
from pathlib import Path

# ── Page Config ──
st.set_page_config(
    page_title="How MLS Works · MLS 2026 Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (reuse project theme + page-specific) ──
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── All page-specific styles fully inlined with High-Contrast Tokens ──
st.markdown("""
<link href="https://unpkg.com/lucide-static@latest/font/lucide.css" rel="stylesheet">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .lucide { font-family: 'lucide' !important; font-style: normal; font-weight: normal; }
    .ic { font-family: 'lucide' !important; font-style: normal; font-weight: normal;
          font-size: 1.1rem; vertical-align: middle; margin-right: 0.3rem; }
    .ic-lg { font-size: 1.3rem; }

    /* Equal height columns utility */
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="column"] > div {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
    }
    div[data-testid="column"] [data-testid="stMarkdownContainer"] {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
        height: 100% !important;
    }

    /* Start Sidebar content higher up */
    [data-testid="stSidebar"] [data-testid="stSidebarContent"],
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    [data-testid="stSidebar"] > div:first-child,
    section[data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1.5rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.55rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] div:has(> [data-testid="stSegmentedControl"]),
    [data-testid="stSidebar"] [data-testid="stSegmentedControl"] {
        margin-top: 0.85rem !important;
        margin-bottom: 0.85rem !important;
    }

    .how-section {
        background: #1a2332; border: 1px solid #2a3545; border-radius: 12px;
        padding: 1.8rem 2rem; margin-bottom: 1.3rem;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    .how-section:hover {
        border-color: #3a5a8a;
        box-shadow: 0 4px 20px rgba(29, 66, 138, 0.15);
    }
    .how-title {
        font-size: 1.45rem; font-weight: 800; color: #f8fafc;
        margin-bottom: 0.75rem; font-family: 'Inter', sans-serif;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .how-body {
        color: #cbd5e1; font-size: 0.94rem; line-height: 1.75;
        font-family: 'Inter', sans-serif;
    }
    .how-body strong { color: #ffffff; font-weight: 700; }
    .how-sub {
        font-size: 1.12rem; font-weight: 700; color: #e2e8f0;
        margin: 1.2rem 0 0.6rem 0; font-family: 'Inter', sans-serif;
        display: flex; align-items: center; gap: 0.4rem;
    }
    .ib {
        border-radius: 8px; padding: 1rem 1.25rem; margin: 1.2rem 0;
        font-size: 0.92rem; line-height: 1.65; font-family: 'Inter', sans-serif;
    }
    .ib strong { color: #ffffff; font-weight: 700; }
    .ib-blue {
        background: rgba(30, 58, 138, 0.28); border: 1px solid rgba(59, 130, 246, 0.45);
        border-left: 4px solid #3b82f6; color: #dbeafe;
    }
    .ib-green {
        background: rgba(20, 83, 45, 0.28); border: 1px solid rgba(34, 197, 94, 0.45);
        border-left: 4px solid #22c55e; color: #dcfce7;
    }
    .ib-gold {
        background: rgba(113, 63, 18, 0.28); border: 1px solid rgba(234, 179, 8, 0.45);
        border-left: 4px solid #eab308; color: #fef9c3;
    }
    .ib-red {
        background: rgba(127, 29, 29, 0.28); border: 1px solid rgba(239, 68, 68, 0.45);
        border-left: 4px solid #ef4444; color: #fee2e2;
    }

    /* Conference Equal Height Cards */
    .conf-card {
        background: #1a2332; border: 1px solid #2a3545; border-radius: 12px;
        padding: 1.3rem; display: flex; flex-direction: column;
        height: 100%; min-height: 100%; box-sizing: border-box;
        font-family: 'Inter', sans-serif;
    }
    .conf-card-east { border-top: 3px solid #ef4444; }
    .conf-card-west { border-top: 3px solid #3b82f6; }

    .team-grid {
        display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
        gap: 0.55rem; margin-top: 0.8rem; flex: 1; align-content: start;
    }
    .team-chip {
        background: #141e2b; border: 1px solid #2a3545; border-radius: 8px;
        padding: 0.45rem 0.65rem; display: flex; align-items: center;
        gap: 0.55rem; font-size: 0.82rem; font-weight: 500; color: #f1f5f9;
        font-family: 'Inter', sans-serif; height: 38px; min-height: 38px;
        box-sizing: border-box;
        transition: border-color 0.2s ease, transform 0.15s ease, background 0.2s ease;
    }
    .team-chip span {
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .team-chip:hover {
        border-color: #38bdf8; transform: translateY(-1px);
        background: #192638;
    }
    .team-chip img { border-radius: 4px; flex-shrink: 0; }

    .stat-card {
        background: #1a2332; border: 1px solid #2a3545; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .stat-num {
        font-size: 2.1rem; font-weight: 800; color: #38bdf8;
        line-height: 1; font-family: 'Inter', sans-serif;
    }
    .stat-lbl {
        font-size: 0.82rem; color: #cbd5e1; margin-top: 0.3rem;
        font-weight: 600; font-family: 'Inter', sans-serif; letter-spacing: 0.3px;
    }

    .pts-badge {
        display: inline-flex; align-items: center; justify-content: center;
        border-radius: 6px; padding: 0.3rem 0.8rem;
        font-weight: 700; font-size: 0.85rem; font-family: 'Inter', sans-serif;
    }
    .pts-w { background: rgba(34, 197, 94, 0.22); color: #4ade80; }
    .pts-d { background: rgba(234, 179, 8, 0.22); color: #facc15; }
    .pts-l { background: rgba(239, 68, 68, 0.22); color: #f87171; }
    .pts-row {
        display: flex; gap: 2.5rem; margin: 1.2rem 0; flex-wrap: wrap; justify-content: center;
    }
    .pts-item {
        text-align: center; background: transparent; border: none;
        padding: 0.2rem 0.8rem; min-width: 100px;
    }
    .pts-big { font-size: 1.4rem; padding: 0.5rem 1.5rem; }
    .pts-label { font-size: 0.85rem; margin-top: 0.45rem; font-weight: 800; letter-spacing: 0.6px; }

    .card-grid-2 {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1rem; margin-top: 1rem;
    }
    .card-grid-3 {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1rem; margin-top: 1rem;
    }
    .card-grid-5 {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 0.75rem; margin-top: 1.2rem;
    }

    .mc {
        background: #141e2b; border: 1px solid #2a3545; border-radius: 10px;
        padding: 1.2rem; font-family: 'Inter', sans-serif; height: 100%;
        display: flex; flex-direction: column; justify-content: flex-start;
    }
    .mc-title { font-weight: 700; font-size: 0.98rem; margin-bottom: 0.45rem; display: flex; align-items: center; gap: 0.3rem; }
    .mc-body { color: #cbd5e1; font-size: 0.88rem; line-height: 1.65; }
    .mc-body strong { color: #ffffff; }

    .tl-step { display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1.2rem; }
    .tl-step:last-child { margin-bottom: 0; }
    .tl-dot {
        width: 36px; height: 36px; min-width: 36px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 0.88rem; color: white;
        font-family: 'Inter', sans-serif;
    }
    .tl-body h4 {
        color: #f8fafc !important; font-size: 1rem; font-weight: 700;
        margin: 0 0 0.25rem 0; font-family: 'Inter', sans-serif;
    }
    .tl-body p {
        color: #cbd5e1; font-size: 0.88rem; line-height: 1.65; margin: 0;
        font-family: 'Inter', sans-serif;
    }
    .tl-body p strong { color: #ffffff; }

    .rbox {
        background: #141e2b; border: 1px solid #2a3545; border-radius: 10px;
        padding: 1rem 0.6rem; text-align: center; font-family: 'Inter', sans-serif;
    }
    .rbox-t {
        font-size: 0.74rem; font-weight: 700; color: #cbd5e1;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.3rem;
    }
    .rbox-n { font-size: 1.45rem; font-weight: 800; color: #f8fafc; }
    .rbox-s { font-size: 0.78rem; color: #94a3b8; margin-top: 0.15rem; font-weight: 500; }

    .dtable {
        width: 100%; border-collapse: separate; border-spacing: 0;
        background: #141e2b; border-radius: 10px; overflow: hidden;
        border: 1px solid #2a3545; font-family: 'Inter', sans-serif;
        margin: 1rem 0;
    }
    .dtable th {
        padding: 0.75rem 0.85rem; color: #e2e8f0; font-size: 0.8rem;
        font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
        background: #0b131b; text-align: left; border-bottom: 1px solid #2a3545;
    }
    .dtable td {
        padding: 0.65rem 0.85rem; color: #f1f5f9; font-size: 0.88rem;
        border-bottom: 1px solid #1a2332;
    }
    .dtable tr:last-child td { border-bottom: none; }
    .dtable tr:hover td { background: rgba(255, 255, 255, 0.03); }

    .phase-card {
        flex: 1; min-width: 250px; background: #141e2b;
        border: 1px solid #2a3545; border-radius: 10px; padding: 1.2rem;
    }
    .phase-title { color: #f8fafc; font-weight: 700; font-size: 0.98rem; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.3rem; }
    .phase-body { color: #cbd5e1; font-size: 0.88rem; line-height: 1.65; }

    .seed-card {
        background: #141e2b; border: 1px solid #2a3545; border-radius: 10px;
        padding: 1.2rem; font-family: 'Inter', sans-serif;
    }
    .seed-title { font-weight: 700; font-size: 0.98rem; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.3rem; }
    .seed-body { color: #cbd5e1; font-size: 0.88rem; line-height: 1.65; }
    .seed-body strong { color: #ffffff; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TEAM DATA
# ═══════════════════════════════════════════════════════════════════════════

TEAM_ESPN_IDS = {
    "Atlanta Utd": 18418, "Austin FC": 20906, "CF Montreal": 9720,
    "Charlotte": 21300, "Chicago Fire": 182, "Colorado Rapids": 184,
    "Columbus Crew": 183, "DC United": 193, "FC Cincinnati": 18267,
    "FC Dallas": 185, "Houston Dynamo": 6077, "Inter Miami": 20232,
    "Los Angeles FC": 18966, "Los Angeles Galaxy": 187,
    "Minnesota United": 17362, "Nashville SC": 18986,
    "New England Revolution": 189, "New York City": 17606,
    "New York Red Bulls": 190, "Orlando City": 12011,
    "Philadelphia Union": 10739, "Portland Timbers": 9723,
    "Real Salt Lake": 4771, "San Diego FC": 22529,
    "San Jose Earthquakes": 191, "Seattle Sounders": 9726,
    "Sporting Kansas City": 186, "St. Louis City": 21812,
    "Toronto FC": 7318, "Vancouver Whitecaps": 9727,
}

EASTERN = [
    "Atlanta Utd", "CF Montreal", "Charlotte", "Chicago Fire",
    "Columbus Crew", "DC United", "FC Cincinnati", "Inter Miami",
    "Nashville SC", "New England Revolution", "New York City",
    "New York Red Bulls", "Orlando City", "Philadelphia Union", "Toronto FC",
]
WESTERN = [
    "Austin FC", "Colorado Rapids", "FC Dallas", "Houston Dynamo",
    "Los Angeles FC", "Los Angeles Galaxy", "Minnesota United",
    "Portland Timbers", "Real Salt Lake", "San Diego FC",
    "San Jose Earthquakes", "Seattle Sounders", "Sporting Kansas City",
    "St. Louis City", "Vancouver Whitecaps",
]


def logo_url(name):
    eid = TEAM_ESPN_IDS.get(name)
    return f"https://a.espncdn.com/i/teamlogos/soccer/500-dark/{eid}.png" if eid else ""


def chip(name):
    url = logo_url(name)
    img = f'<img src="{url}" width="22" height="22">' if url else ""
    return f'<div class="team-chip">{img} <span title="{name}">{name}</span></div>'


# Lucide icon helper
def ic(name, extra_class=""):
    return f'<i class="lucide lucide-{name} ic {extra_class}"></i>'


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:0.1rem 0 0.35rem 0;">
        <div style="font-size:1.4rem; font-weight:800; color:#f8fafc;">
            {ic('book-open', 'ic-lg')} How MLS Works
        </div>
        <div style="color:#cbd5e1; font-size:0.82rem; margin-top:0.15rem; font-weight:500;">Complete Guide to Major League Soccer</div>
    </div>
    """, unsafe_allow_html=True)

    nav_page = st.segmented_control(
        "Page Navigation",
        options=["Dashboard", "How MLS Works"],
        default="How MLS Works",
        label_visibility="collapsed",
        key="sidebar_nav_toggle_how"
    )
    if nav_page and nav_page != "How MLS Works":
        st.switch_page("app.py")

    st.markdown(f"""
    <div class="sidebar-widget">
        <div class="sidebar-widget-header">
            {ic('compass')} Quick Navigation
        </div>
        <div class="sidebar-nav-list">
            <a href="#league-overview" class="sidebar-nav-item">
                <span class="nav-item-icon">{ic('landmark')}</span>
                <span class="nav-item-text">League Overview</span>
                <i class="lucide lucide-chevron-right nav-item-arrow"></i>
            </a>
            <a href="#the-two-conferences" class="sidebar-nav-item">
                <span class="nav-item-icon">{ic('map')}</span>
                <span class="nav-item-text">The Two Conferences</span>
                <i class="lucide lucide-chevron-right nav-item-arrow"></i>
            </a>
            <a href="#the-regular-season" class="sidebar-nav-item">
                <span class="nav-item-icon">{ic('calendar')}</span>
                <span class="nav-item-text">Regular Season</span>
                <i class="lucide lucide-chevron-right nav-item-arrow"></i>
            </a>
            <a href="#mls-cup-playoffs" class="sidebar-nav-item">
                <span class="nav-item-icon">{ic('trophy')}</span>
                <span class="nav-item-text">MLS Cup Playoffs</span>
                <i class="lucide lucide-chevron-right nav-item-arrow"></i>
            </a>
            <a href="#the-mls-cup-final" class="sidebar-nav-item">
                <span class="nav-item-icon">{ic('medal')}</span>
                <span class="nav-item-text">MLS Cup Final</span>
                <i class="lucide lucide-chevron-right nav-item-arrow"></i>
            </a>
            <a href="#awards-trophies" class="sidebar-nav-item">
                <span class="nav-item-icon">{ic('award')}</span>
                <span class="nav-item-text">Awards & Trophies</span>
                <i class="lucide lucide-chevron-right nav-item-arrow"></i>
            </a>
            <a href="#how-our-predictor-fits-in" class="sidebar-nav-item">
                <span class="nav-item-icon">{ic('cpu')}</span>
                <span class="nav-item-text">How Our Predictor Works</span>
                <i class="lucide lucide-chevron-right nav-item-arrow"></i>
            </a>
            <a href="#mls-vs-europe" class="sidebar-nav-item">
                <span class="nav-item-icon">{ic('zap')}</span>
                <span class="nav-item-text">MLS vs European Leagues</span>
                <i class="lucide lucide-chevron-right nav-item-arrow"></i>
            </a>
        </div>
    </div>

    <div class="model-simple-badge">
        <div class="model-simple-title">
            <span class="model-simple-pill">v4.0</span>
            <span>RandomForest Model</span>
        </div>
        <div class="model-simple-sub">
            Train: 2022–2024 · Test: 2025 · <strong style="color:#38bdf8;">2026 Season Prediction</strong>
        </div>
    </div>

    <div class="creator-card">
        <div class="created-by-label">
            <span>Created by</span>
            <span class="created-by-name">Armando Mio</span>
        </div>
        <div class="social-links-grid">
            <a href="https://www.linkedin.com/in/armando-mio" target="_blank" class="social-btn linkedin-btn" title="LinkedIn Profile">
                <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
                </svg>
                <span>LinkedIn</span>
            </a>
            <a href="https://github.com/armando-mio" target="_blank" class="social-btn github-btn" title="GitHub Profile">
                <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                <span>GitHub</span>
            </a>
        </div>
        <a href="https://github.com/armando-mio/MLS-2026-Predictor" target="_blank" class="repo-link-btn" title="View Source Code on GitHub">
            <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
            </svg>
            <span>Project Repository</span>
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="opacity: 0.6; margin-left: auto;">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                <polyline points="15 3 21 3 21 9"></polyline>
                <line x1="10" y1="14" x2="21" y2="3"></line>
            </svg>
        </a>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center; padding:2rem 0 0.5rem 0;">
    <div style="font-size:0.85rem; font-weight:700; color:#38bdf8;
        text-transform:uppercase; letter-spacing:2px; margin-bottom:0.5rem;">
        Complete Guide
    </div>
    <h1 style="font-size:2.4rem; font-weight:800; color:#f8fafc;
        margin:0; line-height:1.2; font-family:'Inter',sans-serif;">
        How Major League Soccer Works
    </h1>
    <p style="color:#cbd5e1; font-size:1rem; max-width:700px;
        margin:0.8rem auto 0 auto; line-height:1.6; font-family:'Inter',sans-serif;">
        From the regular season through the playoffs to the MLS Cup Final:
        everything you need to understand North America's top-flight
        professional soccer league and how our prediction model fits into it.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Key Stats Bar ──
cols = st.columns(5)
for col, (num, lbl) in zip(cols, [
    ("30", "Teams"), ("2", "Conferences"), ("34", "Matches per Team"),
    ("9", "Playoff Teams / Conf."), ("1", "MLS Cup Champion"),
]):
    with col:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: LEAGUE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div id="league-overview"></div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="how-section">
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
        <img src="https://a.espncdn.com/combiner/i?img=/i/leaguelogos/soccer/500/19.png&w=60&h=60"
             width="60" height="60" style="border-radius:8px;">
        <div class="how-title" style="margin-bottom:0;">{ic('landmark')} League Overview</div>
    </div>
    <div class="how-body" style="margin-bottom:0.8rem;">
        <strong>Major League Soccer (MLS)</strong> is the top professional soccer league in
        the <strong>United States and Canada</strong>. Founded in 1993 and first played in 1996,
        MLS has grown into one of the most dynamic and rapidly expanding soccer leagues in the world.
    </div>
    <div class="how-body">
        As of the <strong>2026 season</strong>, the league features <strong>30 clubs</strong>
        divided into two conferences: the <strong style="color:#f87171;">Eastern Conference</strong>
        and the <strong style="color:#60a5fa;">Western Conference</strong>, with 15 teams in each.
        The MLS season runs from <strong>late February through early December</strong> and
        consists of two main phases:
    </div>
    <div style="display:flex; gap:1rem; margin-top:1.2rem; flex-wrap:wrap;">
        <div class="phase-card" style="border-top:3px solid #3b82f6;">
            <div class="phase-title">{ic('calendar-days')} 1. Regular Season</div>
            <div class="phase-body">Feb to Oct · 34 matches per team · Points determine conference standings</div>
        </div>
        <div class="phase-card" style="border-top:3px solid #facc15;">
            <div class="phase-title">{ic('trophy')} 2. MLS Cup Playoffs</div>
            <div class="phase-body">Oct to Dec · Knockout tournament · Top 9 teams per conference qualify</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: CONFERENCES (Equal Height Boxes)
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div id="the-two-conferences"></div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="how-section">
    <div class="how-title">{ic('map')} The Two Conferences</div>
    <div class="how-body">
        MLS teams are split geographically into two conferences. Each conference operates
        as its own standings table during the regular season, and each has its own playoff bracket.
        The two conference champions meet in the <strong>MLS Cup Final</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

col_e, col_w = st.columns(2)
with col_e:
    east_chips = "".join(chip(t) for t in EASTERN)
    st.markdown(f"""
    <div class="conf-card conf-card-east">
        <div style="font-size:1.15rem; font-weight:800; color:#f87171; margin-bottom:0.2rem; display:flex; align-items:center; justify-content:space-between;">
            <span>Eastern Conference</span>
            <span style="color:#cbd5e1; font-size:0.8rem; font-weight:600; background:rgba(239,68,68,0.18); padding:0.2rem 0.6rem; border-radius:6px; border:1px solid rgba(239,68,68,0.35);">15 teams</span>
        </div>
        <div class="team-grid">{east_chips}</div>
    </div>
    """, unsafe_allow_html=True)

with col_w:
    west_chips = "".join(chip(t) for t in WESTERN)
    st.markdown(f"""
    <div class="conf-card conf-card-west">
        <div style="font-size:1.15rem; font-weight:800; color:#60a5fa; margin-bottom:0.2rem; display:flex; align-items:center; justify-content:space-between;">
            <span>Western Conference</span>
            <span style="color:#cbd5e1; font-size:0.8rem; font-weight:600; background:rgba(59,130,246,0.18); padding:0.2rem 0.6rem; border-radius:6px; border:1px solid rgba(59,130,246,0.35);">15 teams</span>
        </div>
        <div class="team-grid">{west_chips}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="ib ib-blue">
    <strong>{ic('lightbulb')} Why conferences matter:</strong> Teams play <em>more matches</em> against
    opponents within their own conference than against cross-conference rivals. This means
    conference strength directly impacts playoff qualification, and our prediction model
    accounts for this through Elo ratings, strength of schedule, and head-to-head records.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: REGULAR SEASON
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div id="the-regular-season"></div>', unsafe_allow_html=True)

# Unified Regular Season Schedule Card
st.markdown(f"""
<div class="how-section">
    <div class="how-title">{ic('calendar')} The Regular Season & Schedule Structure</div>
    <div class="how-body">
        The regular season is the backbone of the MLS calendar. Each of the 30 teams plays
        <strong>34 matches</strong> (17 home, 17 away) from late February through mid-October.
        The schedule balances regional matchups with cross-country battles:
    </div>
    <div class="card-grid-2">
        <div class="mc">
            <div class="mc-title" style="color:#f87171;">{ic('home')} Intra-Conference (~24 matches)</div>
            <div class="mc-body">
                Each team plays every other team in their conference at least <strong>twice</strong>
                (once home, once away), with select regional rivals meeting three times. These matches
                directly decide conference playoff berths.
            </div>
        </div>
        <div class="mc">
            <div class="mc-title" style="color:#60a5fa;">{ic('plane')} Inter-Conference (~10 matches)</div>
            <div class="mc-body">
                Each team also plays a curated selection of teams from the opposite conference. Not every
                cross-conference matchup occurs each season, introducing <strong>travel distance and fatigue</strong>
                as major competitive factors.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Unified Points System Card
st.markdown(f"""
<div class="how-section">
    <div class="how-title">{ic('hash')} Points System & Tiebreakers</div>
    <div class="how-body">
        Like most soccer leagues worldwide, MLS uses a standard <strong>3-1-0 points system</strong>
        to rank clubs within their conference table:
    </div>
    <div class="pts-row">
        <div class="pts-item">
            <span class="pts-badge pts-w pts-big">3 pts</span>
            <div style="color:#4ade80;" class="pts-label">WIN</div>
        </div>
        <div class="pts-item">
            <span class="pts-badge pts-d pts-big">1 pt</span>
            <div style="color:#facc15;" class="pts-label">DRAW</div>
        </div>
        <div class="pts-item">
            <span class="pts-badge pts-l pts-big">0 pts</span>
            <div style="color:#f87171;" class="pts-label">LOSS</div>
        </div>
    </div>
    <div class="how-body" style="margin-top:1rem; border-top:1px solid #2a3545; padding-top:1rem;">
        Teams are ranked within their conference by total points. If two or more teams finish tied on points,
        <strong>tiebreakers</strong> are applied in strict order:
        <strong>1.</strong> Total wins,
        <strong>2.</strong> Goal difference,
        <strong>3.</strong> Goals scored,
        <strong>4.</strong> Away goals,
        <strong>5.</strong> Home goals,
        <strong>6.</strong> Away goal difference,
        <strong>7.</strong> Fewest disciplinary points.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="ib ib-gold">
    <strong>{ic('star')} Decision Day:</strong> The final matchday of the regular season is called
    <strong>"Decision Day"</strong>: all matches kick off simultaneously across each conference. This dramatic
    finale locks in the final playoff seeds and wild card qualifiers in real time.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: MLS CUP PLAYOFFS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div id="mls-cup-playoffs"></div>', unsafe_allow_html=True)

# Unified Playoff Overview & Qualification Card
st.markdown(f"""
<div class="how-section">
    <div class="how-title">{ic('trophy')} MLS Cup Playoffs: Overview & Qualification</div>
    <div class="how-body">
        The MLS Cup Playoffs are a <strong>multi-round knockout tournament</strong> to determine
        the league champion. <strong>18 teams total</strong> (top 9 from each conference) qualify
        based on their regular season standings. Higher regular season seeds earn critical advantages:
    </div>
    <div class="card-grid-3">
        <div class="seed-card" style="border-left:3px solid #22c55e; background:rgba(34,197,94,0.08);">
            <div class="seed-title" style="color:#4ade80;">{ic('award')} Seed #1</div>
            <div class="seed-body">
                <strong>First-round BYE</strong>: Skips the Wild Card match entirely.
                Enters directly at Round One with home-field advantage and extra rest.
            </div>
        </div>
        <div class="seed-card" style="border-left:3px solid #3b82f6; background:rgba(59,130,246,0.08);">
            <div class="seed-title" style="color:#60a5fa;">{ic('home')} Seeds #2 to #4</div>
            <div class="seed-body">
                <strong>Home advantage</strong> in Round One: Host Games 1 and 3 (if needed)
                in the best-of-three series.
            </div>
        </div>
        <div class="seed-card" style="border-left:3px solid #eab308; background:rgba(234,179,8,0.08);">
            <div class="seed-title" style="color:#facc15;">{ic('plane')} Seeds #5 to #9</div>
            <div class="seed-body">
                <strong>Away in Round One</strong>: Must travel. Seeds #8 and #9 must first survive
                a single-elimination Wild Card match.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Unified Playoff Rounds Card (Timeline + Visual Summary)
st.markdown(f"""
<div class="how-section">
    <div class="how-title">{ic('list-ordered')} Playoff Rounds: Step by Step</div>
    <div class="how-body" style="margin-bottom:1.4rem;">
        The playoffs progress through four distinct tournament stages to crown the champions of each conference:
    </div>
    <div class="tl-step">
        <div class="tl-dot" style="background:#64748b;">W</div>
        <div class="tl-body">
            <h4>Wild Card Round</h4>
            <p><strong>Single-elimination</strong> match. Seed #8 hosts Seed #9 in each conference.
            If tied after 90 minutes, goes directly to penalty kicks (no extra time).
            The winner advances to face the #1 seed in Round One.</p>
        </div>
    </div>
    <div class="tl-step">
        <div class="tl-dot" style="background:#2563eb;">1</div>
        <div class="tl-body">
            <h4>Round One</h4>
            <p><strong>Best-of-three series.</strong> The higher seed hosts Games 1 and 3; the lower
            seed hosts Game 2. Matchups: #1 vs Wild Card winner, #2 vs #7, #3 vs #6, #4 vs #5.
            Every game has a winner: tied matches go directly to penalties.</p>
        </div>
    </div>
    <div class="tl-step">
        <div class="tl-dot" style="background:#0284c7;">2</div>
        <div class="tl-body">
            <h4>Conference Semifinals</h4>
            <p><strong>Single-elimination.</strong> The four Round One winners are paired.
            Higher regular season seed hosts. Winners advance to the Conference Finals.</p>
        </div>
    </div>
    <div class="tl-step">
        <div class="tl-dot" style="background:#eab308; color:#0f172a;">3</div>
        <div class="tl-body">
            <h4>Conference Finals</h4>
            <p><strong>Single-elimination.</strong> The two remaining teams in each conference
            battle for the conference crown. Higher seed hosts. The winner becomes the
            <strong>Conference Champion</strong>.</p>
        </div>
    </div>
    <div class="tl-step">
        <div class="tl-dot" style="background:#ef4444;">F</div>
        <div class="tl-body">
            <h4>MLS Cup Final</h4>
            <p>Eastern Conference Champion vs Western Conference Champion.
            <strong>One match</strong>, hosted by the finalist with the better regular season record.
            If tied: 30 minutes of extra time, then penalty kicks.</p>
        </div>
    </div>
    <div style="border-top:1px solid #2a3545; margin-top:1.4rem; padding-top:1.2rem;">
        <div style="font-size:0.85rem; font-weight:700; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.6rem;">
            Total Matches in the Playoff Structure
        </div>
        <div class="card-grid-5">
            <div class="rbox" style="border-top:3px solid #64748b;">
                <div class="rbox-t">WILD CARD</div>
                <div class="rbox-n" style="color:#cbd5e1;">2</div>
                <div class="rbox-s">matches</div>
            </div>
            <div class="rbox" style="border-top:3px solid #2563eb;">
                <div class="rbox-t">ROUND ONE</div>
                <div class="rbox-n" style="color:#60a5fa;">8</div>
                <div class="rbox-s">series</div>
            </div>
            <div class="rbox" style="border-top:3px solid #0284c7;">
                <div class="rbox-t">CONF. SEMIS</div>
                <div class="rbox-n" style="color:#38bdf8;">4</div>
                <div class="rbox-s">matches</div>
            </div>
            <div class="rbox" style="border-top:3px solid #eab308;">
                <div class="rbox-t">CONF. FINALS</div>
                <div class="rbox-n" style="color:#facc15;">2</div>
                <div class="rbox-s">matches</div>
            </div>
            <div class="rbox" style="border-top:3px solid #ef4444;">
                <div class="rbox-t">MLS CUP</div>
                <div class="rbox-n" style="color:#f87171;">1</div>
                <div class="rbox-s">final match</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Unified 2024 Playoff Example Card
st.markdown(f"""
<div class="how-section">
    <div class="how-title">{ic('pin')} Example: 2024 MLS Cup Playoffs (Eastern Conference)</div>
    <div class="how-body">
        Here is how the regular season standings translated into seeds, BYEs, and home advantages in the 2024 season:
    </div>
    <table class="dtable">
        <thead><tr>
            <th style="text-align:center;">Seed</th><th>Team</th>
            <th style="text-align:center;">Points</th><th style="text-align:center;">Advantage Earned</th>
        </tr></thead>
        <tbody>
            <tr><td style="color:#4ade80; font-weight:700; text-align:center;">1</td>
                <td style="color:#f8fafc; font-weight:600;">Inter Miami</td><td style="text-align:center; color:#cbd5e1;">74</td>
                <td style="text-align:center;"><span class="pts-badge pts-w">BYE + R1 Home</span></td></tr>
            <tr><td style="color:#60a5fa; font-weight:700; text-align:center;">2-4</td>
                <td style="color:#f8fafc;">Columbus Crew, FC Cincinnati, Charlotte FC</td><td style="text-align:center; color:#cbd5e1;">54-62</td>
                <td style="text-align:center;"><span class="pts-badge pts-w">R1 Home</span></td></tr>
            <tr><td style="color:#facc15; font-weight:700; text-align:center;">5-7</td>
                <td style="color:#f8fafc;">NYCFC, Orlando City, NY Red Bulls</td><td style="text-align:center; color:#cbd5e1;">48-51</td>
                <td style="text-align:center;"><span class="pts-badge pts-l">R1 Away</span></td></tr>
            <tr><td style="color:#cbd5e1; font-weight:700; text-align:center;">8-9</td>
                <td style="color:#f8fafc;">CF Montreal, Atlanta United</td><td style="text-align:center; color:#cbd5e1;">43-46</td>
                <td style="text-align:center;"><span class="pts-badge pts-l">Wild Card Match</span></td></tr>
        </tbody>
    </table>
    <div class="how-body" style="margin-top:0.8rem;">
        The <strong>2024 MLS Cup Final</strong> was played between the <strong>LA Galaxy</strong>
        (Western Conference Champions) and the <strong>New York Red Bulls</strong>
        (Eastern Conference Champions). The Galaxy won <strong>2-1</strong> to claim their
        6th MLS Cup title: the most in league history.
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: MLS CUP FINAL
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div id="the-mls-cup-final"></div>', unsafe_allow_html=True)

# Unified MLS Cup Final Card
st.markdown(f"""
<div class="how-section">
    <div class="how-title">{ic('medal')} The MLS Cup Final</div>
    <div class="how-body">
        The <strong>MLS Cup</strong> is the league's championship match: the grand culmination of
        the entire season. It pits the <strong>Eastern Conference Champion</strong> against the
        <strong>Western Conference Champion</strong> in a single, winner-take-all showdown.
    </div>
    <div class="card-grid-2">
        <div class="mc">
            <div class="mc-title" style="color:#f8fafc;">{ic('map-pin')} Venue Selection</div>
            <div class="mc-body">
                Hosted at the home stadium of the finalist with the <strong>better regular season record</strong>
                (higher total points). If tied, standard tiebreakers apply. This gives top regular-season performers
                massive home-field motivation all year long.
            </div>
        </div>
        <div class="mc">
            <div class="mc-title" style="color:#f8fafc;">{ic('clock')} Match Format</div>
            <div class="mc-body">
                Standard 90 minutes regulation. If tied after 90: <strong>30 minutes of extra time</strong>
                (two 15-minute halves). If still tied: a dramatic <strong>penalty shootout</strong>.
                There is no second leg or replay.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="ib ib-green">
    <strong>{ic('trophy')} The Philip F. Anschutz Trophy:</strong> Crafted by Tiffany & Co., this sterling
    silver trophy stands 28.5 inches tall and weighs 43 pounds. Named after one of MLS's
    founding investors, it is hoisted by the MLS Cup champion. The winner also earns automatic qualification
    into the prestigious <strong>CONCACAF Champions Cup</strong>.
</div>
""", unsafe_allow_html=True)

# Unified Recent Champions Card
st.markdown(f"""
<div class="how-section">
    <div class="how-title">{ic('crown')} Recent MLS Cup Champions</div>
    <div class="how-body">
        A look at the past five MLS Cup Finals, champions, scores, and host venues:
    </div>
    <table class="dtable">
        <thead><tr>
            <th style="text-align:center;">Year</th><th>Champion</th>
            <th style="text-align:center;">Score</th><th>Runner-Up</th>
            <th>Host Venue</th>
        </tr></thead>
        <tbody>
            <tr><td style="color:#facc15; text-align:center; font-weight:700;">2024</td>
                <td style="color:#f8fafc; font-weight:600;">LA Galaxy</td>
                <td style="text-align:center; font-weight:700; color:#4ade80;">2 - 1</td><td style="color:#cbd5e1;">NY Red Bulls</td>
                <td style="color:#cbd5e1;">Dignity Health Sports Park (Carson, CA)</td></tr>
            <tr><td style="color:#facc15; text-align:center; font-weight:700;">2023</td>
                <td style="color:#f8fafc; font-weight:600;">Columbus Crew</td>
                <td style="text-align:center; font-weight:700; color:#4ade80;">2 - 1</td><td style="color:#cbd5e1;">LAFC</td>
                <td style="color:#cbd5e1;">Lower.com Field (Columbus, OH)</td></tr>
            <tr><td style="color:#facc15; text-align:center; font-weight:700;">2022</td>
                <td style="color:#f8fafc; font-weight:600;">LAFC</td>
                <td style="text-align:center; font-weight:700; color:#4ade80;">3 - 3 (PKs)</td><td style="color:#cbd5e1;">Philadelphia Union</td>
                <td style="color:#cbd5e1;">Banc of California Stadium (Los Angeles, CA)</td></tr>
            <tr><td style="color:#facc15; text-align:center; font-weight:700;">2021</td>
                <td style="color:#f8fafc; font-weight:600;">New York City FC</td>
                <td style="text-align:center; font-weight:700; color:#4ade80;">1 - 1 (PKs)</td><td style="color:#cbd5e1;">Portland Timbers</td>
                <td style="color:#cbd5e1;">Providence Park (Portland, OR)</td></tr>
            <tr><td style="color:#facc15; text-align:center; font-weight:700;">2020</td>
                <td style="color:#f8fafc; font-weight:600;">Columbus Crew</td>
                <td style="text-align:center; font-weight:700; color:#4ade80;">3 - 0</td><td style="color:#cbd5e1;">Seattle Sounders</td>
                <td style="color:#cbd5e1;">MAPFRE Stadium (Columbus, OH)</td></tr>
        </tbody>
    </table>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6: AWARDS & TROPHIES
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div id="awards-trophies"></div>', unsafe_allow_html=True)

# Unified Awards & Trophies Card
st.markdown(f"""
<div class="how-section">
    <div class="how-title">{ic('award')} Awards and Major Trophies</div>
    <div class="how-body">
        Clubs in Major League Soccer compete for multiple major domestic and continental honors across the campaign:
    </div>
    <div class="card-grid-2">
        <div class="mc" style="border-left:3px solid #eab308;">
            <div class="mc-title" style="color:#facc15;">{ic('trophy')} MLS Cup</div>
            <div class="mc-body">
                Awarded to the <strong>playoff champion</strong>. The most prestigious trophy
                in North American soccer: requires navigating the entire knockout tournament.
            </div>
        </div>
        <div class="mc" style="border-left:3px solid #22c55e;">
            <div class="mc-title" style="color:#4ade80;">{ic('shield')} Supporters' Shield</div>
            <div class="mc-body">
                Awarded to the team with the <strong>best overall regular season record</strong>
                across both conferences. The ultimate test of consistency over 34 matches.
            </div>
        </div>
        <div class="mc" style="border-left:3px solid #3b82f6;">
            <div class="mc-title" style="color:#60a5fa;">{ic('globe')} CONCACAF Champions Cup</div>
            <div class="mc-body">
                The MLS Cup champion and other top clubs earn qualification to this premier continental
                tournament against the best teams from Mexico (Liga MX) and Central America.
            </div>
        </div>
        <div class="mc" style="border-left:3px solid #ef4444;">
            <div class="mc-title" style="color:#f87171;">{ic('swords')} Lamar Hunt U.S. Open Cup</div>
            <div class="mc-body">
                A 100+ year-old knockout tournament open to <strong>all tiers</strong> of American soccer.
                Lower-division clubs face MLS squads in classic underdog matchups.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="ib ib-blue">
    <strong>{ic('lightbulb')} MLS Cup is not the same as the Supporters' Shield:</strong> Unlike European leagues where the
    league champion is simply the team that finishes first in the standings, MLS crowns its official champion
    through the postseason playoffs. A team can win the Supporters' Shield (best regular season record) but get eliminated
    in the playoffs. This dual system is <strong>uniquely North American</strong>, sharing DNA with the NFL, NBA, MLB, and NHL.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7: HOW OUR PREDICTOR FITS IN
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div id="how-our-predictor-fits-in"></div>', unsafe_allow_html=True)

# Unified Predictor Explanation Card
st.markdown(f"""
<div class="how-section" style="border:1px solid rgba(59,130,246,0.45);
    background:linear-gradient(135deg, #1a2332 0%, #141e2b 100%);">
    <div class="how-title">{ic('cpu')} How Our Predictor Fits In</div>
    <div class="how-body" style="margin-bottom:1.4rem;">
        Now that you understand the MLS competition rules, here is how the
        <strong>MLS 2026 Predictor</strong> model uses this structure to generate realistic outcomes:
    </div>
    <div class="tl-step">
        <div class="tl-dot" style="background:#2563eb;">1</div>
        <div class="tl-body">
            <h4>Match-Level Predictions</h4>
            <p>Our <strong>RandomForest model</strong> predicts individual match outcomes
            (Home Win, Draw, Away Win) using 40+ dynamic features: Elo ratings, rolling expected goals,
            travel distance, rest days, head-to-head records, and home-field strength.</p>
        </div>
    </div>
    <div class="tl-step">
        <div class="tl-dot" style="background:#0284c7;">2</div>
        <div class="tl-body">
            <h4>Season Simulation</h4>
            <p>Using match-level probabilities, we run <strong>Monte Carlo simulations</strong>
            (thousands of full season replays) to project final points, conference standings,
            and playoff qualification probabilities under realistic uncertainty.</p>
        </div>
    </div>
    <div class="tl-step">
        <div class="tl-dot" style="background:#eab308; color:#0f172a;">3</div>
        <div class="tl-body">
            <h4>Playoff & Championship Probabilities</h4>
            <p>For each simulated season, we seed the exact 18-team bracket and simulate the
            Wild Card, Best-of-3 Round One, Semifinals, and Final to calculate each club's odds of lifting the MLS Cup.</p>
        </div>
    </div>
    <div class="tl-step">
        <div class="tl-dot" style="background:#ef4444;">4</div>
        <div class="tl-body">
            <h4>What-If & Scenario Analysis</h4>
            <p>Simulate key player absences, rest differentials, or travel schedule changes in the
            <strong>What-If Simulator</strong> to inspect instant SHAP feature contributions and probability shifts.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="ib ib-green">
    <strong>{ic('arrow-right')} Ready to explore predictions?</strong> Switch back to the main
    <strong>MLS 2026 Predictor</strong> dashboard via the sidebar navigation to view
    match projections, feature importance, what-if simulations, and title contenders!
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 8: MLS vs EUROPEAN LEAGUES
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div id="mls-vs-europe"></div>', unsafe_allow_html=True)

# Unified MLS vs European Leagues Card
st.markdown(f"""
<div class="how-section">
    <div class="how-title">{ic('zap')} MLS vs European Leagues: Key Differences</div>
    <div class="how-body">
        For fans accustomed to European soccer (Premier League, La Liga, Serie A, Bundesliga),
        MLS has several fundamental structural and regulatory differences:
    </div>
    <table class="dtable">
        <thead><tr>
            <th>Aspect</th>
            <th style="color:#60a5fa;">Major League Soccer (USA / CAN)</th>
            <th style="color:#f87171;">European Leagues (UEFA)</th>
        </tr></thead>
        <tbody>
            <tr><td style="color:#f8fafc; font-weight:600;">Champion Decided By</td>
                <td style="color:#f1f5f9;">Playoffs (MLS Cup Knockout)</td><td style="color:#cbd5e1;">Single league table (most points)</td></tr>
            <tr><td style="color:#f8fafc; font-weight:600;">Relegation / Promotion</td>
                <td style="color:#4ade80; font-weight:600;">No (Closed franchise model)</td><td style="color:#f87171;">Yes (Bottom clubs drop to lower tier)</td></tr>
            <tr><td style="color:#f8fafc; font-weight:600;">Conferences</td>
                <td style="color:#f1f5f9;">2 Conferences (Eastern & Western)</td><td style="color:#cbd5e1;">None (Single unified table)</td></tr>
            <tr><td style="color:#f8fafc; font-weight:600;">Season Calendar</td>
                <td style="color:#f1f5f9;">Late February – Early December</td><td style="color:#cbd5e1;">August – May</td></tr>
            <tr><td style="color:#f8fafc; font-weight:600;">Salary & Roster Rules</td>
                <td style="color:#f1f5f9;">Hard Salary Cap + Designated Player (DP) slots</td><td style="color:#cbd5e1;">Financial Fair Play / Sustainability Rules</td></tr>
            <tr><td style="color:#f8fafc; font-weight:600;">Travel Distances</td>
                <td style="color:#facc15; font-weight:600;">Extreme (Coast-to-coast across 4 time zones)</td><td style="color:#cbd5e1;">Moderate (Domestic within single nation)</td></tr>
        </tbody>
    </table>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="ib ib-red">
    <strong>{ic('globe')} Travel distance is a huge factor in MLS:</strong> When Seattle Sounders travel
    to play Inter Miami, they cover over <strong>4,400 km (2,735 miles)</strong>: roughly equivalent
    to flying from London to Baghdad across 3 time zones. Our model specifically incorporates
    <strong>travel distance</strong> and <strong>fatigue indices</strong> because these
    long distances significantly impact away win probabilities.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:1.5rem 0; border-top:1px solid #2a3545;">
    <div style="color:#94a3b8; font-size:0.85rem; font-weight:500;">
        MLS 2026 Predictor · Built with Streamlit · Powered by Machine Learning
    </div>
    <div style="color:#64748b; font-size:0.75rem; margin-top:0.3rem;">
        Data from football-data.co.uk · Team logos via ESPN CDN
    </div>
</div>
""", unsafe_allow_html=True)
