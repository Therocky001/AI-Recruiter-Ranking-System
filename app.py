import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys
import os
import re
import json
from pathlib import Path
from datetime import datetime

# ── Optional plotly ──────────────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
CSV_PATH = BASE / "team_submission.csv"
CANDIDATES_PATH = (
    BASE.parent
    / "Data"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "candidates.jsonl"
)

# ── Penalty config ────────────────────────────────────────────────────────────
PENALTY_META = {
    "behavioral_downweight": {"color": "#ef4444", "label": "Behavioral ↓"},
    "pure_academic":          {"color": "#f97316", "label": "Pure Academic"},
    "witch_only":             {"color": "#eab308", "label": "WITCH Only"},
    "title_inflation":        {"color": "#a855f7", "label": "Title Inflation"},
    "job_hopper":             {"color": "#06b6d4", "label": "Job Hopper"},
    "exp_below_3yr":          {"color": "#dc2626", "label": "Exp <3yr"},
    "exp_low_3_4yr":          {"color": "#f59e0b", "label": "Exp 3–4yr"},
    "hedging_language":       {"color": "#64748b", "label": "Hedging Language"},
    "no_vector_db_or_eval":   {"color": "#6366f1", "label": "No VectorDB/Eval"},
}

st.set_page_config(
    page_title="AI Recruiter · Redrob Hackathon",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background-color: #F8F9FB !important;
  color: #1e293b !important;
}
.stApp { background-color: #F8F9FB !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: white !important;
  border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] * { color: #334155 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
  font-weight: 500 !important;
  font-size: 0.9rem !important;
  color: #64748b !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: #6366f1 !important;
  border-bottom-color: #6366f1 !important;
  font-weight: 600 !important;
}

/* ── Buttons ── */
div[data-testid="stButton"] > button {
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  padding: 0.55rem 1.5rem !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 8px rgba(99,102,241,0.3) !important;
}
div[data-testid="stButton"] > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
  background: white !important;
  color: #6366f1 !important;
  border: 2px solid #6366f1 !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
}

/* ── Inputs ── */
.stTextInput input, .stSelectbox select, [data-testid="stNumberInput"] input {
  background: white !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 8px !important;
  color: #1e293b !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: white !important;
  padding: 1rem 1.25rem !important;
  border-radius: 14px !important;
  border: 1px solid #e2e8f0 !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.82rem !important; }
[data-testid="stMetricValue"] { color: #1e293b !important; font-size: 1.7rem !important; font-weight: 700 !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: white !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 12px !important;
}

/* ── Utility classes ── */
.card {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 1.25rem 1.5rem;
  box-shadow: 0 1px 6px rgba(0,0,0,0.05);
  margin-bottom: 0.75rem;
}
.hero {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
  border-radius: 18px;
  padding: 2rem 2.5rem;
  margin-bottom: 1.5rem;
  color: white;
}
.hero h1 { margin: 0; font-size: 2rem; font-weight: 800; color: white; }
.hero p  { margin: 0.4rem 0 0; color: rgba(255,255,255,0.85); font-size: 1rem; }
.section-title { font-size: 1.1rem; font-weight: 700; color: #1e293b; margin: 1.2rem 0 0.6rem; }
.badge {
  display: inline-block; padding: 3px 10px; border-radius: 9999px;
  font-size: 0.7rem; font-weight: 600; margin: 2px; color: white;
  white-space: nowrap;
}
.clean-badge {
  display: inline-block; padding: 3px 10px; border-radius: 9999px;
  font-size: 0.7rem; font-weight: 600; margin: 2px;
  background: #dcfce7; color: #16a34a;
}
.rank-card {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.6rem;
  box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  transition: box-shadow 0.15s;
}
.rank-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.rank-num  { font-weight: 800; color: #6366f1; font-size: 1.15rem; }
.cand-id   { font-family: monospace; font-size: 0.82rem; color: #94a3b8; }
.score-val { font-weight: 700; color: #059669; font-size: 1rem; }
.reasoning { color: #475569; font-size: 0.875rem; line-height: 1.55; margin-top: 0.4rem; }
.pill-chip {
  display: inline-block; background: #f1f5f9; border: 1px solid #cbd5e1;
  border-radius: 6px; padding: 2px 8px; font-size: 0.75rem;
  color: #64748b; margin: 2px;
}
.flow-step {
  background: white; border: 2px solid #e2e8f0; border-radius: 12px;
  padding: 1rem; text-align: center; transition: border-color 0.2s;
}
.flow-step:hover { border-color: #6366f1; }
.flow-step .step-num { font-size: 1.5rem; font-weight: 800; color: #6366f1; }
.flow-step .step-name { font-size: 0.85rem; font-weight: 600; color: #1e293b; margin: 0.3rem 0 0.2rem; }
.flow-step .step-desc { font-size: 0.75rem; color: #64748b; }
.status-dot-ok  { display:inline-block; width:8px; height:8px; border-radius:50%; background:#22c55e; margin-right:5px; }
.status-dot-err { display:inline-block; width:8px; height:8px; border-radius:50%; background:#ef4444; margin-right:5px; }
</style>
""", unsafe_allow_html=True)


# ── Data helpers ──────────────────────────────────────────────────────────────
def extract_penalties(reasoning: str) -> list:
    match = re.search(r'\[Penalties: ([^\]]+)\]', str(reasoning))
    return [p.strip() for p in match.group(1).split(',')] if match else []

@st.cache_data(ttl=60)
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["rank"]          = df["rank"].astype(int)
    df["score"]         = df["score"].astype(float)
    df["penalties"]     = df["reasoning"].apply(extract_penalties)
    df["penalty_count"] = df["penalties"].apply(len)
    df["penalized"]     = df["penalty_count"] > 0
    # Parse title / exp from reasoning  e.g. "Senior ML Eng, 6.2yrs, ..."
    def parse_field(r, idx, default="—"):
        try: return r.split(",")[idx].strip()
        except: return default
    df["title_parsed"]  = df["reasoning"].apply(lambda r: parse_field(r, 0))
    df["exp_parsed"]    = df["reasoning"].apply(lambda r: parse_field(r, 1))
    return df

def penalty_badges(penalties: list, size="normal") -> str:
    if not penalties:
        return '<span class="clean-badge">✓ Clean</span>'
    out = ""
    for p in penalties:
        meta = PENALTY_META.get(p, {"color": "#64748b", "label": p})
        out += f'<span class="badge" style="background:{meta["color"]};">{meta["label"]}</span>'
    return out


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
      <span style="font-size:1.4rem;">🎯</span>
      <span style="font-size:1rem;font-weight:700;color:#1e293b;margin-left:0.4rem;">AI Recruiter</span><br>
      <span style="font-size:0.75rem;color:#64748b;margin-left:2rem;">Redrob Hackathon</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    csv_exists = CSV_PATH.exists()
    if csv_exists:
        mtime = os.path.getmtime(CSV_PATH)
        ts = datetime.fromtimestamp(mtime).strftime("%d %b, %H:%M")
        st.markdown(f'<span class="status-dot-ok"></span><b>CSV loaded</b><br><span style="font-size:0.75rem;color:#64748b;padding-left:13px;">Generated: {ts}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot-err"></span><b style="color:#ef4444">No CSV found</b>', unsafe_allow_html=True)
        st.caption("Go to **Run Pipeline** tab to generate it.")

    st.divider()
    st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#94a3b8;letter-spacing:0.05em;">PENALTY LEGEND</div>', unsafe_allow_html=True)
    for key, meta in PENALTY_META.items():
        st.markdown(f'<span class="badge" style="background:{meta["color"]};">{meta["label"]}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Built with sentence-transformers + rule-based scoring · All offline")


# ── Load data ─────────────────────────────────────────────────────────────────
df = None
if CSV_PATH.exists():
    df = load_csv(CSV_PATH)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠  Overview",
    "📋  Ranked Candidates",
    "📊  Analytics",
    "⚙️  Run Pipeline",
    "📖  Methodology",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Hero
    st.markdown("""
    <div class="hero">
      <h1>🎯 AI Recruiter Ranking System</h1>
      <p>Hybrid Semantic Embedding + 23-Signal Behavioral Scoring + 9-Filter Penalty Chain</p>
      <p style="margin-top:0.8rem;font-size:0.85rem;opacity:0.8;">
        100% offline · CPU-only · sentence-transformers all-MiniLM-L6-v2 · Redrob Hackathon 2025
      </p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.warning("No submission CSV found. Go to **Run Pipeline** to generate one.")
    else:
        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        penalized_count = df["penalized"].sum()
        k1.metric("Total Ranked",  len(df))
        k2.metric("Top Score",     f"{df['score'].max():.4f}")
        k3.metric("Avg Score",     f"{df['score'].mean():.4f}",
                  delta=f"{df['score'].std():.4f} σ")
        k4.metric("Penalized",     int(penalized_count),
                  delta=f"{100*penalized_count/len(df):.0f}%", delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns(2)

        with left:
            st.markdown('<div class="section-title">Score Distribution</div>', unsafe_allow_html=True)
            if HAS_PLOTLY:
                fig = px.histogram(
                    df, x="score", nbins=25,
                    color_discrete_sequence=["#6366f1"],
                    labels={"score": "Composite Score", "count": "Candidates"},
                    template="simple_white",
                )
                fig.update_layout(
                    paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(l=0, r=0, t=10, b=0), height=280,
                    showlegend=False,
                    xaxis=dict(gridcolor="#f1f5f9"),
                    yaxis=dict(gridcolor="#f1f5f9"),
                )
                fig.update_traces(marker_line_color="white", marker_line_width=1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df["score"].value_counts(bins=20).sort_index())

        with right:
            st.markdown('<div class="section-title">Penalty Breakdown</div>', unsafe_allow_html=True)
            # Count each penalty type
            penalty_counts = {}
            for penalties in df["penalties"]:
                for p in penalties:
                    penalty_counts[p] = penalty_counts.get(p, 0) + 1

            if penalty_counts and HAS_PLOTLY:
                labels = [PENALTY_META.get(k, {"label": k})["label"] for k in penalty_counts]
                colors = [PENALTY_META.get(k, {"color": "#64748b"})["color"] for k in penalty_counts]
                fig2 = go.Figure(go.Pie(
                    labels=labels,
                    values=list(penalty_counts.values()),
                    hole=0.55,
                    marker_colors=colors,
                    textinfo="label+percent",
                    textfont_size=11,
                ))
                fig2.update_layout(
                    paper_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0),
                    height=280, showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)
            elif penalty_counts:
                st.bar_chart(pd.Series(penalty_counts))
            else:
                st.info("No penalties triggered.")

        # Top 5 preview
        st.markdown('<div class="section-title">Top 5 Candidates</div>', unsafe_allow_html=True)
        for _, row in df.head(5).iterrows():
            badges = penalty_badges(row["penalties"])
            st.markdown(f"""
            <div class="rank-card">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.4rem;">
                <div>
                  <span class="rank-num">#{row['rank']}</span>
                  &nbsp;<span class="cand-id">{row['candidate_id']}</span>
                </div>
                <span class="score-val">Score: {row['score']:.4f}</span>
              </div>
              <div style="margin:0.45rem 0 0.2rem;">{badges}</div>
              <div class="reasoning">{row['reasoning']}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RANKED CANDIDATES
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title" style="margin-top:0;">Candidate Rankings</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("No submission CSV found.")
    else:
        # Filter controls
        fc1, fc2, fc3, fc4 = st.columns([2, 1.5, 1.2, 1])
        with fc1:
            search = st.text_input("🔍 Search Candidate ID", placeholder="e.g. cand_00042", label_visibility="collapsed")
        with fc2:
            pen_filter = st.selectbox("Filter Penalty", ["All"] + list(PENALTY_META.keys()), label_visibility="collapsed")
        with fc3:
            score_min, score_max = float(df["score"].min()), float(df["score"].max())
            score_thresh = st.slider("Min Score", score_min, score_max, score_min, 0.001, label_visibility="collapsed")
        with fc4:
            page_size = st.selectbox("Per page", [10, 25, 50, 100], label_visibility="collapsed")

        # Apply
        filtered = df.copy()
        if search.strip():
            filtered = filtered[filtered["candidate_id"].str.contains(search.strip(), case=False, na=False)]
        if pen_filter != "All":
            filtered = filtered[filtered["penalties"].apply(lambda ps: pen_filter in ps)]
        filtered = filtered[filtered["score"] >= score_thresh]

        # Pagination
        total = len(filtered)
        total_pages = max(1, (total - 1) // page_size + 1)
        page_col, _, count_col = st.columns([1, 3, 1])
        with page_col:
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, label_visibility="collapsed")
        with count_col:
            st.markdown(f'<div style="text-align:right;color:#64748b;font-size:0.85rem;padding-top:0.5rem;">{total} results</div>', unsafe_allow_html=True)

        start = (page - 1) * page_size
        page_df = filtered.iloc[start : start + page_size]

        # Table rows with expander
        for _, row in page_df.iterrows():
            badges = penalty_badges(row["penalties"])
            with st.expander(
                f"#{row['rank']}  ·  {row['candidate_id']}  ·  Score {row['score']:.4f}",
                expanded=False,
            ):
                ec1, ec2 = st.columns([2, 1])
                with ec1:
                    st.markdown(f"""
                    <div style="font-size:0.9rem;color:#1e293b;">
                      <b>Reasoning:</b><br>
                      <span style="color:#475569;">{row['reasoning']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"**Penalties:** {badges}", unsafe_allow_html=True)
                with ec2:
                    st.markdown(f"""
                    <div class="card" style="font-size:0.85rem;">
                      <div><b>Rank:</b> #{row['rank']}</div>
                      <div><b>Score:</b> <span style="color:#059669;font-weight:700;">{row['score']:.4f}</span></div>
                      <div><b>Penalty Count:</b> {row['penalty_count']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.caption(f"Page {page} of {total_pages}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title" style="margin-top:0;">Analytics & Insights</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("No submission CSV found.")
    elif not HAS_PLOTLY:
        st.warning("Install plotly for interactive charts: `pip install plotly`")
    else:
        # Row 1: Penalty frequency + Score by penalty status
        a1, a2 = st.columns(2)

        with a1:
            st.markdown('<div class="section-title">Penalty Frequency (across top 100)</div>', unsafe_allow_html=True)
            penalty_counts = {}
            for penalties in df["penalties"]:
                for p in penalties:
                    meta = PENALTY_META.get(p, {"label": p, "color": "#64748b"})
                    penalty_counts[meta["label"]] = penalty_counts.get(meta["label"], 0) + 1

            if penalty_counts:
                pdf = pd.DataFrame(list(penalty_counts.items()), columns=["Penalty", "Count"])
                pdf = pdf.sort_values("Count", ascending=True)
                colors = [PENALTY_META.get(k, {"color": "#6366f1"})["color"]
                          for k in [list(PENALTY_META.keys())[list(PENALTY_META.values()).index(m)]
                                    for m in [{"label": l, "color": PENALTY_META.get(
                                        next((k for k, v in PENALTY_META.items() if v["label"] == row), "behavioral_downweight"),
                                        {"color": "#6366f1"})["color"]} for l, row in zip(pdf["Penalty"], pdf["Penalty"])]]]
                fig = px.bar(
                    pdf, x="Count", y="Penalty", orientation="h",
                    color="Penalty",
                    color_discrete_sequence=[m["color"] for m in PENALTY_META.values()],
                    template="simple_white",
                )
                fig.update_layout(
                    paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(l=0, r=0, t=10, b=0), height=320,
                    showlegend=False,
                    xaxis=dict(gridcolor="#f1f5f9"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No penalties in this dataset.")

        with a2:
            st.markdown('<div class="section-title">Score: Clean vs Penalized</div>', unsafe_allow_html=True)
            fig2 = px.box(
                df, x="penalized", y="score",
                color="penalized",
                color_discrete_map={False: "#6366f1", True: "#ef4444"},
                labels={"penalized": "Has Penalties", "score": "Composite Score"},
                template="simple_white",
                points="outliers",
            )
            fig2.update_layout(
                paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(l=0, r=0, t=10, b=0), height=320,
                showlegend=False,
                yaxis=dict(gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Row 2: Rank vs Score line + Penalty count vs Score scatter
        b1, b2 = st.columns(2)

        with b1:
            st.markdown('<div class="section-title">Score Decay by Rank</div>', unsafe_allow_html=True)
            fig3 = px.line(
                df.sort_values("rank"), x="rank", y="score",
                template="simple_white",
                labels={"rank": "Rank", "score": "Score"},
                color_discrete_sequence=["#6366f1"],
            )
            fig3.add_scatter(
                x=df[df["penalized"]]["rank"],
                y=df[df["penalized"]]["score"],
                mode="markers",
                marker=dict(color="#ef4444", size=6, symbol="x"),
                name="Penalized",
            )
            fig3.update_layout(
                paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(l=0, r=0, t=10, b=0), height=300,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(gridcolor="#f1f5f9"),
                yaxis=dict(gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig3, use_container_width=True)

        with b2:
            st.markdown('<div class="section-title">Penalty Count Distribution</div>', unsafe_allow_html=True)
            pen_dist = df["penalty_count"].value_counts().sort_index().reset_index()
            pen_dist.columns = ["penalty_count", "num_candidates"]
            fig4 = px.bar(
                pen_dist, x="penalty_count", y="num_candidates",
                color_discrete_sequence=["#8b5cf6"],
                template="simple_white",
                labels={"penalty_count": "# Penalties Applied", "num_candidates": "Candidates"},
            )
            fig4.update_layout(
                paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(l=0, r=0, t=10, b=0), height=300,
                showlegend=False,
                xaxis=dict(gridcolor="#f1f5f9", dtick=1),
                yaxis=dict(gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig4, use_container_width=True)

        # Summary table
        st.markdown('<div class="section-title">Score Summary Statistics</div>', unsafe_allow_html=True)
        stats = df["score"].describe().round(4)
        stat_df = pd.DataFrame({"Metric": stats.index, "Value": stats.values})
        st.dataframe(stat_df, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RUN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title" style="margin-top:0;">Run Ranking Pipeline</div>', unsafe_allow_html=True)

    rc1, rc2 = st.columns([3, 2])

    with rc1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Pipeline Configuration")

        cands_path_input = st.text_input(
            "Candidates file path (.jsonl)",
            value=str(CANDIDATES_PATH),
        )
        out_path_input = st.text_input(
            "Output CSV path",
            value=str(CSV_PATH),
        )

        cands_ok = Path(cands_path_input).exists()
        if cands_ok:
            size_mb = os.path.getsize(cands_path_input) / 1e6
            st.markdown(f'<span class="status-dot-ok"></span> File found ({size_mb:.0f} MB)', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-dot-err"></span> File not found', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        run_col, dl_col = st.columns(2)
        with run_col:
            run_btn = st.button("▶  Run rank.py", use_container_width=True, disabled=not cands_ok)
        with dl_col:
            if CSV_PATH.exists():
                with open(CSV_PATH, "rb") as f:
                    st.download_button(
                        "⬇  Download CSV",
                        data=f,
                        file_name="team_submission.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        if run_btn and cands_ok:
            cmd = [
                sys.executable,
                str(BASE / "rank.py"),
                "--candidates", cands_path_input,
                "--out", out_path_input,
            ]
            log_box = st.empty()
            progress = st.progress(0, text="Starting pipeline...")
            with st.spinner("Ranking candidates — this takes ~4 min for 100K rows..."):
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                lines = []
                for i, line in enumerate(proc.stdout):
                    lines.append(line.rstrip())
                    log_box.code("\n".join(lines[-20:]), language="bash")
                    # Rough progress estimate (50 batches for 100K at batch_size=2000)
                    pct = min(i / 52, 0.99)
                    progress.progress(pct, text=line.rstrip()[:80])
                proc.wait()

            if proc.returncode == 0:
                progress.progress(1.0, text="Complete!")
                st.success("✅ Pipeline complete! CSV saved.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("❌ Pipeline failed. See log above.")

    with rc2:
        st.markdown("""
        <div class="card">
          <div style="font-weight:700;color:#1e293b;margin-bottom:0.8rem;">📋 Pipeline Info</div>
          <div style="font-size:0.85rem;color:#475569;line-height:1.7;">
            <div>⏱ <b>Est. Runtime:</b> ~4 min (100K candidates, 8-core CPU)</div>
            <div>🌐 <b>Network:</b> None — 100% offline</div>
            <div>🖥 <b>GPU:</b> Not required (CPU-only)</div>
            <div>💾 <b>RAM:</b> ~2–3 GB peak</div>
            <div>📦 <b>Model:</b> all-MiniLM-L6-v2 (22M params)</div>
            <div>📊 <b>Output:</b> Top 100 ranked candidates</div>
          </div>
        </div>
        <br>
        <div class="card">
          <div style="font-weight:700;color:#1e293b;margin-bottom:0.8rem;">🖥 CLI Command</div>
          <code style="font-size:0.8rem;color:#6366f1;">
            python rank.py \\<br>
            &nbsp;&nbsp;--candidates candidates.jsonl \\<br>
            &nbsp;&nbsp;--out submission.csv
          </code>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title" style="margin-top:0;">How It Works</div>', unsafe_allow_html=True)

    # Scoring formula cards
    st.markdown("#### Composite Score Formula")
    st.markdown("""
    <div class="card" style="font-family:monospace;font-size:0.9rem;color:#1e293b;background:#fafafa;border:1px solid #e2e8f0;">
      composite_score = base_score &times; behavioral_penalty<br><br>
      base_score =<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Embedding Similarity &nbsp;&nbsp;&times; 0.50<br>
      &nbsp;&nbsp;+ Skill Assessment Match &times; 0.20<br>
      &nbsp;&nbsp;+ Engagement Signals &nbsp;&nbsp;&nbsp;&nbsp;&times; 0.20<br>
      &nbsp;&nbsp;+ Availability / Trust &nbsp;&nbsp;&times; 0.10<br><br>
      behavioral_penalty = ∏ (all applicable multipliers from penalty chain)
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, weight, label, desc, color in [
        (m1, "50%", "Embedding Similarity", "Semantic cosine match of role history vs JD", "#6366f1"),
        (m2, "20%", "Skill Assessment", "Redrob assessment scores × JD skill overlap", "#8b5cf6"),
        (m3, "20%", "Engagement Signals", "Response rate, activity recency, GitHub, interview completion", "#06b6d4"),
        (m4, "10%", "Availability / Trust", "Open-to-work, verified flags, notice period", "#059669"),
    ]:
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;border-top:4px solid {color};">
              <div style="font-size:2rem;font-weight:800;color:{color};">{weight}</div>
              <div style="font-weight:700;color:#1e293b;font-size:0.9rem;margin:0.3rem 0;">{label}</div>
              <div style="font-size:0.75rem;color:#64748b;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 9-Filter Penalty Chain")
    st.caption("All penalties are **multiplicative** — they cannot be overridden by a high base score.")

    penalty_table = []
    multipliers = [0.50, 0.80, 0.50, 0.85, 0.70, 0.60, 0.30, 0.85, 0.75]
    triggers = [
        "Inactive >180d AND response rate <30%",
        "All roles at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini",
        "All roles are research / student / PhD / university",
        "Avg tenure <1.5yr across >2 companies",
        "Senior/Lead title with <4yr total experience",
        "3.0 ≤ years_exp < 4.0 (JD requires 5–9yr)",
        "years_exp < 3.0",
        "'Learning', 'enthusiast', 'still building' in descriptions",
        "No Pinecone/FAISS/Elasticsearch/NDCG/MRR in history",
    ]
    for (key, meta), mult, trigger in zip(PENALTY_META.items(), multipliers, triggers):
        penalty_table.append({
            "Penalty": meta["label"],
            "Multiplier": f"×{mult}",
            "Trigger": trigger,
        })

    pen_df = pd.DataFrame(penalty_table)
    st.dataframe(pen_df, hide_index=True, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Architecture Pipeline")

    steps = [
        ("1", "Load", "Stream candidates.jsonl line-by-line\n(no RAM spike for 487MB file)"),
        ("2", "Extract", "Deterministic parse →\nPydantic CandidateProfile\n(23 redrob signals)"),
        ("3", "Embed", "Build role-history-focused\nsummary string →\nall-MiniLM-L6-v2"),
        ("4", "Score", "Cosine similarity +\nEngagement + Trust +\nSkill Assessment"),
        ("5", "Penalize", "Apply 9-filter\nmultiplicative\npenalty chain"),
        ("6", "Output", "Sort top 100 →\nCSV: candidate_id\nrank, score, reasoning"),
    ]
    cols = st.columns(len(steps))
    for col, (num, name, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="flow-step">
              <div class="step-num">{num}</div>
              <div class="step-name">{name}</div>
              <div class="step-desc" style="white-space:pre-line;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Why Multiplicative Penalties?")
    st.markdown("""
    <div class="card" style="border-left: 4px solid #6366f1;">
      <p style="color:#1e293b;font-size:0.9rem;margin:0;">
        Additive scoring allows a high embedding match to mask a hard disqualifier — a candidate with
        <code>&lt;3 years experience</code> could still rank highly purely on keyword density.
        <br><br>
        Multiplicative penalties enforce a <b>ceiling</b>: no matter how high the base score,
        stacking <code>exp_below_3yr (×0.30)</code> + <code>no_vector_db_or_eval (×0.75)</code>
        + <code>behavioral_downweight (×0.50)</code> = effective multiplier of <b>×0.1125</b> —
        making it mathematically impossible to breach the top tier.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Embedding Strategy: Role History > Skills List")
    st.markdown("""
    <div class="card" style="border-left: 4px solid #8b5cf6;">
      <p style="color:#475569;font-size:0.85rem;margin:0 0 0.5rem;"><b>❌ Old approach (keyword stuffing reward):</b></p>
      <code style="font-size:0.8rem;color:#64748b;">Skills: Python, ML, FAISS, Pinecone, NDCG, Weaviate, RAG...</code>
      <p style="color:#475569;font-size:0.85rem;margin:0.8rem 0 0.5rem;"><b>✅ Our approach (semantic narrative):</b></p>
      <code style="font-size:0.8rem;color:#1e293b;">
        "Senior ML Engineer, 6.2yrs. Key Skills: Python, PyTorch, NLP.<br>
        Role History: ML Engineer at Razorpay: Built production recommendation engine
        serving 2M daily requests, evaluated with NDCG@10 and A/B tested over 6 months."
      </code>
      <p style="color:#475569;font-size:0.85rem;margin-top:0.8rem;">
        A candidate who <i>built</i> a production retrieval system but never listed "RAG" or "Pinecone"
        will outscore one who listed 30 keywords but has no described production output.
      </p>
    </div>
    """, unsafe_allow_html=True)
