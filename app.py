"""
Streamlit web app for signal_tool.

Run with:
    cd signal_tool
    pip install -r requirements.txt
    streamlit run app.py

The app provides an interactive UI for exploring signal pairs:
  - Sidebar: pick a pair, see metadata
  - Overview tab: hypothesis, headline metrics, verdict
  - Data tab: sources, aligned dataset
  - Pre-analysis tab: interactive 3-panel chart (level, 1st, 2nd derivative)
  - Correlation tab: lag correlation with adjustable settings
  - Compare tab: side-by-side comparison of all pairs
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Make sure we can import the signal_tool package
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from core import align, compute_derivatives, lag_correlation, peak_correlation  # noqa: E402
from pairs import PAIRS  # noqa: E402

# The screen module works whether the repo is laid out as a package
# (local) or flat (Streamlit Cloud deploy).
try:
    from signal_tool.screen import screen_all  # noqa: E402
except ModuleNotFoundError:  # flat deploy layout
    from screen import screen_all  # noqa: E402

try:
    from signal_tool.sources import load_fred_monthly  # noqa: E402
except ModuleNotFoundError:  # flat deploy layout
    from sources import load_fred_monthly  # noqa: E402

# =================================================================
# PAGE CONFIG
# =================================================================
st.set_page_config(
    page_title="Signal Tool",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

LEADING_COLOR = "#5B8FF9"
OUTCOME_COLOR = "#1F4E78"


# =================================================================
# DATA LOADING (cached so flipping between pairs is fast)
# =================================================================
@st.cache_data(show_spinner=False)
def load_pair_data(pair_name: str):
    pair = PAIRS[pair_name]
    x = pair.leading.load()
    y = pair.outcome.load()
    df = align(x, y, pair.frequency)
    df = compute_derivatives(df, pair.frequency)
    return df


@st.cache_data(show_spinner=False)
def compute_lag(pair_name: str, max_lag: int):
    pair = PAIRS[pair_name]
    df = load_pair_data(pair_name)
    df_yoy = df.dropna(subset=["x_d1", "y_d1"])
    return lag_correlation(df_yoy["x_d1"], df_yoy["y_d1"], max_lag)


@st.cache_data(show_spinner=False)
def run_screen_cached():
    """Run the false-signal screen on every pair (cached)."""
    return screen_all(PAIRS)


@st.cache_data(show_spinner=False)
def market_analysis():
    """Signal groups + market-control test. Returns (corr matrix, table, clusters)."""
    def yoy(s):
        return s.pct_change(12) * 100

    monthly = {n: p for n, p in PAIRS.items() if p.frequency == "monthly"}
    lead = {p.leading.name: yoy(p.leading.load()) for p in monthly.values()}
    outcome = yoy(load_fred_monthly("transformer_ppi"))
    market = yoy(load_fred_monthly("market_all_commodities"))
    C = pd.concat(lead, axis=1).dropna().corr()

    # cluster leading signals at correlation >= 0.70 (simple union-find, no scipy)
    names = list(C.columns)
    parent = {n: n for n in names}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if C.loc[a, b] >= 0.70:
                parent[find(a)] = find(b)
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    clusters = list(groups.values())

    def _c(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 4 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
            return np.nan
        return np.corrcoef(a[m], b[m])[0, 1]

    def _partial(a, b, z):
        rab, raz, rbz = _c(a, b), _c(a, z), _c(b, z)
        d = np.sqrt((1 - raz ** 2) * (1 - rbz ** 2))
        return (rab - raz * rbz) / d if d else np.nan

    rows = []
    for p in monthly.values():
        x = lead[p.leading.name]
        d = pd.concat([x.rename("x"), outcome.rename("y"), market.rename("z")], axis=1).dropna()
        if len(d) < 30:
            continue
        lag, r, _ = peak_correlation(lag_correlation(d["x"], d["y"], p.max_lag))
        dd = pd.concat([d["x"].rename("x"), d["y"].shift(-lag).rename("y"), d["z"].rename("z")], axis=1).dropna()
        raw = _c(dd["x"].values, dd["y"].values)
        par = _partial(dd["x"].values, dd["y"].values, dd["z"].values)
        rel = _c((dd["x"] - dd["z"]).values, dd["y"].values)
        rows.append({
            "Signal": p.leading.name,
            "Lead (mo)": int(lag),
            "Raw r": round(raw, 2),
            "Partial r (market held constant)": round(par, 2),
            "Relative r": round(rel, 2),
            "% of edge kept": f"{100 * abs(par) / abs(raw):.0f}%" if raw else "—",
        })
    table = pd.DataFrame(rows).sort_values(
        "Partial r (market held constant)", key=lambda s: s.abs(), ascending=False
    )
    return C, table, clusters


# Marks for a check's pass/fail/not-applicable state
CHECK_MARK = {True: "✅", False: "❌", None: "➖"}


# =================================================================
# SIDEBAR
# =================================================================
st.sidebar.title("📊 Signal Tool")
st.sidebar.markdown("Leading-indicator backtest framework")
st.sidebar.caption("Time-series signal pair analysis")
st.sidebar.divider()

pair_names = list(PAIRS.keys())
selected = st.sidebar.selectbox(
    "Signal pair",
    pair_names,
    format_func=lambda name: f"{name} ({PAIRS[name].frequency})",
)
pair = PAIRS[selected]

with st.sidebar.expander("Pair details", expanded=True):
    st.markdown(f"**Hypothesis:** {pair.hypothesis}")
    st.markdown(f"**Expected lead:** {pair.expected_lead_low:.0f}–{pair.expected_lead_high:.0f} {pair.lag_unit}")
    st.markdown(f"**Frequency:** {pair.frequency}")

st.sidebar.divider()
st.sidebar.caption(
    "Methodology: 4-section structure — data collection → processing → "
    "pre-analysis visualization → correlation. See README.md."
)


# =================================================================
# MAIN — HEADER + METRICS
# =================================================================
df = load_pair_data(selected)
lag_results = compute_lag(selected, pair.max_lag)
peak_lag, peak_r, peak_n = peak_correlation(lag_results)

st.title(f"📊 {selected}")
st.markdown(f"**Hypothesis:** *{pair.hypothesis}*")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Peak correlation", f"{peak_r:+.3f}" if not np.isnan(peak_r) else "—")
col2.metric("Measured lead", f"{peak_lag:+d} {pair.lag_unit}")
col3.metric("Sample at peak", f"{peak_n}")
col4.metric("Total observations", f"{len(df)}")

# Robustness-screen result for this pair (computed once, reused in the tabs below)
screen_results = run_screen_cached()
sr = screen_results[selected]

# --- Hypothesis verdict ---
# CONFIRMED = strong + correct direction + passes the robustness screen.
# The lead time is REPORTED as a measured output, not required to match a prior guess.
prior = f"{pair.expected_lead_low:.0f}–{pair.expected_lead_high:.0f} {pair.lag_unit}"
if not np.isnan(peak_r) and peak_lag > 0 and abs(peak_r) >= 0.5 and sr.overall_pass:
    st.success(
        f"✅ **Hypothesis CONFIRMED** — strong correlation ({peak_r:+.2f}) in the predicted "
        f"direction, and it passes the robustness screen. Measured lead: **{peak_lag:+d} "
        f"{pair.lag_unit}** (prior guess was {prior})."
    )
elif not np.isnan(peak_r) and peak_lag > 0 and abs(peak_r) >= 0.5 and not sr.overall_pass:
    st.warning(
        f"⚠️ **Strong, but NOT robust** — the correlation is strong ({peak_r:+.2f}) and points the "
        "right way, but it fails the robustness screen, so it may be a false correlation. "
        "See the **Screen** tab."
    )
elif not np.isnan(peak_r) and peak_lag < 0 and abs(peak_r) >= 0.5:
    st.error(
        f"❌ **Hypothesis REJECTED — direction reversed.** The outcome leads the proposed signal "
        f"({peak_r:+.2f} at {peak_lag:+d} {pair.lag_unit}), not the other way around. "
        "Worth testing the flipped pair."
    )
elif not np.isnan(peak_r) and abs(peak_r) >= 0.4 and peak_lag > 0:
    st.warning(
        f"⚠️ **Hypothesis PARTIALLY supported** — only a moderate correlation ({peak_r:+.2f}); "
        "treat with caution."
    )
elif not np.isnan(peak_r) and abs(peak_r) < 0.3:
    st.error("❌ **Hypothesis REJECTED** — correlation too weak at any tested lag.")
else:
    st.info("ℹ️ **Hypothesis INCONCLUSIVE** — mixed or weak signal.")

# --- Robustness-screen badge ---
if sr.overall_pass:
    st.markdown(
        "🛡️ **Correlation-robustness screen: PASSED** — clears the trust checks, "
        "so the correlation is unlikely to be a fluke. See the **Screen** tab."
    )
else:
    failed = ", ".join(sr.failed_checks) if sr.failed_checks else "it does not actually lead the outcome"
    st.markdown(
        f"🚩 **Correlation-robustness screen: FLAGGED** — {failed}. See the **Screen** tab."
    )


# =================================================================
# TABS
# =================================================================
tab_data, tab_pre, tab_corr, tab_screen, tab_compare, tab_groups = st.tabs(
    ["📋 Data", "📈 Pre-Analysis", "🔄 Correlation", "🛡️ Screen", "🔀 Compare pairs", "🧬 Groups & Market"]
)


# ---- Tab: Data ----
with tab_data:
    st.header("Section 1 — Data Sources")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Leading signal: `{pair.leading.name}`")
        st.markdown(f"**{pair.leading.description}**")
        st.markdown(f"- **Source:** [{pair.leading.source_url}]({pair.leading.source_url})")
        st.markdown(f"- **Units:** {pair.leading.units}")
    with c2:
        st.subheader(f"Outcome signal: `{pair.outcome.name}`")
        st.markdown(f"**{pair.outcome.description}**")
        st.markdown(f"- **Source:** [{pair.outcome.source_url}]({pair.outcome.source_url})")
        st.markdown(f"- **Units:** {pair.outcome.units}")

    st.divider()
    st.subheader("Aligned dataset")
    st.caption(
        f"Frequency: {pair.frequency} | "
        f"Range: {df.index.min().date()} → {df.index.max().date()} | "
        f"N = {len(df)}"
    )
    st.dataframe(df.round(3), use_container_width=True, height=400)

    st.download_button(
        "Download aligned data as CSV",
        df.to_csv(),
        file_name=f"{selected}_aligned.csv",
        mime="text/csv",
    )


# ---- Tab: Pre-Analysis ----
with tab_pre:
    st.header("Section 3 — Pre-analysis Visualization")
    st.caption(
        "Three derivative orders side by side. Look for synchronized peaks in any panel "
        "before running correlation — the 2nd derivative often reveals inflection points "
        "the 1st derivative obscures."
    )

    show_d2 = st.checkbox("Show 2nd derivative panel", value=True)
    overlay_band = st.checkbox(
        "Highlight expected lead range on time axis",
        value=False,
        help="Visual aid: shades a band corresponding to the hypothesized lead range",
    )

    n_rows = 3 if show_d2 else 2
    subplot_titles = [
        "LEVEL (0th derivative) — where each variable is",
        "1st DERIVATIVE (YoY %) — how fast each is moving (velocity)",
    ]
    if show_d2:
        subplot_titles.append("2nd DERIVATIVE (Δ YoY %) — whether each is accelerating or decelerating")

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] * (n_rows - 1),
    )

    # Panel 1: Level (dual y-axis)
    fig.add_trace(
        go.Scatter(x=df.index, y=df["x"], name=pair.leading.name,
                   line=dict(color=LEADING_COLOR, width=1.6)),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["y"], name=pair.outcome.name,
                   line=dict(color=OUTCOME_COLOR, width=1.6)),
        row=1, col=1, secondary_y=True,
    )
    fig.update_yaxes(title_text=pair.leading.units.split(",")[0], color=LEADING_COLOR, row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text=pair.outcome.units.split(",")[0], color=OUTCOME_COLOR, row=1, col=1, secondary_y=True)

    # Panel 2: 1st derivative
    fig.add_trace(
        go.Scatter(x=df.index, y=df["x_d1"], name=f"{pair.leading.name} YoY %",
                   line=dict(color=LEADING_COLOR, width=1.4)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["y_d1"], name=f"{pair.outcome.name} YoY %",
                   line=dict(color=OUTCOME_COLOR, width=1.6)),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_yaxes(title_text="YoY % change", row=2, col=1)

    # Panel 3: 2nd derivative
    if show_d2:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["x_d2"], name=f"{pair.leading.name} 2nd deriv",
                       line=dict(color=LEADING_COLOR, width=1.2)),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["y_d2"], name=f"{pair.outcome.name} 2nd deriv",
                       line=dict(color=OUTCOME_COLOR, width=1.4)),
            row=3, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        fig.update_yaxes(title_text="Change in YoY %", row=3, col=1)

    fig.update_layout(
        height=350 * n_rows,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=80),
    )
    fig.update_xaxes(title_text="Date", row=n_rows, col=1)
    st.plotly_chart(fig, use_container_width=True)


# ---- Tab: Correlation ----
with tab_corr:
    st.header("Section 4 — Lag Correlation")
    st.caption(
        f"Tests how strongly {pair.leading.name}'s YoY % change correlates with "
        f"{pair.outcome.name}'s YoY % change at different leads/lags. "
        "Positive lag = leading signal moves earlier."
    )

    max_lag = st.slider(
        "Max lag to test",
        min_value=3, max_value=36, value=pair.max_lag,
        help="Window size (months for monthly pairs, years for annual)",
    )
    lag_results = compute_lag(selected, max_lag)
    peak_lag, peak_r, peak_n = peak_correlation(lag_results)

    lags = [r[0] for r in lag_results]
    corrs = [r[1] for r in lag_results]
    ns = [r[2] for r in lag_results]
    colors = [OUTCOME_COLOR if c == peak_r else "#9BB7D4" for c in corrs]

    fig_corr = go.Figure(
        data=[
            go.Bar(
                x=lags, y=corrs, marker_color=colors,
                customdata=ns,
                hovertemplate=(
                    f"Lag: %{{x}} {pair.lag_unit}<br>"
                    "r: %{y:+.3f}<br>"
                    "n: %{customdata}<extra></extra>"
                ),
            )
        ]
    )
    fig_corr.add_hline(y=0, line_color="black", line_width=0.5)

    # Shade expected lead band
    fig_corr.add_vrect(
        x0=pair.expected_lead_low, x1=pair.expected_lead_high,
        fillcolor="green", opacity=0.08,
        annotation_text="Expected lead range",
        annotation_position="top left",
    )

    fig_corr.update_layout(
        xaxis_title=f"{pair.lag_unit.capitalize()} {pair.leading.name} leads {pair.outcome.name}",
        yaxis_title="Correlation (r)",
        height=500,
        hovermode="x unified",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Lag correlation table")
    lag_df = pd.DataFrame(lag_results, columns=["lag", "r", "n"])
    lag_df["expected"] = (lag_df["lag"] >= pair.expected_lead_low) & (lag_df["lag"] <= pair.expected_lead_high)
    st.dataframe(
        lag_df.style.format({"r": "{:+.3f}"})
        .background_gradient(subset=["r"], cmap="RdBu_r", vmin=-1, vmax=1),
        use_container_width=True,
        height=400,
    )


# ---- Tab: Screen ----
with tab_screen:
    st.header("🛡️ Correlation-robustness screen")
    st.caption(
        "A strong correlation can still be a fluke (a 'spurious' or false correlation). "
        "This screen tests whether the correlation is genuine and usable, using four checks. "
        "Two are disqualifying gates; two add confidence but never flag a pair on their own."
    )

    with st.expander("What the four checks mean", expanded=False):
        st.markdown(
            "**Disqualifying gates** (a pair fails the screen if either fails):\n\n"
            "- **Survives smoothing** — If we smooth out the random jitter, is the "
            "pattern still there? Real patterns survive; noise washes out.\n"
            "- **Holds over time** — Split the history into an early half and a late "
            "half. Does the relationship show up in both? A real link keeps recurring.\n\n"
            "**Confirmatory checks** (they build confidence, but a 'no' only means "
            "'not yet shown', not 'failed'):\n\n"
            "- **Second opinion** — Is there another, independent signal pointing the "
            "same way at a similar lead? Becomes meaningful as the signal library grows.\n"
            "- **Has a reason** — Is there a written, common-sense reason the two things "
            "should be connected? (A human still judges whether it's sensible.)"
        )

    # --- Selected pair, detailed ---
    st.subheader(f"This pair: `{selected}`")
    if sr.overall_pass:
        st.success("✅ **PASSES the screen** — clears all four checks and leads the outcome.")
    else:
        st.error(
            "🚩 **FLAGGED** — "
            + ("does not actually lead the outcome. " if not sr.leads else "")
            + ("Failed checks: " + ", ".join(sr.failed_checks) if sr.failed_checks else "")
        )
    st.caption(
        f"Peak correlation r={sr.peak_r:+.3f} at a lead of {sr.peak_lag:+.0f} "
        f"{pair.lag_unit} · leads the outcome: {'yes' if sr.leads else 'NO'}"
    )
    for c in sr.checks:
        st.markdown(f"{CHECK_MARK[c.passed]} **{c.name}** — {c.detail}")

    st.divider()

    # --- All pairs, at a glance ---
    st.subheader("All signal pairs at a glance")
    st.caption("✅ pass · ❌ fail · ➖ not applicable (usually too few data points)")
    screen_rows = []
    for name, r in screen_results.items():
        row = {
            "Pair": name,
            "Verdict": "✅ Passes" if r.overall_pass else "🚩 Flagged",
            "Leads?": "yes" if r.leads else "NO",
            "Peak r": f"{r.peak_r:+.3f}" if not np.isnan(r.peak_r) else "—",
        }
        for c in r.checks:
            row[c.name] = CHECK_MARK[c.passed]
        screen_rows.append(row)
    st.dataframe(pd.DataFrame(screen_rows), use_container_width=True, hide_index=True)


# ---- Tab: Compare pairs ----
with tab_compare:
    st.header("Side-by-side: all signal pairs")
    st.caption("Quick summary of every defined pair. Click any row to switch.")

    rows = []
    for name, p in PAIRS.items():
        df_p = load_pair_data(name)
        lags_p = compute_lag(name, p.max_lag)
        lag, r, n = peak_correlation(lags_p)
        passed_p = screen_results[name].overall_pass
        verdict = (
            "✅ Confirmed"
            if (not np.isnan(r) and abs(r) >= 0.5 and lag > 0 and passed_p)
            else "⚠️ Strong / not robust"
            if (not np.isnan(r) and abs(r) >= 0.5 and lag > 0)
            else "❌ Reversed"
            if (not np.isnan(r) and lag < 0 and abs(r) >= 0.5)
            else "⚠️ Partial"
            if (not np.isnan(r) and abs(r) >= 0.4 and lag > 0)
            else "❌ Rejected"
            if (not np.isnan(r) and abs(r) < 0.3)
            else "ℹ️ Inconclusive"
        )
        rows.append({
            "Pair": name,
            "Leading": p.leading.name,
            "Outcome": p.outcome.name,
            "Frequency": p.frequency,
            "Peak r": f"{r:+.3f}" if not np.isnan(r) else "—",
            "Measured lead": f"{lag:+d} {p.lag_unit}",
            "Prior guess": f"{p.expected_lead_low:.0f}–{p.expected_lead_high:.0f} {p.lag_unit}",
            "N (total)": len(df_p),
            "Verdict": verdict,
        })

    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Visual comparison: lag correlation curves overlaid")
    st.caption("All pairs on one axis — see which has the strongest, cleanest peak")

    fig_overlay = go.Figure()
    palette = ["#1F4E78", "#C9750D", "#0A6622", "#A6201F", "#5B2F90"]
    for i, (name, p) in enumerate(PAIRS.items()):
        lags_p = compute_lag(name, p.max_lag)
        lag_vals = [r[0] for r in lags_p]
        corr_vals = [r[1] for r in lags_p]
        fig_overlay.add_trace(go.Scatter(
            x=lag_vals, y=corr_vals,
            mode="lines+markers",
            name=name,
            line=dict(color=palette[i % len(palette)], width=2),
        ))
    fig_overlay.add_hline(y=0, line_color="black", line_width=0.5)
    fig_overlay.update_layout(
        xaxis_title="Lag (positive = leading)",
        yaxis_title="Correlation (r)",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig_overlay, use_container_width=True)


# ---- Tab: Groups & Market ----
with tab_groups:
    st.header("🧬 Signal groups & the market test")
    st.caption(
        "Two questions: (1) which signals are secretly the same thing, and "
        "(2) does each signal really beat the overall commodity market, or is it "
        "just riding the tide?"
    )
    C_mat, mkt_table, clusters = market_analysis()

    st.subheader("1. Which signals are the same underlying force?")
    st.caption("Signals that move together ≥ 0.70 are grouped — they count as ONE witness, not several.")
    multi = [g for g in clusters if len(g) > 1]
    if multi:
        for g in multi:
            st.markdown(f"- **Same force:** {', '.join(g)}")
    singles = [g[0] for g in clusters if len(g) == 1]
    if singles:
        st.markdown(f"- **Stand on their own:** {', '.join(singles)}")

    fig_h = go.Figure(data=go.Heatmap(
        z=C_mat.values, x=list(C_mat.columns), y=list(C_mat.index),
        zmin=-1, zmax=1, colorscale="RdBu_r",
        text=C_mat.round(2).values, texttemplate="%{text}", textfont={"size": 9},
        colorbar=dict(title="corr"),
    ))
    fig_h.update_layout(height=520, title="How correlated the leading signals are with each other")
    st.plotly_chart(fig_h, use_container_width=True)

    # optional richer family-tree image if it was generated by market_analysis.py
    dpath = HERE / "output" / "signal_dendrogram.png"
    if dpath.exists():
        with st.expander("Family tree (dendrogram)"):
            st.image(str(dpath))

    st.divider()
    st.subheader("2. Does each signal beat the overall market?")
    st.caption(
        "Raw = plain predictive power. Partial = power after the broad commodity market is held "
        "constant. If Partial stays close to Raw (high % kept), the signal has its own edge; if it "
        "collapses, the signal was mostly riding the commodity cycle."
    )
    st.dataframe(mkt_table, use_container_width=True, hide_index=True)
    st.caption(
        "Reading: copper keeps the most of its own edge; oil keeps the least (it's mostly a "
        "commodity-cycle passenger, not a transformer-specific driver)."
    )


# =================================================================
# FOOTER
# =================================================================
st.divider()
st.caption(
    f"signal_tool v0 | Generated reports available in `signal_tool/output/`. "
    f"To add a new signal pair, edit `pairs.py`. To run the static reports, "
    f"`python run.py`."
)
