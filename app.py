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
from signal_tool import (  # noqa: E402
    PAIRS,
    align,
    compute_derivatives,
    lag_correlation,
    peak_correlation,
)

# The screen module works whether the repo is laid out as a package
# (local) or flat (Streamlit Cloud deploy).
try:
    from signal_tool.screen import (  # noqa: E402
        screen_all,
        check_survives_smoothing,
        check_holds_over_time,
        check_lead_sharpness,
    )
except ModuleNotFoundError:  # flat deploy layout
    from screen import (  # noqa: E402
        screen_all,
        check_survives_smoothing,
        check_holds_over_time,
        check_lead_sharpness,
    )

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
# CLUSTERING (pure-Python average linkage, matches the scripts; no scipy)
# =================================================================
def average_linkage_clusters(C: pd.DataFrame, cut: float = 0.70):
    """Agglomerative AVERAGE-linkage clustering on a correlation matrix.

    Repeatedly merges the two clusters whose MEAN pairwise correlation is highest,
    stopping once no remaining pair reaches `cut`. Average linkage (not single-link
    union-find) so we don't chain weakly-related signals into one grab-bag — this
    matches perforce.py / force_tiers.py exactly.
    """
    clusters = [[n] for n in C.columns]

    def avg_corr(a, b):
        vals = [C.loc[x, y] for x in a for y in b]
        return sum(vals) / len(vals)

    while len(clusters) > 1:
        best, bi, bj = None, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                m = avg_corr(clusters[i], clusters[j])
                if best is None or m > best:
                    best, bi, bj = m, i, j
        if best is None or best < cut:
            break
        clusters[bi] = clusters[bi] + clusters.pop(bj)
    return [sorted(c) for c in clusters]


# ---- lead-lag map: which item leads which OTHER item (propagation chain) ----
def _corr_at(a, b, k):
    pair = (pd.concat([a, b.shift(-k)], axis=1).dropna() if k >= 0
            else pd.concat([a.shift(k), b], axis=1).dropna())
    if len(pair) < 8:
        return np.nan
    return pair.iloc[:, 0].corr(pair.iloc[:, 1])


def _best_lead(a, b, max_lag=24):
    r0 = _corr_at(a, b, 0)
    best_k, best_r = 0, 0.0
    for k in range(1, max_lag + 1):
        r = _corr_at(a, b, k)
        if not np.isnan(r) and abs(r) > abs(best_r):
            best_k, best_r = k, r
    return best_k, best_r, (0.0 if np.isnan(r0) else r0)


def lead_lag_edges(series, r_min=0.5, gain_min=0.05, max_lag=24):
    """Directed edges A→B where A genuinely LEADS B (positive lag, strong, sharp)."""
    names = list(series)
    edges = []
    for a in names:
        for b in names:
            if a == b:
                continue
            lead, r, r0 = _best_lead(series[a], series[b], max_lag)
            if lead >= 1 and abs(r) >= r_min and (abs(r) - abs(r0)) >= gain_min:
                edges.append({"leader": a, "follower": b, "lead": int(lead),
                              "r": round(float(r), 3), "gain": round(abs(r) - abs(r0), 3)})
    return edges


def leadlag_figure(series, edges, outcome):
    """Left→right propagation graph: sources (leads, not led) left, sinks (led) right."""
    from collections import defaultdict
    nodes = list(series)
    outc = {n: 0 for n in nodes}
    inc = {n: 0 for n in nodes}
    for e in edges:
        outc[e["leader"]] += 1
        inc[e["follower"]] += 1
    xval = {n: inc[n] - outc[n] for n in nodes}   # sinks (right) have high inc
    cols = defaultdict(list)
    for n in sorted(nodes, key=lambda n: (xval[n], n)):
        cols[xval[n]].append(n)
    pos = {}
    for x, ns in cols.items():
        for i, n in enumerate(ns):
            pos[n] = (x, i - (len(ns) - 1) / 2)
    edge_traces, ann = [], []
    for e in edges:
        x0, y0 = pos[e["leader"]]
        x1, y1 = pos[e["follower"]]
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=1 + 3 * abs(e["r"]), color="rgba(120,120,120,0.30)"),
            hoverinfo="text",
            text=f'{e["leader"]} → {e["follower"]} · {e["lead"]}mo · r={e["r"]:+.2f}',
            showlegend=False))
        ann.append(dict(ax=x0, ay=y0, x=x1, y=y1, xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1,
                        arrowcolor="rgba(80,80,80,0.45)"))
    colors = []
    for n in nodes:
        if n == outcome:
            colors.append("#1F4E78")
        elif inc[n] and not outc[n]:
            colors.append("#A6201F")   # sink
        elif outc[n] and not inc[n]:
            colors.append("#0A6622")   # source
        elif inc[n] and outc[n]:
            colors.append("#C9750D")   # relay
        else:
            colors.append("#9AA5B1")   # isolated
    node_trace = go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
        mode="markers+text",
        marker=dict(size=[13 + 4 * inc[n] for n in nodes], color=colors, line=dict(width=1, color="white")),
        text=[n.replace("_", " ") for n in nodes], textposition="middle right", textfont=dict(size=11),
        hovertext=[f"{n}: leads {outc[n]}, led by {inc[n]}" for n in nodes],
        hoverinfo="text", showlegend=False)
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        annotations=ann, height=560, margin=dict(l=10, r=160, t=30, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def complete_linkage_split(C: pd.DataFrame, members, cut: float = 0.70):
    """Re-cluster `members` with COMPLETE linkage: two groups merge only if their
    WEAKEST cross-pair still clears `cut`. Used to break an opposite-sign grab-bag
    into its genuinely tight pieces (the in-UI version of force_tiers.split_force)."""
    clusters = [[m] for m in members]

    def link(a, b):  # complete linkage distance = weakest pair
        return min(C.loc[x, y] for x in a for y in b)

    while len(clusters) > 1:
        best, bi, bj = None, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                m = link(clusters[i], clusters[j])
                if best is None or m > best:
                    best, bi, bj = m, i, j
        if best is None or best < cut:
            break
        clusters[bi] = clusters[bi] + clusters.pop(bj)
    return [sorted(c) for c in clusters]


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

    # cluster leading signals TIGHT by construction (complete linkage, no scipy)
    clusters = complete_linkage_split(C, list(C.columns), 0.70)

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


# Force tiers (Andrew's three categories)
TIER_TIGHT = "TIGHT"
TIER_LOOSE_SAME = "LOOSE / SAME-SIGN"
TIER_LOOSE_OPP = "LOOSE / OPPOSITE-SIGN"
TIER_USABLE = {TIER_TIGHT, TIER_LOOSE_SAME}
# A force is only ever a tight cluster: both opposite-sign AND same-sign loose
# groups get split into their tight pieces (matches force_tiers.FORCES_MUST_BE_TIGHT).
FORCES_MUST_BE_TIGHT = True


@st.cache_data(show_spinner=False)
def force_level_analysis(tight=True):
    """Collapse correlated signals into forces (one witness each) and score each
    force the SAME way we score an individual signal: tier it, run the composite
    through the disqualifying gates, and market-control it.

    tight=True  → complete-linkage clusters (cohesive, granular; cousins remain).
    tight=False → average-linkage 'blocs' that merge cousins into bigger, MORE
                  INDEPENDENT groups (lower cross-correlation), splitting only
                  opposite-sign grab-bags. Returns (table, details)."""
    from types import SimpleNamespace

    def yoy(s):
        return s.pct_change(12) * 100

    monthly = {n: p for n, p in PAIRS.items() if p.frequency == "monthly"}
    lead = {p.leading.name: yoy(p.leading.load()) for p in monthly.values()}
    outcome = yoy(load_fred_monthly("transformer_ppi"))
    market = yoy(load_fred_monthly("market_all_commodities"))

    # Exclude short-sample members before forming forces: a short member truncates
    # the shared window of both the correlation matrix and any composite it joins.
    def _aligned_n(s):
        return len(pd.concat([s, outcome], axis=1).dropna())
    lead = {m: s for m, s in lead.items() if _aligned_n(s) >= 150}

    C = pd.concat(lead, axis=1).dropna().corr()

    # tight → complete linkage (every pair ≥0.70); loose → average linkage (merges cousins)
    clusters = (complete_linkage_split(C, list(C.columns), 0.70) if tight
                else average_linkage_clusters(C, 0.70))

    def _c(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 4 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
            return np.nan
        return np.corrcoef(a[m], b[m])[0, 1]

    def _partial(a, b, z):
        rab, raz, rbz = _c(a, b), _c(a, z), _c(b, z)
        d = np.sqrt((1 - raz ** 2) * (1 - rbz ** 2))
        return (rab - raz * rbz) / d if d else np.nan

    def tightness(members):
        if len(members) == 1:
            return 1.0
        return min(C.loc[a, b] for i, a in enumerate(members) for b in members[i + 1:])

    def member_sign(nm):
        lag, r, _ = peak_correlation(lag_correlation(lead[nm], outcome, 24))
        return 0 if np.isnan(r) else int(np.sign(r))

    def tier_of(members):
        tj = tightness(members)
        signs = {m: member_sign(m) for m in members}
        pos = sum(1 for s in signs.values() if s > 0)
        neg = sum(1 for s in signs.values() if s < 0)
        tot = pos + neg
        purity = 1.0 if tot == 0 else max(pos, neg) / tot
        if len(members) == 1 or tj >= 0.70:
            tier = TIER_TIGHT
        elif purity >= 1.0:
            tier = TIER_LOOSE_SAME
        else:
            tier = TIER_LOOSE_OPP
        return tier, tj, signs

    # Auto-split rule: a loose grab-bag is re-clustered tighter (complete linkage)
    # into its coherent pieces, which are then scored individually. With
    # FORCES_MUST_BE_TIGHT, BOTH opposite-sign and same-sign loose groups are split
    # so every force on the board is a genuinely tight cluster.
    resolved = []
    for members in sorted(clusters, key=len, reverse=True):
        tier0, _tj0, _s0 = tier_of(members)
        split_it = (tier0 == TIER_LOOSE_OPP) or (tight and tier0 == TIER_LOOSE_SAME)
        if split_it and len(members) >= 2:
            origin = "split-from-opposite" if tier0 == TIER_LOOSE_OPP else "split-from-loose"
            for sub in sorted(complete_linkage_split(C, members, 0.70), key=len, reverse=True):
                resolved.append((sub, origin))
        else:
            resolved.append((members, "direct"))

    rows, details = [], {}
    for members, origin in resolved:
        tier, tj, signs = tier_of(members)

        comp = pd.concat([lead[m] for m in members], axis=1).mean(axis=1)
        d = pd.concat([comp.rename("x"), outcome.rename("y"), market.rename("m")], axis=1).dropna()
        if len(d) < 30:
            continue
        lag, r, _ = peak_correlation(lag_correlation(d["x"], d["y"], 24))
        dd = pd.concat([d["x"].rename("x"), d["y"].shift(-lag).rename("y"), d["m"].rename("m")], axis=1).dropna()
        raw = _c(dd["x"].values, dd["y"].values)
        par = _partial(dd["x"].values, dd["y"].values, dd["m"].values)

        df_yoy = pd.DataFrame({"x_d1": d["x"].values, "y_d1": d["y"].values}, index=d.index)
        fake = SimpleNamespace(frequency="monthly", max_lag=24, name="force")
        raw_peak = (lag, r, len(d))
        checks = [
            check_survives_smoothing(fake, df_yoy, raw_peak),
            check_holds_over_time(fake, df_yoy, raw_peak),
            check_lead_sharpness(fake, df_yoy, raw_peak),
        ]
        gate_fail = [c.name for c in checks if c.passed is False]
        leads = lag >= 0
        if tier == TIER_LOOSE_OPP:
            screen, usable = "EXCLUDED", False
        else:
            gates_ok = leads and abs(r) >= 0.30 and not gate_fail
            # Only "cycle-driven" if the cycle actually STRIPS the edge: partial is
            # weak AND it dropped meaningfully from raw. A raw≈partial pair means the
            # cycle explained nothing (it's just at its own level), not cycle-driven.
            if gates_ok and abs(par) < 0.20 and (abs(raw) - abs(par)) >= 0.15:
                screen, usable = "CYCLE-DRIVEN", False
            else:
                usable = gates_ok
                screen = "PASS" if usable else "FAIL"

        is_split = origin in ("split-from-opposite", "split-from-loose")
        fname = members[0] if len(members) == 1 else " + ".join(m[:14] for m in members)
        if is_split:
            fname = "↳ " + fname   # visually mark a piece recovered from a split
        rows.append({
            "Force": fname,
            "Tier": tier,
            "Size": len(members),
            "Origin": "split" if is_split else "direct",
            "Lead (mo)": int(lag),
            "Raw r": round(raw, 2),
            "Partial r": round(par, 2),
            "Screen": screen,
            "Usable": usable,
            "Members": ", ".join(members),
        })
        details[fname] = {
            "members": members, "comp": comp, "outcome": outcome,
            "lag": int(lag), "r": float(r), "raw": float(raw), "par": float(par),
            "tier": tier, "tightness": float(tj), "checks": checks,
            "screen": screen, "usable": usable, "signs": signs, "origin": origin,
        }

    table = pd.DataFrame(rows)
    return table, details


# Marks for a check's pass/fail/not-applicable state
CHECK_MARK = {True: "✅", False: "❌", None: "➖"}
TIER_ICON = {TIER_TIGHT: "🟢", TIER_LOOSE_SAME: "🟡", TIER_LOOSE_OPP: "🔴"}

# How far the correlation must DROP after removing the cycle to call it "cycle-driven".
CYCLE_MIN_DROP = 0.15


def cycle_label(raw, par):
    """Market-control verdict that doesn't mislead on already-weak items.

    'Cycle-driven' should mean the commodity cycle REMOVED a real edge — partial is
    weak AND it fell a lot from raw. If raw ≈ partial, the cycle explained nothing;
    the item is simply at its own (often low) level, which is NOT cycle-driven."""
    if abs(par) >= 0.30:
        return "survives"
    if abs(par) >= 0.20:
        return "partly survives"
    if abs(raw) - abs(par) >= CYCLE_MIN_DROP:
        return "cycle-driven"
    return "not cycle-driven (weak on its own)"


def render_force_leaderboard(force_table):
    """Render a force leaderboard dataframe (used for both tight forces and loose blocs)."""
    disp = force_table.copy()
    disp["Tier"] = disp["Tier"].map(lambda t: f"{TIER_ICON[t]} {t}")
    disp["Screen"] = disp["Screen"].map(
        lambda s: {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "CYCLE-DRIVEN": "🌀 CYCLE-DRIVEN",
                   "EXCLUDED": "🔴 EXCLUDED"}.get(s, s))
    disp = disp.assign(_rank=force_table["Screen"].map({"PASS": 0, "CYCLE-DRIVEN": 1, "FAIL": 2, "EXCLUDED": 3})) \
               .sort_values(["_rank", "Partial r"], key=lambda c: c if c.name == "_rank" else c.abs(),
                            ascending=[True, False]).drop(columns="_rank")
    st.dataframe(disp[["Force", "Tier", "Size", "Origin", "Lead (mo)", "Raw r", "Partial r", "Screen", "Members"]],
                 use_container_width=True, hide_index=True)
    return int((force_table["Screen"] == "PASS").sum())


def force_cross_corr(details):
    """Mean absolute inter-force correlation — lower = more independent groupings."""
    comps = {fn: fd["comp"] for fn, fd in details.items() if fn != "transformer_ppi"}
    if len(comps) < 2:
        return np.nan, 0
    Cf = pd.concat(comps, axis=1).dropna().corr()
    vals = [abs(Cf.iloc[i, j]) for i in range(len(Cf)) for j in range(i + 1, len(Cf))]
    cousins = sum(1 for v in vals if 0.5 <= v < 0.7)
    return (float(np.mean(vals)) if vals else np.nan), cousins


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
# CONFIRMED = strong + correct direction + passes the robustness screen + keeps its
# edge after the broad commodity cycle is removed (partial |r| >= 0.20). A strong
# signal that is really just a commodity-cycle passenger is NOT confirmed.
prior = f"{pair.expected_lead_low:.0f}–{pair.expected_lead_high:.0f} {pair.lag_unit}"

# cycle test: partial r vs the broad commodity cycle (monthly signals only)
if pair.frequency == "monthly":
    _, _mkt_table, _ = market_analysis()
    _pmap = dict(zip(_mkt_table["Signal"], _mkt_table["Partial r (market held constant)"]))
    sig_partial = _pmap.get(pair.leading.name, np.nan)
    cycle_ok = (not np.isnan(sig_partial)) and abs(sig_partial) >= 0.20
else:
    sig_partial, cycle_ok = np.nan, True   # no commodity-cycle control for annual pairs

if not np.isnan(peak_r) and peak_lag > 0 and abs(peak_r) >= 0.5 and sr.overall_pass and cycle_ok:
    st.success(
        f"✅ **Hypothesis CONFIRMED** — strong correlation ({peak_r:+.2f}) in the predicted "
        f"direction, it passes the robustness screen, and it keeps its edge after the commodity "
        f"cycle is removed (partial r {sig_partial:+.2f}). Measured lead: **{peak_lag:+d} "
        f"{pair.lag_unit}** (prior guess was {prior})."
    )
elif not np.isnan(peak_r) and peak_lag > 0 and abs(peak_r) >= 0.5 and sr.overall_pass and not cycle_ok:
    st.warning(
        f"🌀 **STRONG BUT CYCLE-DRIVEN** — the correlation is strong ({peak_r:+.2f}), leads, and "
        f"passes the robustness screen, but its edge disappears once the broad commodity cycle is "
        f"removed (partial r {sig_partial:+.2f}). It rides the commodity tide rather than being "
        "transformer-specific, so it is not confirmed. See the **Groups & Market** tab."
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
st.caption(
    "**Three stacked layers, from most granular to most independent.** "
    "🔹 **Signal** — one individual series. "
    "🔸 **Force (tight)** — cohesive ≥0.70 clusters, one composite witness each (still cousins between them). "
    "🔶 **Independent blocs (loose)** — the looser grouping that merges cousins into bigger, more "
    "mutually-independent blocs, closest to a true count of independent bets. "
    "The 🔶 loose view appears where independence matters most (Compare & Correlation)."
)

tab_data, tab_pre, tab_corr, tab_screen, tab_compare, tab_map = st.tabs(
    ["📋 Data", "📈 Pre-Analysis", "🔄 Correlation", "🛡️ Screen", "🔀 Compare", "🗺️ Lead-lag map"]
)


# ---- Tab: Data ----
with tab_data:
    st.header("📋 Data")
    st.markdown("### 🔹 Signal level")
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

    st.divider()
    st.markdown("### 🔸 Force level")
    st.caption("A force's equal-weighted composite and the members that make it up.")
    _dt, _ddet = force_level_analysis(tight=True)
    _dpick = st.selectbox("Force", list(_ddet.keys()), key="data_force_pick")
    _dfd = _ddet[_dpick]
    st.caption(f"Composite of {len(_dfd['members'])}: {', '.join(_dfd['members'])}")
    _cdf = pd.concat([_dfd["comp"].rename("composite YoY %"),
                      _dfd["outcome"].rename("transformer YoY %")], axis=1).dropna()
    st.dataframe(_cdf.round(3), use_container_width=True, height=320)


# ---- Tab: Pre-Analysis ----
with tab_pre:
    st.header("📈 Pre-Analysis")
    st.markdown("### 🔹 Signal level")
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

    st.divider()
    st.markdown("### 🔸 Force level")
    st.caption("A force's equal-weighted composite (YoY %) against transformer PPI.")
    _pt, _pdet = force_level_analysis(tight=True)
    _ppick = st.selectbox("Force", list(_pdet.keys()), key="pre_force_pick")
    _pfd = _pdet[_ppick]
    _pcdf = pd.concat([_pfd["comp"].rename("Force composite (YoY %)"),
                       _pfd["outcome"].rename("Transformer PPI (YoY %)")], axis=1).dropna()
    figp = go.Figure()
    figp.add_trace(go.Scatter(x=_pcdf.index, y=_pcdf.iloc[:, 0], name="Force composite (YoY %)",
                              line=dict(color=LEADING_COLOR, width=1.8)))
    figp.add_trace(go.Scatter(x=_pcdf.index, y=_pcdf.iloc[:, 1], name="Transformer PPI (YoY %)",
                              line=dict(color=OUTCOME_COLOR, width=1.8)))
    figp.add_hline(y=0, line_dash="dash", line_color="gray")
    figp.update_layout(height=420, hovermode="x unified",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                       margin=dict(t=40))
    st.plotly_chart(figp, use_container_width=True)


# ---- Tab: Correlation (with Groups folded in) ----
with tab_corr:
    st.header("🔄 Correlation")

    # ---------- 🔹 Signal level ----------
    st.markdown("### 🔹 Signal level")
    st.caption(
        f"This signal's lag correlation with transformer PPI, plus how the signals "
        "correlate with each other (who is secretly the same force)."
    )

    max_lag = st.slider(
        "Max lag to test", min_value=3, max_value=36, value=pair.max_lag,
        help="Window size (months for monthly pairs, years for annual)",
    )
    lag_results = compute_lag(selected, max_lag)
    peak_lag, peak_r, peak_n = peak_correlation(lag_results)
    lags = [r[0] for r in lag_results]
    corrs = [r[1] for r in lag_results]
    ns = [r[2] for r in lag_results]
    colors = [OUTCOME_COLOR if c == peak_r else "#9BB7D4" for c in corrs]
    fig_corr = go.Figure(data=[go.Bar(
        x=lags, y=corrs, marker_color=colors, customdata=ns,
        hovertemplate=(f"Lag: %{{x}} {pair.lag_unit}<br>r: %{{y:+.3f}}<br>n: %{{customdata}}<extra></extra>"))])
    fig_corr.add_hline(y=0, line_color="black", line_width=0.5)
    fig_corr.add_vrect(x0=pair.expected_lead_low, x1=pair.expected_lead_high,
                       fillcolor="green", opacity=0.08,
                       annotation_text="Expected lead range", annotation_position="top left")
    fig_corr.update_layout(
        xaxis_title=f"{pair.lag_unit.capitalize()} {pair.leading.name} leads {pair.outcome.name}",
        yaxis_title="Correlation (r)", height=440, hovermode="x unified")
    st.plotly_chart(fig_corr, use_container_width=True)

    C_mat, mkt_table, clusters = market_analysis()
    st.markdown("**Which signals are secretly the same force?** (they move together ≥ 0.70)")
    for g in [g for g in clusters if len(g) > 1]:
        st.markdown(f"- **Same force:** {', '.join(g)}")
    singles = [g[0] for g in clusters if len(g) == 1]
    if singles:
        st.markdown(f"- **Stand alone:** {', '.join(singles)}")
    fig_h = go.Figure(data=go.Heatmap(
        z=C_mat.values, x=list(C_mat.columns), y=list(C_mat.index),
        zmin=-1, zmax=1, colorscale="RdBu_r",
        text=C_mat.round(2).values, texttemplate="%{text}", textfont={"size": 9},
        colorbar=dict(title="corr")))
    fig_h.update_layout(height=480, title="Inter-signal correlation")
    st.plotly_chart(fig_h, use_container_width=True)
    with st.expander("Does each signal beat the commodity market? (raw vs partial r)"):
        st.caption("Partial r holds the broad commodity market constant. If it stays near raw, the "
                   "signal has its own edge; if it collapses, it was riding the cycle.")
        st.dataframe(mkt_table, use_container_width=True, hide_index=True)

    st.divider()

    # ---------- 🔸 Force level ----------
    st.markdown("### 🔸 Force level")
    st.caption(
        "How the FORCES correlate with each other. Two *tight* forces can still be **cousins** "
        "(0.5–0.7) — that's why a force count is not a count of independent bets."
    )
    _ft, _fdet = force_level_analysis()
    comps = {fn: fd["comp"] for fn, fd in _fdet.items() if fn != "transformer_ppi"}
    if len(comps) >= 2:
        Cf = pd.concat(comps, axis=1).dropna().corr()
        fig_hf = go.Figure(data=go.Heatmap(
            z=Cf.values, x=list(Cf.columns), y=list(Cf.index),
            zmin=-1, zmax=1, colorscale="RdBu_r",
            text=Cf.round(2).values, texttemplate="%{text}", textfont={"size": 9},
            colorbar=dict(title="corr")))
        fig_hf.update_layout(height=480, title="Inter-force correlation")
        st.plotly_chart(fig_hf, use_container_width=True)
        cousins = [(a, b, Cf.loc[a, b]) for i, a in enumerate(Cf.columns)
                   for b in Cf.columns[i + 1:] if 0.5 <= abs(Cf.loc[a, b]) < 0.7]
        if cousins:
            st.warning(
                f"⚠️ **{len(cousins)} cousin pair(s)** correlate 0.5–0.7 — related but not the same "
                "force. Count them as partly-overlapping evidence, not independent bets."
            )
        else:
            st.info("No cousin pairs (0.5–0.7) among the tight forces — they're reasonably independent.")

    st.divider()

    # ---------- 🔶 Independent blocs (loose) ----------
    st.markdown("### 🔶 Independent blocs — loose grouping")
    st.caption(
        "The looser (average-linkage) grouping merges cousins into bigger blocs, so the blocs "
        "correlate *less* with each other — closer to a true count of independent bets."
    )
    _lt, _ldet = force_level_analysis(tight=False)
    lcomps = {fn: fd["comp"] for fn, fd in _ldet.items() if fn != "transformer_ppi"}
    if len(lcomps) >= 2:
        Cl = pd.concat(lcomps, axis=1).dropna().corr()
        fig_hl = go.Figure(data=go.Heatmap(
            z=Cl.values, x=list(Cl.columns), y=list(Cl.index),
            zmin=-1, zmax=1, colorscale="RdBu_r",
            text=Cl.round(2).values, texttemplate="%{text}", textfont={"size": 9},
            colorbar=dict(title="corr")))
        fig_hl.update_layout(height=480, title="Inter-bloc correlation (loose grouping)")
        st.plotly_chart(fig_hl, use_container_width=True)
        mt, ct = force_cross_corr(_fdet)
        ml, cl = force_cross_corr(_ldet)
        if not np.isnan(mt) and not np.isnan(ml):
            st.markdown(
                f"➡️ Mean inter-group |r| falls from **{mt:.2f}** (tight forces) to **{ml:.2f}** "
                f"(loose blocs); cousin pairs from **{ct}** to **{cl}**."
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

    # ===== Two stacked layers: the SAME screen, on a signal (top) then a force (below) =====

    # ---------- 🔹 Signal level ----------
    st.markdown("### 🔹 Signal level")
    st.caption("The screen run on one individual signal.")
    st.markdown(f"**This signal:** `{selected}`")
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

    with st.expander("All signals at a glance"):
        st.caption("✅ pass · ❌ fail · ➖ not applicable (usually too few data points)")
        screen_rows = []
        for name, r in screen_results.items():
            row = {
                "Signal": name,
                "Verdict": "✅ Passes" if r.overall_pass else "🚩 Flagged",
                "Leads?": "yes" if r.leads else "NO",
                "Peak r": f"{r.peak_r:+.3f}" if not np.isnan(r.peak_r) else "—",
            }
            for c in r.checks:
                row[c.name] = CHECK_MARK[c.passed]
            screen_rows.append(row)
        st.dataframe(pd.DataFrame(screen_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ---------- 🔸 Force level ----------
    st.markdown("### 🔸 Force level")
    st.caption(
        "The **identical** gates, run on a force — a tight cluster of correlated signals "
        "collapsed into one equal-weighted composite (**one witness, not many**). Same checks, "
        "coarser unit. Note: the force count is not a count of independent bets — cousins remain "
        "(see the partial r and the Groups tab's inter-force correlations)."
    )
    force_table, force_details = force_level_analysis()

    pick = st.selectbox("Force", list(force_details.keys()), key="screen_force_pick")
    fd = force_details[pick]
    if fd["screen"] == "PASS":
        st.success("✅ **PASSES the screen** — leads, strong, clears every gate, AND keeps its edge after the commodity cycle is removed.")
    elif fd["screen"] == "CYCLE-DRIVEN":
        st.warning("🌀 **STRONG BUT CYCLE-DRIVEN** — leads, strong, and passes the gates, but its edge disappears once the commodity cycle is removed (partial |r| < 0.20). Not transformer-specific.")
    elif fd["screen"] == "EXCLUDED":
        st.error("🔴 **EXCLUDED — opposite-sign grab-bag** (not a coherent force).")
    else:
        st.error("🚩 **FLAGGED** — see the failing gate(s) below.")
    tight_txt = (f"tight cluster of {len(fd['members'])} · weakest internal link "
                 f"r={fd['tightness']:+.2f}" if len(fd["members"]) > 1 else "single-signal force")
    st.caption(
        f"Composite peak r={fd['r']:+.3f} at a lead of {fd['lag']:+d} mo · "
        f"leads: {'yes' if fd['lag'] >= 0 else 'NO'} · {tight_txt}"
    )
    for c in fd["checks"]:
        st.markdown(f"{CHECK_MARK[c.passed]} **{c.name}** — {c.detail}")
    surv = cycle_label(fd["raw"], fd["par"])
    st.markdown(f"📉 **Beats the commodity cycle?** raw r = {fd['raw']:+.2f} → partial r = {fd['par']:+.2f} → **{surv}**")
    st.caption(
        f"'Cycle-driven' only applies when the correlation actually drops by ≥ {CYCLE_MIN_DROP:.2f} "
        "after the cycle is removed. If raw and partial are close, the cycle explained nothing — "
        "the item is just at its own level, not cycle-driven."
    )

    with st.expander("All forces at a glance"):
        st.caption("Each force scored once, with the same gates. 🟢 tight · 🟡 loose same-sign · 🔴 excluded")
        frows = []
        for _, r in force_table.iterrows():
            frows.append({
                "Force": r["Force"],
                "Tier": f"{TIER_ICON.get(r['Tier'], '')} {r['Tier']}",
                "Size": r["Size"],
                "Lead (mo)": r["Lead (mo)"],
                "Peak r": f"{r['Raw r']:+.2f}",
                "Partial r": f"{r['Partial r']:+.2f}",
                "Screen": {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "CYCLE-DRIVEN": "🌀 CYCLE-DRIVEN", "EXCLUDED": "🔴 EXCLUDED"}.get(r["Screen"], r["Screen"]),
            })
        st.dataframe(pd.DataFrame(frows), use_container_width=True, hide_index=True)


# ---- Tab: Compare pairs ----
with tab_compare:
    st.header("🔀 Compare — everything side by side")

    # ---------- 🔹 Signal level ----------
    st.markdown("### 🔹 Signal level")
    st.caption("Every individual signal, one row each.")

    _, _mkt_t, _ = market_analysis()
    _pmap_cmp = dict(zip(_mkt_t["Signal"], _mkt_t["Partial r (market held constant)"]))
    rows = []
    for name, p in PAIRS.items():
        df_p = load_pair_data(name)
        lags_p = compute_lag(name, p.max_lag)
        lag, r, n = peak_correlation(lags_p)
        passed_p = screen_results[name].overall_pass
        # cycle test: does its edge survive removing the commodity cycle?
        _sp = _pmap_cmp.get(p.leading.name, np.nan)
        cycle_ok_p = True if p.frequency != "monthly" else ((not np.isnan(_sp)) and abs(_sp) >= 0.20)
        verdict = (
            "✅ Confirmed"
            if (not np.isnan(r) and abs(r) >= 0.5 and lag > 0 and passed_p and cycle_ok_p)
            else "🌀 Strong but cycle-driven"
            if (not np.isnan(r) and abs(r) >= 0.5 and lag > 0 and passed_p and not cycle_ok_p)
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
        gate = {c.name: CHECK_MARK[c.passed] for c in screen_results[name].checks}
        rows.append({
            "Signal": p.leading.name,
            "Verdict": verdict,
            "Lead (mo)": f"{lag:+d}",
            "Raw r": f"{r:+.2f}" if not np.isnan(r) else "—",
            "Partial r": f"{_sp:+.2f}" if not np.isnan(_sp) else "—",
            "Leads?": "yes" if (not np.isnan(r) and lag >= 0) else "NO",
            "Smoothing": gate.get("Survives smoothing", "➖"),
            "Holds over time": gate.get("Holds over time", "➖"),
            "Real lead": gate.get("Real lead, not a co-mover", "➖"),
            "N": len(df_p),
        })

    # well-ordered: identity → verdict → the measured numbers → the screen gates → sample
    col_order = ["Signal", "Verdict", "Lead (mo)", "Raw r", "Partial r", "Leads?",
                 "Smoothing", "Holds over time", "Real lead", "N"]
    summary_df = pd.DataFrame(rows)[col_order]
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

    st.divider()

    # ---------- 🔸 Force level (tight) ----------
    st.markdown("### 🔸 Force level — tight forces (cohesive)")
    st.caption(
        "Correlated signals collapsed into tight ≥0.70 clusters, one witness each. Cohesive and "
        "granular, but adjacent forces can still be cousins (0.5–0.7), so this is not yet a count of "
        "independent bets."
    )
    ft_tight, fd_tight = force_level_analysis(tight=True)
    n_pass_t = render_force_leaderboard(ft_tight)
    mean_t, cous_t = force_cross_corr(fd_tight)
    st.caption(f"{len(ft_tight)} forces · {n_pass_t} pass · mean inter-force |r| = "
               f"{mean_t:.2f} · {cous_t} cousin pair(s) at 0.5–0.7.")

    st.divider()

    # ---------- 🔶 Independent blocs (loose) ----------
    st.markdown("### 🔶 Independent blocs — loose grouping (max independence)")
    st.caption(
        "The original *looser* (average-linkage) grouping: it MERGES cousins into bigger blocs, "
        "so the blocs correlate less with each other and come closest to a true count of "
        "**independent bets**. The trade-off: each bloc is less internally cohesive than a tight force. "
        "Opposite-sign grab-bags are still split out."
    )
    ft_loose, fd_loose = force_level_analysis(tight=False)
    n_pass_l = render_force_leaderboard(ft_loose)
    mean_l, cous_l = force_cross_corr(fd_loose)
    st.caption(f"{len(ft_loose)} blocs · {n_pass_l} pass · mean inter-bloc |r| = "
               f"{mean_l:.2f} · {cous_l} cousin pair(s) at 0.5–0.7.")
    if not np.isnan(mean_t) and not np.isnan(mean_l):
        st.markdown(
            f"➡️ Merging cousins drops the average cross-correlation from **{mean_t:.2f}** (tight forces) "
            f"to **{mean_l:.2f}** (loose blocs) and cousin pairs from **{cous_t}** to **{cous_l}** — "
            "fewer, more independent groupings."
        )


# ---- Groups & Market: dissolved into the Correlation tab (force layer) ----
if False:  # noqa — retained temporarily; content now lives in tab_corr
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


# ---- Force level: dissolved into Compare (leaderboards) + Screen (detail) ----
if False:  # noqa — retained temporarily; content now lives in tab_compare & tab_screen
    st.header("🏭 Force level — independent forces, de-duplicated")
    st.caption(
        "At the signal level, five copper series look like five pieces of evidence. "
        "They are really one. Here, correlated signals (≥ 0.70) are collapsed into a single "
        "**force** and scored once, so the count reflects how many *independent* things "
        "actually lead transformer prices."
    )

    with st.expander("How a force is graded (three tiers)", expanded=False):
        st.markdown(
            "Every cluster is sorted by how cohesive it is:\n\n"
            "- 🟢 **TIGHT** — every member correlates ≥ 0.70 with every other. A real, "
            "cohesive force. Trust the composite.\n"
            "- 🟡 **LOOSE / SAME-SIGN** — held together by a chain (the far ends are weakly "
            "related), but every member points the *same* way toward transformer prices. "
            "Treating it as one force merely *under-counts* witnesses — a conservative, safe error.\n"
            "- 🔴 **LOOSE / OPPOSITE-SIGN** — loose *and* members disagree on direction (a "
            "positive lead averaged with a negative relationship). The composite is meaningless, "
            "so it is **excluded** from the leaderboard by construction.\n\n"
            "A usable force then goes through the *same* gates an individual signal does: it must "
            "lead, be strong enough, survive smoothing, hold over time, and be a real lead (not a "
            "co-mover). Finally it is market-controlled (partial r vs the broad commodity cycle)."
        )

    force_table, force_details = force_level_analysis()

    # --- Force-level leaderboard (usable forces only, opposite-sign excluded) ---
    st.subheader("Force-level leaderboard")
    st.caption(
        "Built on usable forces. 🔴 opposite-sign grab-bags are never scored as a coherent force — "
        "instead they are split (complete linkage) into their tight pieces, shown with a ↳ and "
        "Origin = split, and each piece is re-tested on its own."
    )

    disp = force_table.copy()
    disp["Tier"] = disp["Tier"].map(lambda t: f"{TIER_ICON[t]} {t}")
    disp["Screen"] = disp["Screen"].map(
        lambda s: {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "CYCLE-DRIVEN": "🌀 CYCLE-DRIVEN", "EXCLUDED": "🔴 EXCLUDED"}.get(s, s)
    )
    # passing first, then cycle-driven, then failed, then excluded; strongest partial r on top
    disp = disp.assign(_rank=force_table["Screen"].map({"PASS": 0, "CYCLE-DRIVEN": 1, "FAIL": 2, "EXCLUDED": 3})) \
               .sort_values(["_rank", "Partial r"], key=lambda c: c if c.name == "_rank" else c.abs(),
                            ascending=[True, False]) \
               .drop(columns="_rank")
    st.dataframe(
        disp[["Force", "Tier", "Size", "Origin", "Lead (mo)", "Raw r", "Partial r", "Screen", "Members"]],
        use_container_width=True, hide_index=True,
    )

    n_pass = int((force_table["Screen"] == "PASS").sum())
    n_split = int((force_table["Origin"] == "split").sum())
    st.markdown(
        f"**{n_pass}** forces pass every check. "
        + (f"**{n_split}** force(s) shown are tight pieces recovered by splitting an "
           "opposite-sign grab-bag (auto re-clustered, then re-tested)."
           if n_split else "No opposite-sign grab-bags in this set — nothing needed splitting.")
    )

    st.divider()

    # --- Force detail (mirror of the signal-detail view) ---
    st.subheader("Force detail")
    pick = st.selectbox("Pick a force", list(force_details.keys()))
    fd = force_details[pick]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tier", f"{TIER_ICON[fd['tier']]} {fd['tier'].split(' / ')[0]}")
    m2.metric("Measured lead", f"{fd['lag']:+d} mo")
    m3.metric("Raw r", f"{fd['raw']:+.2f}")
    m4.metric("Partial r (cycle held)", f"{fd['par']:+.2f}")

    if fd["screen"] == "PASS":
        st.success("✅ **Force PASSES** — leads, strong, clears every gate, and keeps its edge after the commodity cycle is removed.")
    elif fd["screen"] == "CYCLE-DRIVEN":
        st.warning("🌀 **STRONG BUT CYCLE-DRIVEN** — passes the gates but its edge vanishes once the commodity cycle is removed (partial |r| < 0.20). Not transformer-specific.")
    elif fd["screen"] == "EXCLUDED":
        st.error("🔴 **EXCLUDED — opposite-sign grab-bag.** Members disagree on direction, so the composite is not a coherent force.")
    else:
        st.warning("❌ **Force FLAGGED** — see the failing gate(s) below.")

    if len(fd["members"]) > 1:
        st.caption(
            f"Force of **{len(fd['members'])}** signals · weakest internal link "
            f"r={fd['tightness']:+.2f} · members: {', '.join(fd['members'])}"
        )
    else:
        st.caption("Single-signal force (stands on its own).")

    # gates on the composite
    st.markdown("**Gates on the composite:**")
    for c in fd["checks"]:
        st.markdown(f"{CHECK_MARK[c.passed]} **{c.name}** — {c.detail}")
    surv = cycle_label(fd["raw"], fd["par"])
    st.markdown(f"📉 **Beats the commodity cycle?** raw r = {fd['raw']:+.2f} → partial r = {fd['par']:+.2f} → **{surv}**")
    st.caption(
        f"'Cycle-driven' only when the correlation drops by ≥ {CYCLE_MIN_DROP:.2f} after the cycle "
        "is removed. Raw ≈ partial means it was never cycle-driven, just at its own level."
    )

    # composite vs transformer chart (YoY)
    comp_df = pd.concat([fd["comp"].rename("Force composite (YoY %)"),
                         fd["outcome"].rename("Transformer PPI (YoY %)")], axis=1).dropna()
    figf = go.Figure()
    figf.add_trace(go.Scatter(x=comp_df.index, y=comp_df.iloc[:, 0],
                              name="Force composite (YoY %)", line=dict(color=LEADING_COLOR, width=1.8)))
    figf.add_trace(go.Scatter(x=comp_df.index, y=comp_df.iloc[:, 1],
                              name="Transformer PPI (YoY %)", line=dict(color=OUTCOME_COLOR, width=1.8)))
    figf.add_hline(y=0, line_dash="dash", line_color="gray")
    figf.update_layout(
        height=420, hovermode="x unified",
        title=f"{pick} composite vs transformer PPI (measured lead {fd['lag']:+d} mo)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=70),
    )
    st.plotly_chart(figf, use_container_width=True)


# ---- Tab: Lead-lag map ----
with tab_map:
    st.header("🗺️ Lead-lag map — which items lead which OTHERS")
    st.caption(
        "Everything else measures each item vs transformer prices. This maps item → *other item*, "
        "so the propagation **chain** shows up (e.g. steel scrap → steel → transformer). "
        "An arrow A → B means A's move shows up first and B follows. 🟢 sources lead but aren't led · "
        "🟠 relays pass it along · 🔴 sinks are led but lead nothing. Transformer PPI should sit at the "
        "far right as a pure sink — it's led, it doesn't lead."
    )
    st.caption("Edges shown only when the lead is genuine: positive lag, |r| ≥ 0.50, and a real "
               "lead-sharpness gain over lag 0 (not a co-mover).")

    OUTCOME_NAME = "transformer_ppi"

    def _yoy(s):
        return s.pct_change(12) * 100

    def _render_map(series, layer_label):
        edges = lead_lag_edges(series)
        if not edges:
            st.info("No genuine lead relationships clear the bar at this layer.")
            return
        st.plotly_chart(leadlag_figure(series, edges, OUTCOME_NAME), use_container_width=True)

        into = sorted([e for e in edges if e["follower"] == OUTCOME_NAME], key=lambda e: -abs(e["r"]))
        if into:
            st.markdown("**The final step — what leads transformer prices:**")
            st.markdown("  ·  ".join(f"{e['leader'].replace('_',' ')} (**{e['lead']}mo**, r={e['r']:+.2f})"
                                     for e in into))
        leads_out = [e for e in edges if e["leader"] == OUTCOME_NAME]
        st.caption(f"Does transformer lead anything? {'nothing — it is a pure sink ✅' if not leads_out else '⚠️ ' + str(len(leads_out)) + ' edges'}")

        with st.expander(f"All {len(edges)} lead edges ({layer_label})"):
            ed = pd.DataFrame(edges).sort_values("r", key=lambda s: s.abs(), ascending=False)
            ed["leader"] = ed["leader"].str.replace("_", " ")
            ed["follower"] = ed["follower"].str.replace("_", " ")
            st.dataframe(ed.rename(columns={"leader": "Leader", "follower": "Follower",
                                            "lead": "Lead (mo)", "r": "r", "gain": "Lead sharpness"}),
                         use_container_width=True, hide_index=True)

    # 🔹 Signal layer
    st.markdown("### 🔹 Signal level")
    st.caption("Propagation among individual signals.")
    sig_series = {p.leading.name: _yoy(p.leading.load())
                  for p in PAIRS.values() if p.frequency == "monthly"}
    sig_series[OUTCOME_NAME] = _yoy(load_fred_monthly(OUTCOME_NAME))
    _render_map(sig_series, "signal level")

    st.divider()

    # 🔸 Force layer
    st.markdown("### 🔸 Force level")
    st.caption("Propagation among forces (composites) — one witness each.")
    _ftable, _fdetails = force_level_analysis()
    force_series = {}
    for fname, fd in _fdetails.items():
        force_series[fname] = fd["comp"]
    # add transformer as the outcome node
    any_fd = next(iter(_fdetails.values())) if _fdetails else None
    if any_fd is not None:
        force_series[OUTCOME_NAME] = any_fd["outcome"]
    _render_map(force_series, "force level")


# =================================================================
# FOOTER
# =================================================================
st.divider()
st.caption(
    f"signal_tool v0 | Generated reports available in `signal_tool/output/`. "
    f"To add a new signal pair, edit `pairs.py`. To run the static reports, "
    f"`python run.py`."
)
