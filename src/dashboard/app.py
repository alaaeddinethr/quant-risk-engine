"""
app.py
------
Interactive Streamlit dashboard for the Quantitative Risk Engine.

Sections:
  1. Portfolio Builder — ticker input, weights, data load
  2. Performance Overview — return, vol, Sharpe, drawdown
  3. Risk Analysis — VaR/CVaR across 4 methods, comparison chart
  4. Monte Carlo — 10k path simulation with percentile bands
  5. Efficient Frontier — optimal portfolio visualization
  6. Stress Tests — historical crisis scenario results

Run:
    streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.market_data import PriceLoader
from src.portfolio.metrics import PortfolioAnalyzer, asset_contribution
from src.portfolio.optimizer import EfficientFrontier
from src.risk.monte_carlo import MonteCarloEngine
from src.risk.var import PortfolioVaR

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quantitative Risk Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=JetBrains+Mono:wght@300;400;500&display=swap');
html,[class*="css"]{font-family:'Libre Baskerville',serif}
.stApp{background:#faf8f4}
h1,h2,h3{font-family:'Libre Baskerville',serif;color:#1a1208}
.metric-card{background:#ffffff;border:1px solid #e8e0d0;border-radius:4px;padding:1.2rem;text-align:center}
.metric-label{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#8a7a65;text-transform:uppercase;letter-spacing:.1em}
.metric-value{font-size:1.8rem;font-weight:700;color:#1a1208;font-family:'Libre Baskerville',serif}
.metric-sub{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:#8a7a65;margin-top:.2rem}
.positive{color:#1a6b3a}
.negative{color:#8b1a1a}
.section-rule{border:none;border-top:1px solid #e8e0d0;margin:2rem 0}
</style>
""", unsafe_allow_html=True)

# ─── Plotly theme ─────────────────────────────────────────────────────────────

THEME = dict(
    paper_bgcolor="#faf8f4",
    plot_bgcolor="#faf8f4",
    font=dict(color="#1a1208", family="JetBrains Mono"),
    xaxis=dict(gridcolor="#ede8df", linecolor="#ccc4b4", showgrid=True),
    yaxis=dict(gridcolor="#ede8df", linecolor="#ccc4b4", showgrid=True),
)
COLORS = ["#2d5a27", "#8b1a1a", "#1a3f6b", "#7a4f1a", "#4a1a6b", "#1a5f6b"]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def metric_card(label, value, sub="", color=""):
    color_class = f' {color}' if color else ''
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value{color_class}">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(tickers: tuple, years: int):
    loader = PriceLoader(list(tickers), years=years)
    returns = loader.get_returns()
    prices  = loader.get_prices()
    return returns, prices

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Configuration")
    st.divider()

    ticker_input = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL,MSFT,GOOGL,JPM,GS",
    )
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    years = st.slider("Historical data (years)", 2, 10, 5)

    equal_weight = st.checkbox("Equal weight", value=True)
    weights_dict = {}
    if not equal_weight:
        st.markdown("**Custom weights (%)**")
        for t in tickers:
            weights_dict[t] = st.slider(t, 0, 100, int(100 / len(tickers)), key=f"w_{t}")

    rf_rate = st.slider("Risk-free rate (%)", 0.0, 8.0, 4.5, 0.1) / 100
    n_sim = st.select_slider("MC simulations", options=[1000, 5000, 10000], value=5000)
    conf = st.selectbox("VaR confidence", ["95%", "99%"], index=0)
    conf_val = 0.95 if conf == "95%" else 0.99

    st.divider()
    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)

# ─── Main ─────────────────────────────────────────────────────────────────────

st.markdown("# 📊 Quantitative Risk Engine")
st.markdown("*Portfolio analytics — VaR, CVaR, Monte Carlo, Markowitz optimization*")
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

if not run_btn:
    st.info("Configure your portfolio in the sidebar and click **Run Analysis**.")
    st.stop()

# Load data
with st.spinner("Fetching market data..."):
    try:
        returns_df, prices_df = load_data(tuple(tickers), years)
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

# Filter to available tickers
available = [t for t in tickers if t in returns_df.columns]
if len(available) < 2:
    st.error("Need at least 2 tickers with data.")
    st.stop()

returns_df = returns_df[available]

# Weights
if equal_weight:
    weights = np.ones(len(available)) / len(available)
else:
    raw = np.array([weights_dict.get(t, 1) for t in available], dtype=float)
    weights = raw / raw.sum()

port_returns = (returns_df * weights).sum(axis=1)

# ─── 1. Performance Overview ──────────────────────────────────────────────────

st.markdown("## Performance Overview")
analyzer = PortfolioAnalyzer(port_returns, risk_free_rate=rf_rate)
m = analyzer.compute_all()

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: metric_card("Ann. Return", f"{m.ann_return*100:.1f}%", f"CAGR {m.cagr*100:.1f}%", "positive" if m.ann_return >= 0 else "negative")
with c2: metric_card("Ann. Volatility", f"{m.ann_volatility*100:.1f}%", f"Daily {m.ann_volatility/16:.2f}%")
with c3: metric_card("Sharpe Ratio", f"{m.sharpe_ratio:.2f}", f"rf = {rf_rate*100:.1f}%", "positive" if m.sharpe_ratio >= 1 else "")
with c4: metric_card("Sortino Ratio", f"{m.sortino_ratio:.2f}", "downside vol")
with c5: metric_card("Max Drawdown", f"{m.max_drawdown*100:.1f}%", f"{m.max_drawdown_duration}d duration", "negative")
with c6: metric_card("Calmar Ratio", f"{m.calmar_ratio:.2f}", "CAGR / |MDD|")

st.divider()

# Cumulative return chart
col_left, col_right = st.columns([2, 1])
with col_left:
    cum = np.exp(port_returns.cumsum()) - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=cum * 100, mode="lines",
        line=dict(color="#2d5a27", width=1.8), fill="tozeroy",
        fillcolor="rgba(45,90,39,0.08)", name="Portfolio"))
    fig.update_layout(title="Cumulative Return (%)", height=300,
        yaxis_title="Return (%)", **THEME)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    dd = analyzer.drawdown_series() * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines",
        line=dict(color="#8b1a1a", width=1.2), fill="tozeroy",
        fillcolor="rgba(139,26,26,0.1)", name="Drawdown"))
    fig2.update_layout(title="Drawdown (%)", height=300,
        yaxis_title="Drawdown (%)", **THEME)
    st.plotly_chart(fig2, use_container_width=True)

# ─── 2. Risk Analysis ─────────────────────────────────────────────────────────

st.markdown("## Risk Analysis — VaR / CVaR")

pvar = PortfolioVaR(returns_df, weights)
report_df = pvar.full_report(confidence_levels=[0.95, 0.99], n_simulations=n_sim)

# Filter to selected confidence
disp = report_df[report_df["Confidence"] == conf].copy()

c1, c2, c3 = st.columns(3)
methods = disp["Method"].tolist()
with c1:
    row = disp[disp["Method"] == "Historical"].iloc[0] if "Historical" in methods else None
    if row is not None:
        metric_card("Historical VaR", f"{row['VaR (%)']:.3f}%", f"CVaR {row['CVaR (%)']:.3f}%", "negative")
with c2:
    row = disp[disp["Method"] == "Parametric (Student-t)"].iloc[0] if "Parametric (Student-t)" in methods else None
    if row is not None:
        metric_card("Parametric VaR (t)", f"{row['VaR (%)']:.3f}%", f"CVaR {row['CVaR (%)']:.3f}%", "negative")
with c3:
    row = disp[disp["Method"] == "Monte Carlo (Correlated)"].iloc[0] if "Monte Carlo (Correlated)" in methods else None
    if row is not None:
        metric_card("Monte Carlo VaR", f"{row['VaR (%)']:.3f}%", f"CVaR {row['CVaR (%)']:.3f}%", "negative")

# VaR comparison bar chart
fig = go.Figure()
colors_var  = ["#2d5a27","#1a3f6b","#7a4f1a","#4a1a6b","#8b1a1a"]
colors_cvar = ["rgba(45,90,39,.4)","rgba(26,63,107,.4)","rgba(122,79,26,.4)","rgba(74,26,107,.4)","rgba(139,26,26,.4)"]

for i, (_, row) in enumerate(disp.iterrows()):
    fig.add_trace(go.Bar(
        x=[row["Method"]], y=[abs(row["VaR (%)"])],
        name=f"VaR {row['Method']}", marker_color=colors_var[i % len(colors_var)],
        legendgroup=row["Method"],
    ))
    fig.add_trace(go.Bar(
        x=[row["Method"]], y=[abs(row["CVaR (%)"])],
        name=f"CVaR {row['Method']}", marker_color=colors_cvar[i % len(colors_cvar)],
        legendgroup=row["Method"],
    ))

fig.update_layout(
    title=f"VaR vs CVaR by Method ({conf})", barmode="group",
    yaxis_title="Loss (%)", height=340, **THEME,
)
st.plotly_chart(fig, use_container_width=True)

# Return distribution
fig = go.Figure()
fig.add_trace(go.Histogram(x=port_returns*100, nbinsx=80,
    marker_color="#2d5a27", opacity=0.75, name="Daily Returns"))
var_val = analyzer.var(conf_val) * 100
cvar_val = analyzer.cvar(conf_val) * 100
fig.add_vline(x=var_val, line=dict(color="#8b1a1a", dash="dash", width=2),
    annotation_text=f"VaR {conf}", annotation_font=dict(color="#8b1a1a", size=11))
fig.add_vline(x=cvar_val, line=dict(color="#7a4f1a", dash="dot", width=1.5),
    annotation_text=f"CVaR {conf}", annotation_font=dict(color="#7a4f1a", size=11))
fig.update_layout(title="Daily Return Distribution", xaxis_title="Return (%)",
    height=300, **THEME)
st.plotly_chart(fig, use_container_width=True)

# ─── 3. Monte Carlo Simulation ────────────────────────────────────────────────

st.markdown("## Monte Carlo Simulation")
with st.spinner(f"Running {n_sim:,} simulations..."):
    mc_engine = MonteCarloEngine(returns_df, weights, initial_value=1_000_000)
    horizon = st.select_slider("Horizon", options=[21, 63, 126, 252], value=252,
        format_func=lambda x: {21:"1M",63:"3M",126:"6M",252:"1Y"}[x])
    result = mc_engine.run(horizon=horizon, n_simulations=n_sim)

bands = result.percentile_bands()
days = np.arange(horizon + 1)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.concatenate([days, days[::-1]]),
    y=np.concatenate([bands["p95"], bands["p5"][::-1]]),
    fill="toself", fillcolor="rgba(45,90,39,0.08)",
    line=dict(color="rgba(0,0,0,0)"), name="5th–95th pct", showlegend=True))
fig.add_trace(go.Scatter(
    x=np.concatenate([days, days[::-1]]),
    y=np.concatenate([bands["p75"], bands["p25"][::-1]]),
    fill="toself", fillcolor="rgba(45,90,39,0.18)",
    line=dict(color="rgba(0,0,0,0)"), name="25th–75th pct"))
fig.add_trace(go.Scatter(x=days, y=bands["median"], mode="lines",
    line=dict(color="#2d5a27", width=2), name="Median"))
fig.add_trace(go.Scatter(x=days, y=bands["mean"], mode="lines",
    line=dict(color="#1a3f6b", width=1.5, dash="dash"), name="Mean"))
fig.add_hline(y=1_000_000, line=dict(color="#8b1a1a", dash="dot", width=1))

fig.update_layout(title=f"Portfolio Simulation — {n_sim:,} paths × {horizon} days",
    xaxis_title="Trading Days", yaxis_title="Portfolio Value ($)",
    height=380, **THEME)
st.plotly_chart(fig, use_container_width=True)

s = result.summary()
c1, c2, c3, c4 = st.columns(4)
with c1: metric_card("Median Terminal", f"${s['median_terminal']:,.0f}", f"{(s['median_terminal']/1e6-1)*100:+.1f}%")
with c2: metric_card("VaR 95% (horizon)", f"{s['var_95']*100:.2f}%", "terminal return")
with c3: metric_card("P(loss > 10%)", f"{s['prob_loss_10pct']*100:.1f}%", "probability")
with c4: metric_card("P(loss > 20%)", f"{s['prob_loss_20pct']*100:.1f}%", "probability")

# ─── 4. Efficient Frontier ────────────────────────────────────────────────────

st.markdown("## Portfolio Optimization — Efficient Frontier")
with st.spinner("Computing efficient frontier..."):
    ef = EfficientFrontier(returns_df, risk_free_rate=rf_rate)
    frontier = ef.compute_frontier()
    mvp = ef.min_variance()
    msr = ef.max_sharpe()
    rp  = ef.risk_parity()

fig = go.Figure()
# Frontier
fig.add_trace(go.Scatter(
    x=frontier["ann_volatility"]*100, y=frontier["ann_return"]*100,
    mode="lines", line=dict(color="#ccc4b4", width=2), name="Efficient Frontier"))
# Colour by Sharpe
fig.add_trace(go.Scatter(
    x=frontier["ann_volatility"]*100, y=frontier["ann_return"]*100,
    mode="markers",
    marker=dict(color=frontier["sharpe_ratio"], colorscale="RdYlGn",
        size=3, colorbar=dict(title="Sharpe"), showscale=True),
    name="Sharpe ratio", showlegend=False))

for port, color, sym, label in [
    (mvp, "#1a3f6b", "diamond", "Min Variance"),
    (msr, "#2d5a27", "star",    "Max Sharpe"),
    (rp,  "#7a4f1a", "circle",  "Risk Parity"),
]:
    fig.add_trace(go.Scatter(
        x=[port.ann_volatility*100], y=[port.ann_return*100],
        mode="markers+text",
        marker=dict(color=color, size=14, symbol=sym,
            line=dict(color="white", width=1.5)),
        text=[label], textposition="top center",
        textfont=dict(size=10, color=color),
        name=label))

# Current portfolio
curr_ret = analyzer.ann_return()
curr_vol = analyzer.ann_volatility()
fig.add_trace(go.Scatter(
    x=[curr_vol*100], y=[curr_ret*100],
    mode="markers+text",
    marker=dict(color="#8b1a1a", size=12, symbol="x",
        line=dict(color="white", width=1.5)),
    text=["Current"], textposition="top center",
    textfont=dict(size=10, color="#8b1a1a"),
    name="Current Portfolio"))

fig.update_layout(
    title="Efficient Frontier (Markowitz)",
    xaxis_title="Annual Volatility (%)",
    yaxis_title="Annual Return (%)",
    height=420, **THEME,
)
st.plotly_chart(fig, use_container_width=True)

# Optimal weights comparison
weight_df = pd.DataFrame({
    "Min Variance": mvp.weights,
    "Max Sharpe":   msr.weights,
    "Risk Parity":  rp.weights,
    "Current":      weights,
}, index=available) * 100

fig = px.bar(weight_df.T, barmode="group",
    color_discrete_sequence=["#1a3f6b","#2d5a27","#7a4f1a","#8b1a1a"],
    title="Portfolio Weights Comparison (%)",
    labels={"value": "Weight (%)", "variable": "Ticker", "index": "Strategy"})
fig.update_layout(height=320, **THEME)
st.plotly_chart(fig, use_container_width=True)

# ─── 5. Asset Contribution ────────────────────────────────────────────────────

st.markdown("## Asset-Level Analytics")
contrib = asset_contribution(returns_df, weights, rf_rate)

c1, c2 = st.columns(2)
with c1:
    corr = returns_df.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="Correlation Matrix",
        color_continuous_midpoint=0)
    fig.update_layout(height=360, **THEME)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = go.Figure(go.Bar(
        x=contrib.index, y=contrib["Risk Contribution (%)"],
        marker_color=COLORS[:len(available)],
        text=contrib["Risk Contribution (%)"].round(1),
        textposition="outside"))
    fig.update_layout(title="Risk Contribution per Asset (%)",
        yaxis_title="%", height=360, **THEME)
    st.plotly_chart(fig, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<div style='text-align:center;font-family:JetBrains Mono,monospace;font-size:.75rem;color:#8a7a65'>"
    "Quantitative Risk Engine · Alaa Eddine TAHIR · Arts et Métiers ParisTech"
    "</div>",
    unsafe_allow_html=True
)
