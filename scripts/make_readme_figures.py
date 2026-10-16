"""Reproduce the README results on real market data (Yahoo Finance).

    python scripts/make_readme_figures.py

Equal-weight portfolio of AAPL, MSFT, GOOGL, JPM, GS, BRK-B, rebalanced daily,
2019-01-01 to 2024-12-31. Writes assets/var_backtest.png and
assets/sample_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.portfolio.metrics import PortfolioAnalyzer  # noqa: E402
from src.risk.backtest import backtest  # noqa: E402
from src.risk.var import VaREngine  # noqa: E402

TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "GS", "BRK-B"]
START, END = "2019-01-01", "2024-12-31"


def portfolio_log_returns() -> pd.Series:
    prices = yf.download(TICKERS, start=START, end=END, progress=False, auto_adjust=True)["Close"]
    simple = prices.pct_change().dropna()
    port_simple = simple.mean(axis=1)            # equal weight, daily rebalancing
    return np.log1p(port_simple).rename("portfolio")


def main() -> dict:
    r = portfolio_log_returns()
    m = PortfolioAnalyzer(r, risk_free_rate=0.0)
    var = VaREngine(r)
    np.random.seed(42)
    summary = {
        "period": f"{r.index[0].date()} to {r.index[-1].date()}",
        "days": int(len(r)),
        "cagr": round(float(m.cagr()), 4),
        "ann_volatility": round(float(m.ann_volatility()), 4),
        "sharpe_rf0": round(float(m.sharpe_ratio()), 2),
        "max_drawdown": round(float(m.max_drawdown()), 4),
        "excess_kurtosis": round(float(m.excess_kurtosis()), 2),
        "var_full_sample": {
            f"{res.method} {int(res.confidence * 100)}%": {"var": round(float(res.var), 4), "cvar": round(float(res.cvar), 4)}
            for c in (0.95, 0.99) for res in var.all_methods(c)
        },
        "backtest_99": {},
    }
    details = {}
    for method in ("historical", "normal", "student_t", "ewma", "fhs"):
        res, det = backtest(r, 0.99, method, window=250)
        details[method] = det
        summary["backtest_99"][method] = {
            "days_tested": res.n_obs, "exceptions": res.n_exceptions,
            "expected": round(res.n_obs * 0.01, 1), "per_250_days": round(res.exceptions_per_250, 1),
            "kupiec_p": round(res.p_value, 3), "basel_zone": res.zone,
        }

    fig, ax = plt.subplots(figsize=(11, 4.4))
    det_n, det_h = details["normal"], details["fhs"]
    ax.plot(det_n.index, det_n["return"] * 100, lw=0.5, color="#8c959f", label="Daily portfolio return")
    ax.plot(det_n.index, det_n["var"] * 100, lw=1.2, color="#cf222e", label="99 % VaR, normal")
    ax.plot(det_h.index, det_h["var"] * 100, lw=1.2, color="#1f6feb", label="99 % VaR, filtered historical simulation")
    exc = det_n[det_n["exception"]]
    ax.scatter(exc.index, exc["return"] * 100, s=14, color="#cf222e", zorder=3,
               label=f"Normal VaR breached ({len(exc)} days)")
    ax.set_ylabel("Daily return (%)")
    ax.set_title("Rolling 250-day VaR forecasts vs realised returns (out of sample)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (ROOT / "assets").mkdir(exist_ok=True)
    fig.savefig(ROOT / "assets" / "var_backtest.png", dpi=150)
    (ROOT / "assets" / "sample_results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
