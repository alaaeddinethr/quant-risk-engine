"""
stress_test.py
--------------
Historical scenario stress testing — replays market crises on the
current portfolio composition.

For each scenario (GFC, COVID, 2022 Rate Shock, Dot-com):
  1. Download the actual market returns during that period
  2. Apply portfolio weights to compute hypothetical P&L
  3. Report drawdown, VaR breach, recovery time

Also includes a custom scenario builder (user-defined shocks per asset).

Reference:
    Basel Committee on Banking Supervision (2009).
    Principles for sound stress testing practices and supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


# ─── Scenario Definitions ─────────────────────────────────────────────────────

SCENARIOS = {
    "gfc": {
        "name":        "Global Financial Crisis",
        "start":       "2008-09-01",
        "end":         "2009-03-31",
        "description": "Lehman Brothers collapse, credit market freeze",
        "benchmark_drop": -0.54,
    },
    "covid": {
        "name":        "COVID-19 Crash",
        "start":       "2020-02-19",
        "end":         "2020-03-23",
        "description": "Fastest 30%+ decline in S&P 500 history (33 days)",
        "benchmark_drop": -0.34,
    },
    "rate_shock_2022": {
        "name":        "2022 Rate Shock",
        "start":       "2022-01-03",
        "end":         "2022-10-13",
        "description": "Fed's fastest rate-hiking cycle since 1980",
        "benchmark_drop": -0.25,
    },
    "dotcom": {
        "name":        "Dot-com Bubble Burst",
        "start":       "2000-03-10",
        "end":         "2002-10-09",
        "description": "NASDAQ lost ~78% of its value over 2.5 years",
        "benchmark_drop": -0.49,
    },
}


@dataclass
class StressResult:
    """Results of a single stress scenario."""
    scenario_key: str
    scenario_name: str
    period_start: str
    period_end: str
    portfolio_return: float    # cumulative return over scenario
    max_drawdown: float        # worst peak-to-trough within scenario
    daily_returns: pd.Series
    available_tickers: list[str]
    missing_tickers: list[str]

    @property
    def loss_1m(self) -> float:
        """Initial portfolio value $1M → terminal value."""
        return 1_000_000 * (1 + self.portfolio_return)

    def __str__(self) -> str:
        return (
            f"[{self.scenario_name}] "
            f"Return: {self.portfolio_return*100:.2f}% | "
            f"Max Drawdown: {self.max_drawdown*100:.2f}%"
        )


class StressTester:
    """
    Runs historical stress tests on a portfolio.

    Args:
        tickers: List of ticker symbols
        weights: Portfolio weights (None = equal weight)
    """

    def __init__(
        self,
        tickers: list[str],
        weights: np.ndarray | None = None,
    ):
        self.tickers = [t.upper() for t in tickers]
        n = len(self.tickers)

        if weights is None:
            self.weights = np.ones(n) / n
        else:
            w = np.asarray(weights, dtype=float)
            self.weights = w / w.sum()

        self._weight_map = dict(zip(self.tickers, self.weights))

    def _download_scenario(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Download prices for a historical scenario window."""
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw[["Close"]] if len(tickers) == 1 else raw
            return prices.ffill().dropna()
        except Exception as e:
            print(f"  ⚠ Download failed for scenario {start}→{end}: {e}")
            return pd.DataFrame()

    def _max_drawdown(self, cum_returns: pd.Series) -> float:
        """Compute maximum drawdown from cumulative return series."""
        wealth = 1 + cum_returns
        rolling_max = wealth.cummax()
        drawdowns = (wealth - rolling_max) / rolling_max
        return drawdowns.min()

    def run_scenario(self, scenario_key: str) -> StressResult | None:
        """Run a single named scenario."""
        if scenario_key not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_key}. Choose from {list(SCENARIOS.keys())}")

        cfg = SCENARIOS[scenario_key]
        print(f"\n  📉 {cfg['name']} ({cfg['start']} → {cfg['end']})")

        prices = self._download_scenario(self.tickers, cfg["start"], cfg["end"])

        if prices.empty:
            print(f"     ⚠ No data available")
            return None

        # Some tickers may not exist in that period
        available = [t for t in self.tickers if t in prices.columns]
        missing   = [t for t in self.tickers if t not in prices.columns]

        if missing:
            print(f"     ⚠ Missing tickers (not listed then): {missing}")

        if not available:
            return None

        # Rebalance weights for available tickers
        avail_weights = np.array([self._weight_map[t] for t in available])
        avail_weights /= avail_weights.sum()

        # Daily log returns
        log_returns = np.log(prices[available] / prices[available].shift(1)).dropna()

        # Portfolio returns
        port_daily = (log_returns * avail_weights).sum(axis=1)

        # Cumulative return
        port_cum = port_daily.cumsum()
        total_return = np.exp(port_cum.iloc[-1]) - 1

        max_dd = self._max_drawdown(port_cum)

        result = StressResult(
            scenario_key=scenario_key,
            scenario_name=cfg["name"],
            period_start=cfg["start"],
            period_end=cfg["end"],
            portfolio_return=total_return,
            max_drawdown=max_dd,
            daily_returns=port_daily,
            available_tickers=available,
            missing_tickers=missing,
        )
        print(f"     Return: {total_return*100:.2f}% | Max Drawdown: {max_dd*100:.2f}%")
        return result

    def run_all(self, scenarios: list[str] | None = None) -> dict[str, StressResult]:
        """
        Run all (or selected) stress scenarios.

        Returns:
            Dict mapping scenario_key → StressResult
        """
        keys = scenarios or list(SCENARIOS.keys())
        results = {}
        print(f"\n🔥 Running {len(keys)} stress scenarios on portfolio: {self.tickers}")

        for key in keys:
            r = self.run_scenario(key)
            if r is not None:
                results[key] = r

        return results

    def run_custom(
        self,
        name: str,
        shocks: dict[str, float],
    ) -> StressResult:
        """
        Custom scenario: apply user-defined shocks per asset.

        Args:
            name: Scenario name
            shocks: Dict {ticker: shock_return}, e.g. {"AAPL": -0.15, "JPM": -0.30}
                    Assets not specified get 0% shock.

        Returns:
            StressResult with a single-period shock
        """
        available = []
        port_return = 0.0

        for ticker, weight in self._weight_map.items():
            shock = shocks.get(ticker, 0.0)
            port_return += weight * shock
            available.append(ticker)

        # Synthetic daily return series (single day shock)
        daily = pd.Series([port_return], index=[pd.Timestamp.today()], name="custom")

        return StressResult(
            scenario_key="custom",
            scenario_name=name,
            period_start=str(pd.Timestamp.today().date()),
            period_end=str(pd.Timestamp.today().date()),
            portfolio_return=port_return,
            max_drawdown=port_return,
            daily_returns=daily,
            available_tickers=available,
            missing_tickers=[],
        )

    def report(self, results: dict[str, StressResult]) -> pd.DataFrame:
        """Generate summary DataFrame from stress test results."""
        rows = []
        for key, r in results.items():
            cfg = SCENARIOS.get(key, {})
            rows.append({
                "Scenario":          r.scenario_name,
                "Period":            f"{r.period_start} → {r.period_end}",
                "Portfolio Return":  f"{r.portfolio_return*100:.2f}%",
                "Max Drawdown":      f"{r.max_drawdown*100:.2f}%",
                "P&L ($1M)":         f"${r.loss_1m - 1_000_000:+,.0f}",
                "Benchmark Drop":    f"{cfg.get('benchmark_drop', 0)*100:.0f}%",
            })

        df = pd.DataFrame(rows)
        print(f"\n{'═'*70}")
        print("  STRESS TEST REPORT")
        print(f"{'═'*70}")
        print(df.to_string(index=False))
        return df


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "JPM", "GS"]
    tester = StressTester(tickers=tickers)
    results = tester.run_all(scenarios=["covid", "rate_shock_2022"])
    tester.report(results)
