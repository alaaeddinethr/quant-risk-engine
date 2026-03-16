"""
optimizer.py
------------
Portfolio optimization via Modern Portfolio Theory (Markowitz, 1952).

Implements:
  1. Efficient Frontier — full Pareto curve of optimal risk/return tradeoffs
  2. Minimum Variance Portfolio — lowest achievable risk
  3. Maximum Sharpe Ratio Portfolio — best risk-adjusted return (tangency portfolio)
  4. Risk Parity Portfolio — equal risk contribution per asset

All optimizations use SLSQP (Sequential Least Squares Programming) via scipy.

Reference:
    Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77–91.
    Sharpe, W. (1966). Mutual Fund Performance. Journal of Business, 39(1), 119–138.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint
from dataclasses import dataclass


# ─── Optimized Portfolio ──────────────────────────────────────────────────────

@dataclass
class OptimizedPortfolio:
    """Result of a portfolio optimization."""
    weights: np.ndarray
    tickers: list[str]
    ann_return: float
    ann_volatility: float
    sharpe_ratio: float
    label: str = ""

    @property
    def weight_dict(self) -> dict[str, float]:
        return dict(zip(self.tickers, self.weights))

    def __str__(self) -> str:
        lines = [
            f"\n  {self.label}",
            f"  {'─'*35}",
            f"  Annual Return:    {self.ann_return*100:.2f}%",
            f"  Annual Volatility:{self.ann_volatility*100:.2f}%",
            f"  Sharpe Ratio:     {self.sharpe_ratio:.3f}",
            f"  {'─'*35}",
        ]
        for ticker, w in sorted(self.weight_dict.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 30)
            lines.append(f"  {ticker:<8}: {w*100:5.1f}%  {bar}")
        return "\n".join(lines)


# ─── Efficient Frontier ───────────────────────────────────────────────────────

class EfficientFrontier:
    """
    Computes the Markowitz efficient frontier.

    Args:
        returns_df: Daily asset returns (dates × assets)
        risk_free_rate: Annual risk-free rate
        min_weight: Minimum weight per asset (0 = long-only allowed to go to 0)
        max_weight: Maximum weight per asset
        n_portfolios: Number of frontier points to compute
    """

    TRADING_DAYS = 252

    def __init__(
        self,
        returns_df: pd.DataFrame,
        risk_free_rate: float = 0.045,
        min_weight: float = 0.0,
        max_weight: float = 0.40,
        n_portfolios: int = 3000,
    ):
        self.returns_df = returns_df
        self.tickers = list(returns_df.columns)
        self.n = len(self.tickers)
        self.rf = risk_free_rate
        self.rf_daily = risk_free_rate / self.TRADING_DAYS
        self.min_w = min_weight
        self.max_w = max_weight
        self.n_portfolios = n_portfolios

        # Annualized stats
        self.mu  = returns_df.mean().values * self.TRADING_DAYS
        self.cov = returns_df.cov().values  * self.TRADING_DAYS

        # Optimization bounds & constraints
        self._bounds = Bounds(lb=self.min_w, ub=self.max_w)
        self._constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    def _port_return(self, w: np.ndarray) -> float:
        return float(w @ self.mu)

    def _port_volatility(self, w: np.ndarray) -> float:
        return float(np.sqrt(w @ self.cov @ w))

    def _neg_sharpe(self, w: np.ndarray) -> float:
        ret  = self._port_return(w)
        vol  = self._port_volatility(w)
        return -(ret - self.rf) / (vol + 1e-12)

    def _port_variance(self, w: np.ndarray) -> float:
        return float(w @ self.cov @ w)

    # ── Minimum Variance ─────────────────────────────────────────────────────

    def min_variance(self) -> OptimizedPortfolio:
        """Global minimum variance portfolio."""
        w0 = np.ones(self.n) / self.n
        result = minimize(
            self._port_variance,
            w0,
            method="SLSQP",
            bounds=self._bounds,
            constraints=self._constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if not result.success:
            print(f"  ⚠ Min variance: {result.message}")

        w = result.x
        ret = self._port_return(w)
        vol = self._port_volatility(w)
        return OptimizedPortfolio(
            weights=w, tickers=self.tickers,
            ann_return=ret, ann_volatility=vol,
            sharpe_ratio=(ret - self.rf) / vol,
            label="Minimum Variance Portfolio",
        )

    # ── Maximum Sharpe ───────────────────────────────────────────────────────

    def max_sharpe(self) -> OptimizedPortfolio:
        """
        Tangency portfolio — maximum Sharpe ratio.
        This is the optimal portfolio when combining with a risk-free asset.
        """
        w0 = np.ones(self.n) / self.n
        result = minimize(
            self._neg_sharpe,
            w0,
            method="SLSQP",
            bounds=self._bounds,
            constraints=self._constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if not result.success:
            print(f"  ⚠ Max Sharpe: {result.message}")

        w = result.x
        ret = self._port_return(w)
        vol = self._port_volatility(w)
        return OptimizedPortfolio(
            weights=w, tickers=self.tickers,
            ann_return=ret, ann_volatility=vol,
            sharpe_ratio=(ret - self.rf) / vol,
            label="Maximum Sharpe Ratio Portfolio",
        )

    # ── Risk Parity ──────────────────────────────────────────────────────────

    def risk_parity(self) -> OptimizedPortfolio:
        """
        Risk Parity — each asset contributes equally to total portfolio risk.

        Minimizes the sum of squared differences between risk contributions.
        Popular in institutional asset management (Bridgewater All Weather).
        """
        target_rc = np.ones(self.n) / self.n  # equal risk contribution

        def objective(w: np.ndarray) -> float:
            port_var = w @ self.cov @ w
            mrc = (self.cov @ w) * w / (np.sqrt(port_var) + 1e-12)
            rc  = mrc / (mrc.sum() + 1e-12)
            return float(np.sum((rc - target_rc) ** 2))

        w0 = np.ones(self.n) / self.n
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=Bounds(lb=0.001, ub=self.max_w),
            constraints=self._constraints,
            options={"maxiter": 2000, "ftol": 1e-14},
        )
        w = result.x / result.x.sum()
        ret = self._port_return(w)
        vol = self._port_volatility(w)
        return OptimizedPortfolio(
            weights=w, tickers=self.tickers,
            ann_return=ret, ann_volatility=vol,
            sharpe_ratio=(ret - self.rf) / vol,
            label="Risk Parity Portfolio",
        )

    # ── Efficient Frontier ───────────────────────────────────────────────────

    def compute_frontier(self) -> pd.DataFrame:
        """
        Compute the full efficient frontier.

        Sweeps from min-variance to max-return portfolio,
        finding the minimum-risk portfolio for each target return level.

        Returns:
            DataFrame with columns: return, volatility, sharpe_ratio, weights_*
        """
        # Bounds for target returns
        ret_min = self._port_return(self.min_variance().weights)
        ret_max = self.mu.max()  # upper bound: max single asset return

        target_returns = np.linspace(ret_min, ret_max * 0.99, self.n_portfolios)

        records = []
        for target in target_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: w.sum() - 1.0},
                {"type": "eq", "fun": lambda w, t=target: self._port_return(w) - t},
            ]
            result = minimize(
                self._port_variance,
                np.ones(self.n) / self.n,
                method="SLSQP",
                bounds=self._bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if not result.success:
                continue

            w = result.x
            ret = self._port_return(w)
            vol = self._port_volatility(w)
            row = {
                "ann_return":    ret,
                "ann_volatility": vol,
                "sharpe_ratio":  (ret - self.rf) / (vol + 1e-12),
            }
            for i, ticker in enumerate(self.tickers):
                row[f"w_{ticker}"] = w[i]
            records.append(row)

        if not records:
            raise RuntimeError("Efficient frontier computation failed.")

        return pd.DataFrame(records)

    def summary(self) -> None:
        """Print the three key optimal portfolios."""
        print("\n" + "═"*45)
        print("  PORTFOLIO OPTIMIZATION RESULTS")
        print("═"*45)
        for portfolio in [self.min_variance(), self.max_sharpe(), self.risk_parity()]:
            print(portfolio)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    tickers = ["AAPL", "MSFT", "JPM", "GS", "GOOGL"]
    n = len(tickers)
    mu_true = np.array([0.0006, 0.0007, 0.0004, 0.0005, 0.0008])
    cov_base = np.full((n, n), 0.00015)
    np.fill_diagonal(cov_base, 0.0003)
    fake_rets = rng.multivariate_normal(mu_true, cov_base, size=1260)
    returns_df = pd.DataFrame(fake_rets, columns=tickers)

    ef = EfficientFrontier(returns_df)
    ef.summary()
