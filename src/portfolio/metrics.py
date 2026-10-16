"""
metrics.py
----------
Portfolio performance and risk metrics.

Covers:
  - Return metrics: CAGR, annualized return, Sharpe, Sortino, Calmar
  - Risk metrics: Volatility, Max Drawdown, Drawdown duration
  - Tail metrics: Skewness, Kurtosis, VaR contributions per asset

All metrics annualized assuming 252 trading days/year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PortfolioMetrics:
    """All performance and risk metrics for a portfolio."""
    # Return
    total_return: float
    cagr: float
    ann_return: float
    ann_volatility: float
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int   # trading days
    current_drawdown: float
    # Tail
    skewness: float
    excess_kurtosis: float
    var_95: float
    cvar_95: float
    # Info
    n_days: int
    start_date: str
    end_date: str

    def __str__(self) -> str:
        lines = [
            f"  Annual Return:    {self.ann_return*100:>8.2f}%",
            f"  Annual Volatility:{self.ann_volatility*100:>8.2f}%",
            f"  Sharpe Ratio:     {self.sharpe_ratio:>8.3f}",
            f"  Sortino Ratio:    {self.sortino_ratio:>8.3f}",
            f"  Calmar Ratio:     {self.calmar_ratio:>8.3f}",
            f"  Max Drawdown:     {self.max_drawdown*100:>8.2f}%",
            f"  VaR 95%:          {self.var_95*100:>8.4f}%",
            f"  CVaR 95%:         {self.cvar_95*100:>8.4f}%",
            f"  Skewness:         {self.skewness:>8.3f}",
            f"  Excess Kurtosis:  {self.excess_kurtosis:>8.3f}",
        ]
        return "\n".join(lines)


class PortfolioAnalyzer:
    """
    Computes comprehensive performance and risk metrics for a portfolio.

    Args:
        returns: Daily portfolio returns as a pd.Series
        risk_free_rate: Annual risk-free rate (default: 4.5%)
    """

    TRADING_DAYS = 252

    def __init__(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.045,
    ):
        self.returns = returns.dropna()
        self.rf_daily = risk_free_rate / self.TRADING_DAYS
        self.n = len(self.returns)

    # ── Return metrics ────────────────────────────────────────────────────────

    def total_return(self) -> float:
        return np.exp(self.returns.sum()) - 1

    def cagr(self) -> float:
        """Compound Annual Growth Rate."""
        total = np.exp(self.returns.sum())
        years = self.n / self.TRADING_DAYS
        return total ** (1 / years) - 1

    def ann_return(self) -> float:
        return self.returns.mean() * self.TRADING_DAYS

    def ann_volatility(self) -> float:
        return self.returns.std(ddof=1) * np.sqrt(self.TRADING_DAYS)

    # ── Drawdown ─────────────────────────────────────────────────────────────

    def drawdown_series(self) -> pd.Series:
        """Daily drawdown from peak as a fraction."""
        wealth = np.exp(self.returns.cumsum())
        rolling_max = wealth.cummax()
        return (wealth - rolling_max) / rolling_max

    def max_drawdown(self) -> float:
        return self.drawdown_series().min()

    def max_drawdown_duration(self) -> int:
        """Longest drawdown in trading days."""
        dd = self.drawdown_series()
        in_drawdown = (dd < 0).astype(int)
        # Count consecutive days below peak
        max_dur = 0
        cur_dur = 0
        for v in in_drawdown:
            cur_dur = cur_dur + 1 if v else 0
            max_dur = max(max_dur, cur_dur)
        return max_dur

    def current_drawdown(self) -> float:
        return self.drawdown_series().iloc[-1]

    # ── Risk-adjusted returns ─────────────────────────────────────────────────

    def sharpe_ratio(self) -> float:
        """Annualized Sharpe Ratio."""
        excess = self.returns - self.rf_daily
        if excess.std(ddof=1) == 0:
            return 0.0
        return (excess.mean() / excess.std(ddof=1)) * np.sqrt(self.TRADING_DAYS)

    def sortino_ratio(self) -> float:
        """
        Sortino Ratio — penalizes only downside volatility.
        More appropriate than Sharpe when return distribution is skewed.
        """
        excess = self.returns - self.rf_daily
        downside = excess[excess < 0]
        downside_std = np.sqrt((downside**2).mean()) * np.sqrt(self.TRADING_DAYS)
        if downside_std == 0:
            return 0.0
        return (excess.mean() * self.TRADING_DAYS) / downside_std

    def calmar_ratio(self) -> float:
        """
        Calmar Ratio = CAGR / |Max Drawdown|
        Useful for comparing strategies with different drawdown profiles.
        """
        mdd = abs(self.max_drawdown())
        if mdd == 0:
            return 0.0
        return self.cagr() / mdd

    # ── Tail risk ─────────────────────────────────────────────────────────────

    def var(self, confidence: float = 0.95) -> float:
        """Historical VaR."""
        return np.percentile(self.returns, (1 - confidence) * 100)

    def cvar(self, confidence: float = 0.95) -> float:
        """Historical CVaR (Expected Shortfall)."""
        v = self.var(confidence)
        tail = self.returns[self.returns <= v]
        return tail.mean() if len(tail) > 0 else v

    # ── Distribution ──────────────────────────────────────────────────────────

    def skewness(self) -> float:
        return float(stats.skew(self.returns))

    def excess_kurtosis(self) -> float:
        return float(stats.kurtosis(self.returns))  # Fisher's (excess)

    def normality_test(self) -> tuple[float, float]:
        """Jarque-Bera test for normality. Returns (statistic, p-value)."""
        stat, p = stats.jarque_bera(self.returns)
        return float(stat), float(p)

    # ── Full report ───────────────────────────────────────────────────────────

    def compute_all(self) -> PortfolioMetrics:
        """Compute and return all metrics."""
        idx = self.returns.index
        return PortfolioMetrics(
            total_return      = self.total_return(),
            cagr              = self.cagr(),
            ann_return        = self.ann_return(),
            ann_volatility    = self.ann_volatility(),
            sharpe_ratio      = self.sharpe_ratio(),
            sortino_ratio     = self.sortino_ratio(),
            calmar_ratio      = self.calmar_ratio(),
            max_drawdown      = self.max_drawdown(),
            max_drawdown_duration = self.max_drawdown_duration(),
            current_drawdown  = self.current_drawdown(),
            skewness          = self.skewness(),
            excess_kurtosis   = self.excess_kurtosis(),
            var_95            = self.var(0.95),
            cvar_95           = self.cvar(0.95),
            n_days            = self.n,
            start_date        = str(idx[0].date()),
            end_date          = str(idx[-1].date()),
        )

    def rolling_metrics(self, window: int = 63) -> pd.DataFrame:
        """
        Rolling performance metrics over a sliding window.

        Args:
            window: Rolling window in trading days (63 ≈ 1 quarter)

        Returns:
            DataFrame with columns: sharpe, volatility, var_95
        """
        rolling_sharpe = []
        rolling_vol    = []
        rolling_var    = []

        for i in range(window, len(self.returns) + 1):
            window_rets = self.returns.iloc[i - window:i]
            analyzer = PortfolioAnalyzer(window_rets, self.rf_daily * self.TRADING_DAYS)
            rolling_sharpe.append(analyzer.sharpe_ratio())
            rolling_vol.append(analyzer.ann_volatility())
            rolling_var.append(analyzer.var(0.95))

        idx = self.returns.index[window - 1:]
        return pd.DataFrame({
            "rolling_sharpe":     rolling_sharpe,
            "rolling_volatility": rolling_vol,
            "rolling_var_95":     rolling_var,
        }, index=idx)


def asset_contribution(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float = 0.045,
) -> pd.DataFrame:
    """
    Marginal Risk Contribution (MRC) per asset.

    MRC_i = w_i * (Σw)_i / σ_portfolio

    Returns DataFrame showing each asset's contribution to total portfolio risk.
    """
    w = np.asarray(weights) / np.asarray(weights).sum()
    cov = returns_df.cov().values * 252
    port_var = w @ cov @ w
    port_vol = np.sqrt(port_var)

    mrc = (cov @ w) * w / port_vol
    pct_contribution = mrc / port_vol * 100

    return pd.DataFrame({
        "Weight (%)":            w * 100,
        "Marginal Risk Contrib": mrc,
        "Risk Contribution (%)": pct_contribution,
    }, index=returns_df.columns)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    rets = pd.Series(
        rng.normal(0.0006, 0.011, 1260),
        index=pd.bdate_range("2019-01-01", periods=1260),
        name="Portfolio",
    )
    analyzer = PortfolioAnalyzer(rets)
    metrics = analyzer.compute_all()
    print(metrics)
