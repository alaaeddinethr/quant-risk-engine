"""
test_portfolio.py
-----------------
Unit tests for portfolio metrics and optimization.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.portfolio.metrics import PortfolioAnalyzer, PortfolioMetrics, asset_contribution
from src.portfolio.optimizer import EfficientFrontier, OptimizedPortfolio


def make_returns(seed=0, n=1260, mu=0.0005, sigma=0.011):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n),
                     index=pd.bdate_range("2019-01-01", periods=n))

def make_multi_returns(seed=0, n=500, n_assets=3):
    rng = np.random.default_rng(seed)
    tickers = ["AAPL", "MSFT", "JPM"][:n_assets]
    cov = np.array([[0.0003, 0.0001, 0.00005],
                    [0.0001, 0.0003, 0.00005],
                    [0.00005, 0.00005, 0.0003]])[:n_assets, :n_assets]
    mu = np.array([0.0005, 0.0006, 0.0004])[:n_assets]
    data = rng.multivariate_normal(mu, cov, size=n)
    return pd.DataFrame(data, columns=tickers)


class TestPortfolioMetrics:

    def setup_method(self):
        self.returns = make_returns()
        self.analyzer = PortfolioAnalyzer(self.returns)

    def test_compute_all_returns_dataclass(self):
        m = self.analyzer.compute_all()
        assert isinstance(m, PortfolioMetrics)

    def test_sharpe_formula(self):
        rf_daily = 0.045 / 252
        excess = self.returns - rf_daily
        expected = (excess.mean() / excess.std(ddof=1)) * np.sqrt(252)
        assert abs(self.analyzer.sharpe_ratio() - expected) < 1e-8

    def test_sortino_positive_high_return(self):
        rng = np.random.default_rng(5)
        good = pd.Series(rng.normal(0.002, 0.005, 500))
        analyzer = PortfolioAnalyzer(good)
        assert analyzer.sortino_ratio() > 0

    def test_max_drawdown_negative(self):
        mdd = self.analyzer.max_drawdown()
        assert mdd <= 0, "Max drawdown must be ≤ 0"

    def test_max_drawdown_bounded(self):
        mdd = self.analyzer.max_drawdown()
        assert mdd >= -1.0, "Max drawdown can't exceed -100%"

    def test_drawdown_series_starts_at_zero(self):
        dd = self.analyzer.drawdown_series()
        assert dd.iloc[0] == 0.0

    def test_var_negative(self):
        assert self.analyzer.var(0.95) < 0

    def test_cvar_worse_than_var(self):
        assert self.analyzer.cvar(0.95) <= self.analyzer.var(0.95)

    def test_all_metrics_finite(self):
        m = self.analyzer.compute_all()
        for field_name in m.__dataclass_fields__:
            val = getattr(m, field_name)
            if isinstance(val, float):
                assert np.isfinite(val), f"{field_name} is not finite: {val}"

    def test_rolling_metrics_shape(self):
        rolling = self.analyzer.rolling_metrics(window=63)
        assert "rolling_sharpe" in rolling.columns
        assert len(rolling) == len(self.returns) - 63 + 1

    def test_annualized_vol_scaling(self):
        """Annual vol = daily vol × √252"""
        daily_std = self.returns.std(ddof=1)
        expected_ann = daily_std * np.sqrt(252)
        assert abs(self.analyzer.ann_volatility() - expected_ann) < 1e-10

    def test_cagr_consistent_with_total_return(self):
        """CAGR over N years ≈ (1 + total_return)^(1/N) - 1"""
        n_years = len(self.returns) / 252
        total = self.analyzer.total_return()
        expected_cagr = (1 + total) ** (1 / n_years) - 1
        assert abs(self.analyzer.cagr() - expected_cagr) < 1e-8


class TestEfficientFrontier:

    def setup_method(self):
        self.rets = make_multi_returns(n=500, n_assets=3)
        self.ef   = EfficientFrontier(self.rets, n_portfolios=50)

    def test_min_variance_weights_sum_to_one(self):
        mvp = self.ef.min_variance()
        assert abs(mvp.weights.sum() - 1.0) < 1e-6

    def test_max_sharpe_weights_sum_to_one(self):
        msr = self.ef.max_sharpe()
        assert abs(msr.weights.sum() - 1.0) < 1e-6

    def test_risk_parity_weights_sum_to_one(self):
        rp = self.ef.risk_parity()
        assert abs(rp.weights.sum() - 1.0) < 1e-6

    def test_max_sharpe_better_than_min_variance_sharpe(self):
        mvp = self.ef.min_variance()
        msr = self.ef.max_sharpe()
        assert msr.sharpe_ratio >= mvp.sharpe_ratio - 1e-4

    def test_min_variance_lower_vol_than_equal_weight(self):
        mvp = self.ef.min_variance()
        eq_w = np.ones(3) / 3
        eq_vol = np.sqrt(eq_w @ (self.rets.cov().values * 252) @ eq_w)
        assert mvp.ann_volatility <= eq_vol + 1e-6

    def test_all_weights_within_bounds(self):
        for port in [self.ef.min_variance(), self.ef.max_sharpe()]:
            assert all(w >= self.ef.min_w - 1e-6 for w in port.weights)
            assert all(w <= self.ef.max_w + 1e-6 for w in port.weights)

    def test_optimized_portfolio_is_dataclass(self):
        assert isinstance(self.ef.min_variance(), OptimizedPortfolio)

    def test_frontier_returns_dataframe(self):
        # Use a larger n_portfolios to ensure enough converging points
        ef = EfficientFrontier(self.rets, n_portfolios=200)
        frontier = ef.compute_frontier()
        assert isinstance(frontier, pd.DataFrame)
        assert "ann_return" in frontier.columns
        assert "ann_volatility" in frontier.columns
        assert len(frontier) > 5

    def test_risk_parity_equal_contributions(self):
        """Risk Parity portfolio should have roughly equal risk contributions."""
        rp = self.ef.risk_parity()
        w = rp.weights
        cov = self.rets.cov().values * 252
        port_vol = np.sqrt(w @ cov @ w)
        mrc = (cov @ w) * w / port_vol
        rc = mrc / port_vol
        # All contributions should be within 10% of each other
        assert np.std(rc) < 0.10, f"Risk contributions not balanced: {rc}"


class TestAssetContribution:

    def test_contributions_sum_to_100(self):
        rets = make_multi_returns(n=500, n_assets=3)
        w = np.ones(3) / 3
        contrib = asset_contribution(rets, w)
        total = contrib["Risk Contribution (%)"].sum()
        assert abs(total - 100.0) < 0.5

    def test_output_has_correct_columns(self):
        rets = make_multi_returns(n=500, n_assets=3)
        contrib = asset_contribution(rets, np.ones(3)/3)
        assert "Weight (%)" in contrib.columns
        assert "Risk Contribution (%)" in contrib.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
