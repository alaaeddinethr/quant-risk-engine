"""
test_var.py
-----------
Unit tests for VaR/CVaR computations.
Tests mathematical properties, not exact values.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.risk.var import PortfolioVaR, VaREngine, VaRResult


def make_returns(seed=42, n=1260, mu=0.0005, sigma=0.012):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n))

def make_multi_returns(seed=0, n=1260, n_assets=4):
    rng = np.random.default_rng(seed)
    tickers = [f"A{i}" for i in range(n_assets)]
    data = rng.normal(0.0004, 0.011, (n, n_assets))
    return pd.DataFrame(data, columns=tickers)


class TestVaREngine:

    def setup_method(self):
        self.returns = make_returns()
        self.engine  = VaREngine(self.returns, confidence_levels=[0.95, 0.99])

    def test_historical_var_negative(self):
        r = self.engine.historical(0.95)
        assert r.var < 0, "VaR must be negative (a loss)"

    def test_cvar_worse_than_var(self):
        r = self.engine.historical(0.95)
        assert r.cvar <= r.var, "CVaR must be ≤ VaR (more extreme)"

    def test_var_99_worse_than_95(self):
        v95 = self.engine.historical(0.95).var
        v99 = self.engine.historical(0.99).var
        assert v99 <= v95, "99% VaR must be ≤ 95% VaR"

    def test_parametric_normal_formula(self):
        from scipy import stats
        mu  = self.returns.mean()
        sig = self.returns.std(ddof=1)
        expected_var = mu + stats.norm.ppf(0.05) * sig
        result = self.engine.parametric_normal(0.95)
        assert abs(result.var - expected_var) < 1e-10

    def test_student_t_fatter_tails_than_normal_on_fat_tailed_data(self):
        """Student-t has fatter tails — at extreme quantiles (99%) with very fat-tailed data
        the fitted df will be low and the t-distribution quantile more extreme than normal."""
        rng = np.random.default_rng(77)
        # Very fat-tailed data: t(df=2.5) has infinite 3rd moment
        fat_returns = pd.Series(rng.standard_t(df=2.5, size=5000) * 0.008)
        engine_fat = VaREngine(fat_returns)
        t_var = engine_fat.parametric_t(0.99).var
        n_var = engine_fat.parametric_normal(0.99).var
        assert t_var <= n_var, "Student-t VaR must be ≤ Normal VaR on very fat-tailed data at 99%"

    def test_monte_carlo_converges(self):
        r_large = self.engine.monte_carlo(0.95, n_simulations=100_000, random_seed=0)
        r_hist  = self.engine.historical(0.95)
        # With large N, MC should be close to historical
        assert abs(r_large.var - r_hist.var) < 0.005

    def test_all_methods_returns_4(self):
        results = self.engine.all_methods(0.95)
        assert len(results) == 4

    def test_var_result_is_dataclass(self):
        r = self.engine.historical(0.95)
        assert isinstance(r, VaRResult)
        assert hasattr(r, "var")
        assert hasattr(r, "cvar")
        assert hasattr(r, "method")

    def test_report_has_correct_shape(self):
        df = self.engine.report()
        # 2 confidence levels × 4 methods = 8 rows
        assert len(df) == 8
        assert "VaR (%)" in df.columns
        assert "CVaR (%)" in df.columns

    def test_parametric_normal_at_extreme_confidence(self):
        r = self.engine.parametric_normal(0.999)
        assert r.var < self.engine.parametric_normal(0.99).var

    def test_finite_values(self):
        for method in ["historical", "parametric_normal", "parametric_t", "monte_carlo"]:
            r = getattr(self.engine, method)(0.95)
            assert np.isfinite(r.var),  f"{method}: VaR is not finite"
            assert np.isfinite(r.cvar), f"{method}: CVaR is not finite"


class TestPortfolioVaR:

    def setup_method(self):
        self.rets = make_multi_returns(n_assets=4)
        self.pvar = PortfolioVaR(self.rets)

    def test_equal_weights_sum_to_one(self):
        assert abs(self.pvar.weights.sum() - 1.0) < 1e-10

    def test_custom_weights_normalized(self):
        pvar = PortfolioVaR(self.rets, weights=[1, 2, 1, 2])
        assert abs(pvar.weights.sum() - 1.0) < 1e-10

    def test_correlated_mc_var_negative(self):
        r = self.pvar.monte_carlo_correlated(0.95)
        assert r.var < 0

    def test_correlated_cvar_worse_than_var(self):
        r = self.pvar.monte_carlo_correlated(0.95)
        assert r.cvar <= r.var

    def test_full_report_runs(self):
        df = self.pvar.full_report(confidence_levels=[0.95], n_simulations=500)
        assert len(df) > 0
        assert "VaR (%)" in df.columns


class TestVaRProperties:
    """Mathematical properties that must hold for any VaR implementation."""

    def test_subadditivity_cvar(self):
        """CVaR is sub-additive: CVaR(A+B) ≤ CVaR(A) + CVaR(B)"""
        rng = np.random.default_rng(99)
        rA = pd.Series(rng.normal(0, 0.01, 500))
        rB = pd.Series(rng.normal(0, 0.01, 500))
        rAB = rA + rB

        eA  = VaREngine(rA).historical(0.95).cvar
        eB  = VaREngine(rB).historical(0.95).cvar
        eAB = VaREngine(rAB).historical(0.95).cvar

        assert eAB >= eA + eB, "CVaR should be sub-additive (diversification benefit)"

    def test_var_increases_with_portfolio_risk(self):
        """Higher-vol portfolio → worse VaR."""
        rng = np.random.default_rng(7)
        low_vol  = pd.Series(rng.normal(0, 0.005, 1000))
        high_vol = pd.Series(rng.normal(0, 0.020, 1000))

        var_low  = VaREngine(low_vol).parametric_normal(0.95).var
        var_high = VaREngine(high_vol).parametric_normal(0.95).var

        assert var_high < var_low, "Higher volatility → more extreme VaR"

    def test_positive_excess_return_shifts_var_up(self):
        """Adding a constant positive drift makes VaR less extreme."""
        rng = np.random.default_rng(11)
        base = pd.Series(rng.normal(0, 0.01, 1000))
        shifted = base + 0.01

        var_base    = VaREngine(base).parametric_normal(0.95).var
        var_shifted = VaREngine(shifted).parametric_normal(0.95).var

        assert var_shifted > var_base, "Positive drift should improve VaR"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
