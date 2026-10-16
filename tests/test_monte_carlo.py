"""
test_monte_carlo.py
-------------------
Unit tests for Monte Carlo simulation engine.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.risk.monte_carlo import MonteCarloEngine, SimulationResult


def make_multi_returns(seed=0, n=500, n_assets=3):
    rng = np.random.default_rng(seed)
    tickers = ["A", "B", "C"][:n_assets]
    cov = np.eye(n_assets) * 0.0002 + np.full((n_assets, n_assets), 0.00005)
    mu  = np.full(n_assets, 0.0004)
    data = rng.multivariate_normal(mu, cov, size=n)
    return pd.DataFrame(data, columns=tickers)


class TestMonteCarloEngine:

    def setup_method(self):
        self.rets = make_multi_returns()
        self.engine = MonteCarloEngine(self.rets, initial_value=1_000_000)

    def test_run_returns_result(self):
        result = self.engine.run(horizon=21, n_simulations=500)
        assert isinstance(result, SimulationResult)

    def test_paths_shape(self):
        result = self.engine.run(horizon=21, n_simulations=500)
        assert result.portfolio_values.shape == (500, 22)  # 21 days + day 0

    def test_initial_value_correct(self):
        result = self.engine.run(horizon=21, n_simulations=200)
        initial_col = result.portfolio_values[:, 0]
        assert np.allclose(initial_col, 1_000_000, rtol=1e-6)

    def test_var_negative(self):
        result = self.engine.run(horizon=21, n_simulations=1000)
        assert result.var(0.95) < 0

    def test_cvar_worse_than_var(self):
        result = self.engine.run(horizon=21, n_simulations=1000)
        assert result.cvar(0.95) <= result.var(0.95)

    def test_var_99_worse_than_95(self):
        result = self.engine.run(horizon=21, n_simulations=2000)
        assert result.var(0.99) <= result.var(0.95)

    def test_prob_loss_between_zero_and_one(self):
        result = self.engine.run(horizon=63, n_simulations=500)
        p = result.prob_loss()
        assert 0.0 <= p <= 1.0

    def test_percentile_bands_ordering(self):
        result = self.engine.run(horizon=21, n_simulations=500)
        bands = result.percentile_bands()
        # p5 ≤ p25 ≤ median ≤ p75 ≤ p95 at every point
        assert np.all(bands["p5"] <= bands["p25"] + 1e-8)
        assert np.all(bands["p25"] <= bands["median"] + 1e-8)
        assert np.all(bands["median"] <= bands["p75"] + 1e-8)
        assert np.all(bands["p75"] <= bands["p95"] + 1e-8)

    def test_reproducibility(self):
        r1 = self.engine.run(horizon=21, n_simulations=100, random_seed=42)
        r2 = self.engine.run(horizon=21, n_simulations=100, random_seed=42)
        assert np.allclose(r1.portfolio_values, r2.portfolio_values)

    def test_different_seeds_different_results(self):
        r1 = self.engine.run(horizon=21, n_simulations=100, random_seed=1)
        r2 = self.engine.run(horizon=21, n_simulations=100, random_seed=2)
        assert not np.allclose(r1.portfolio_values, r2.portfolio_values)

    def test_bootstrap_method(self):
        result = self.engine.run(horizon=21, n_simulations=200, method="historical")
        assert result.portfolio_values.shape == (200, 22)

    def test_higher_vol_wider_bands(self):
        """Higher volatility assets should produce wider simulation bands."""
        rng = np.random.default_rng(99)
        tickers = ["X", "Y"]
        low_cov  = np.eye(2) * 0.00005
        high_cov = np.eye(2) * 0.0005
        low_rets  = pd.DataFrame(rng.multivariate_normal([0.0004]*2, low_cov,  300), columns=tickers)
        high_rets = pd.DataFrame(rng.multivariate_normal([0.0004]*2, high_cov, 300), columns=tickers)

        r_low  = MonteCarloEngine(low_rets,  1e6).run(horizon=63, n_simulations=1000)
        r_high = MonteCarloEngine(high_rets, 1e6).run(horizon=63, n_simulations=1000)

        spread_low  = r_high.terminal_values.std()
        spread_high = r_low.terminal_values.std()
        assert spread_low > spread_high

    def test_summary_keys(self):
        result = self.engine.run(horizon=21, n_simulations=200)
        summary = result.summary()
        required = ["var_95", "cvar_95", "prob_loss", "median_terminal"]
        for key in required:
            assert key in summary

    def test_equal_weights_default(self):
        engine = MonteCarloEngine(self.rets)
        assert np.allclose(engine.weights, np.ones(3) / 3)

    def test_weights_normalization(self):
        engine = MonteCarloEngine(self.rets, weights=[1, 2, 1])
        assert abs(engine.weights.sum() - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
