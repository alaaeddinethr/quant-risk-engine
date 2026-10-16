"""
monte_carlo.py
--------------
Full portfolio Monte Carlo simulation engine.

Simulates N trajectories of portfolio value over a given time horizon
using Geometric Brownian Motion (GBM) with correlated assets.

Model: dS_i = μ_i S_i dt + σ_i S_i dW_i
where dW_i are correlated Wiener processes (Cholesky decomposition).

Output: N × T matrix of portfolio values, from which we compute:
  - Percentile bands (5th, 25th, 50th, 75th, 95th)
  - VaR at horizon H
  - Probability of loss > threshold
  - Distribution of terminal wealth

Reference:
    Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation output."""
    paths: np.ndarray           # shape: (n_sim, horizon+1)
    portfolio_values: np.ndarray  # initial value × paths
    initial_value: float
    n_simulations: int
    horizon: int
    weights: np.ndarray
    ticker_names: list[str]

    @property
    def terminal_values(self) -> np.ndarray:
        return self.portfolio_values[:, -1]

    @property
    def terminal_returns(self) -> np.ndarray:
        return self.terminal_values / self.initial_value - 1

    def var(self, confidence: float = 0.95) -> float:
        """VaR at horizon H as a return (negative = loss)."""
        return np.percentile(self.terminal_returns, (1 - confidence) * 100)

    def cvar(self, confidence: float = 0.95) -> float:
        """CVaR: mean of returns below VaR."""
        v = self.var(confidence)
        tail = self.terminal_returns[self.terminal_returns <= v]
        return tail.mean() if len(tail) > 0 else v

    def prob_loss(self, threshold: float = 0.0) -> float:
        """P(terminal return < threshold)"""
        return (self.terminal_returns < threshold).mean()

    def prob_loss_exceeds(self, loss_pct: float = 0.10) -> float:
        """P(loss > loss_pct), e.g. P(loss > 10%) = P(return < -0.10)"""
        return (self.terminal_returns < -loss_pct).mean()

    def percentile_bands(self) -> dict[str, np.ndarray]:
        """Returns time-series percentile bands across all simulations."""
        return {
            "p5":     np.percentile(self.portfolio_values, 5,  axis=0),
            "p25":    np.percentile(self.portfolio_values, 25, axis=0),
            "median": np.percentile(self.portfolio_values, 50, axis=0),
            "p75":    np.percentile(self.portfolio_values, 75, axis=0),
            "p95":    np.percentile(self.portfolio_values, 95, axis=0),
            "mean":   self.portfolio_values.mean(axis=0),
        }

    def summary(self) -> dict:
        bands = self.percentile_bands()
        return {
            "n_simulations":     self.n_simulations,
            "horizon_days":      self.horizon,
            "initial_value":     self.initial_value,
            "median_terminal":   bands["median"][-1],
            "mean_terminal":     bands["mean"][-1],
            "p5_terminal":       bands["p5"][-1],
            "p95_terminal":      bands["p95"][-1],
            "var_95":            self.var(0.95),
            "var_99":            self.var(0.99),
            "cvar_95":           self.cvar(0.95),
            "cvar_99":           self.cvar(0.99),
            "prob_loss":         self.prob_loss(0.0),
            "prob_loss_10pct":   self.prob_loss_exceeds(0.10),
            "prob_loss_20pct":   self.prob_loss_exceeds(0.20),
        }


class MonteCarloEngine:
    """
    Simulates portfolio value trajectories using correlated GBM.

    Args:
        returns_df: Historical returns (dates × assets)
        weights: Portfolio weights (None = equal weight)
        initial_value: Starting portfolio value in $
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray | None = None,
        initial_value: float = 1_000_000,
    ):
        self.returns_df = returns_df
        self.n_assets = returns_df.shape[1]
        self.ticker_names = list(returns_df.columns)
        self.initial_value = initial_value

        if weights is None:
            self.weights = np.ones(self.n_assets) / self.n_assets
        else:
            w = np.asarray(weights, dtype=float)
            self.weights = w / w.sum()

        # Estimate parameters from historical data
        rets = returns_df.values
        self.mu  = rets.mean(axis=0)           # daily mean returns
        self.cov = np.cov(rets, rowvar=False)  # covariance matrix

        # Cholesky decomposition for correlated simulation
        self._L = self._cholesky(self.cov)

    def _cholesky(self, cov: np.ndarray) -> np.ndarray:
        """Compute Cholesky factor with regularization if needed."""
        try:
            return np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Add small diagonal regularization
            eps = 1e-8
            while eps < 1e-3:
                try:
                    return np.linalg.cholesky(cov + eps * np.eye(cov.shape[0]))
                except np.linalg.LinAlgError:
                    eps *= 10
            raise ValueError("Covariance matrix cannot be made positive definite.")

    def run(
        self,
        horizon: int = 252,
        n_simulations: int = 10_000,
        random_seed: int = 42,
        method: str = "gbm",
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            horizon: Simulation horizon in trading days (252 = 1 year)
            n_simulations: Number of paths to simulate
            random_seed: Reproducibility seed
            method: 'gbm' (Geometric Brownian Motion) or 'historical' (bootstrap)

        Returns:
            SimulationResult with all paths and computed statistics
        """
        rng = np.random.default_rng(random_seed)

        if method == "gbm":
            paths = self._simulate_gbm(rng, horizon, n_simulations)
        elif method == "historical":
            paths = self._simulate_bootstrap(rng, horizon, n_simulations)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Convert asset paths to portfolio value paths
        # paths shape: (n_sim, horizon+1, n_assets) — relative to initial=1
        port_paths = (paths * self.weights).sum(axis=2)  # (n_sim, horizon+1)
        portfolio_values = self.initial_value * port_paths

        return SimulationResult(
            paths=port_paths,
            portfolio_values=portfolio_values,
            initial_value=self.initial_value,
            n_simulations=n_simulations,
            horizon=horizon,
            weights=self.weights,
            ticker_names=self.ticker_names,
        )

    def _simulate_gbm(
        self,
        rng: np.random.Generator,
        horizon: int,
        n_simulations: int,
    ) -> np.ndarray:
        """
        Geometric Brownian Motion simulation.

        For each time step:
          ln(S_{t+1}/S_t) = (μ - σ²/2)dt + σ√dt · ε
        where ε ~ N(0,1) are correlated via Cholesky.

        Returns:
            Array of shape (n_sim, horizon+1, n_assets)
            Values are price relatives (S_t / S_0), starting at 1.0
        """
        dt = 1.0  # 1 trading day
        sig = np.sqrt(np.diag(self.cov))  # individual volatilities
        drift = (self.mu - 0.5 * sig**2) * dt

        # Draw standard normal shocks: (n_sim, horizon, n_assets)
        z = rng.standard_normal((n_simulations, horizon, self.n_assets))

        # Apply Cholesky correlation: z_corr = z @ L.T
        z_corr = z @ self._L.T  # shape: (n_sim, horizon, n_assets)

        # Log returns at each step
        log_returns = drift + z_corr  # (n_sim, horizon, n_assets)

        # Cumulative sum → price paths (normalized to start at 1)
        cum_log = np.concatenate(
            [np.zeros((n_simulations, 1, self.n_assets)), np.cumsum(log_returns, axis=1)],
            axis=1,
        )
        return np.exp(cum_log)  # (n_sim, horizon+1, n_assets)

    def _simulate_bootstrap(
        self,
        rng: np.random.Generator,
        horizon: int,
        n_simulations: int,
    ) -> np.ndarray:
        """
        Historical bootstrap simulation.
        Resamples actual observed return days (with replacement).
        No distributional assumption — captures real market regime changes.
        """
        rets = self.returns_df.values  # (T, n_assets)
        T = len(rets)

        # Draw random indices
        idx = rng.integers(0, T, size=(n_simulations, horizon))
        sampled = rets[idx]  # (n_sim, horizon, n_assets)

        cum_log = np.concatenate(
            [np.zeros((n_simulations, 1, self.n_assets)), np.cumsum(sampled, axis=1)],
            axis=1,
        )
        return np.exp(cum_log)

    def print_summary(self, result: SimulationResult) -> None:
        s = result.summary()
        iv = s["initial_value"]
        print(f"\n{'═'*55}")
        print(f"  Monte Carlo Simulation — {s['n_simulations']:,} paths × {s['horizon_days']} days")
        print(f"{'═'*55}")
        print(f"  Initial value:     ${iv:>15,.0f}")
        print(f"  Median terminal:   ${s['median_terminal']:>15,.0f}  ({(s['median_terminal']/iv-1)*100:+.1f}%)")
        print(f"  95th percentile:   ${s['p95_terminal']:>15,.0f}  ({(s['p95_terminal']/iv-1)*100:+.1f}%)")
        print(f"  5th percentile:    ${s['p5_terminal']:>15,.0f}  ({(s['p5_terminal']/iv-1)*100:+.1f}%)")
        print(f"{'─'*55}")
        print(f"  VaR 95% (horizon): {s['var_95']*100:>10.4f}%")
        print(f"  CVaR 95%:          {s['cvar_95']*100:>10.4f}%")
        print(f"  VaR 99%:           {s['var_99']*100:>10.4f}%")
        print(f"  CVaR 99%:          {s['cvar_99']*100:>10.4f}%")
        print(f"{'─'*55}")
        print(f"  P(loss > 0%):      {s['prob_loss']*100:>10.1f}%")
        print(f"  P(loss > 10%):     {s['prob_loss_10pct']*100:>10.1f}%")
        print(f"  P(loss > 20%):     {s['prob_loss_20pct']*100:>10.1f}%")
        print(f"{'═'*55}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Synthetic 5-asset portfolio
    tickers = ["AAPL", "MSFT", "JPM", "GS", "GOOGL"]
    n_assets = len(tickers)
    # Simulate returns with realistic correlation
    cov_base = np.full((n_assets, n_assets), 0.0002)
    np.fill_diagonal(cov_base, 0.0003)
    fake_rets = rng.multivariate_normal([0.0004]*n_assets, cov_base, size=1260)
    returns_df = pd.DataFrame(fake_rets, columns=tickers)

    engine = MonteCarloEngine(returns_df, initial_value=1_000_000)
    result = engine.run(horizon=252, n_simulations=5000, random_seed=42)
    engine.print_summary(result)
