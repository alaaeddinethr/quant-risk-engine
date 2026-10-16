"""
var.py
------
Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR / Expected Shortfall)
computed via four methods:

  1. Historical Simulation   — empirical percentile of P&L distribution
  2. Parametric Normal       — Gaussian assumption (mean, std)
  3. Parametric Student-t    — heavy-tailed distribution (fitted df)
  4. Monte Carlo             — Cholesky-correlated GBM simulation

All methods return NEGATIVE values (losses expressed as negative returns).

Theory:
  VaR_α   = quantile(losses, 1 - α)
  CVaR_α  = E[loss | loss < VaR_α]  (average of tail losses)

Reference:
  Jorion, P. (2006). Value at Risk: The New Benchmark for Managing Financial Risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# ─── Result Container ─────────────────────────────────────────────────────────

@dataclass
class VaRResult:
    """Holds VaR and CVaR values for a given method and confidence level."""
    method: str
    confidence: float
    var: float         # negative number (loss)
    cvar: float        # negative number (worse than VaR)
    n_obs: int = 0
    params: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[{self.method}] α={self.confidence:.0%} | "
            f"VaR={self.var*100:.4f}% | CVaR={self.cvar*100:.4f}%"
        )


# ─── VaR Engine ───────────────────────────────────────────────────────────────

class VaREngine:
    """
    Computes VaR and CVaR for a portfolio given its return series.

    Args:
        returns: pd.Series of daily portfolio returns
        confidence_levels: List of confidence levels [0.95, 0.99]
    """

    def __init__(
        self,
        returns: pd.Series,
        confidence_levels: list[float] = None,
    ):
        self.returns = np.asarray(returns, dtype=np.float64)
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.n = len(self.returns)

    # ── 1. Historical Simulation ─────────────────────────────────────────────

    def historical(self, confidence: float) -> VaRResult:
        """
        Non-parametric VaR: uses the empirical distribution of past returns.
        Most robust — no distributional assumption required.

        Limitation: requires sufficient history, does not extrapolate to
        unseen extreme events.
        """
        sorted_r = np.sort(self.returns)
        idx = int(np.floor((1 - confidence) * self.n))
        idx = max(0, min(idx, self.n - 1))

        var  = sorted_r[idx]
        cvar = sorted_r[:idx+1].mean() if idx > 0 else sorted_r[0]

        return VaRResult(
            method="Historical",
            confidence=confidence,
            var=var,
            cvar=cvar,
            n_obs=self.n,
        )

    # ── 2. Parametric Normal ─────────────────────────────────────────────────

    def parametric_normal(self, confidence: float) -> VaRResult:
        """
        Assumes returns follow a Normal distribution N(μ, σ²).
        Fast and analytically tractable.

        Limitation: underestimates tail risk (fat tails in real markets).
        """
        mu  = self.returns.mean()
        sig = self.returns.std(ddof=1)

        z   = stats.norm.ppf(1 - confidence)
        var = mu + z * sig
        # CVaR = μ - σ · φ(z) / (1-α)
        cvar = mu - sig * stats.norm.pdf(z) / (1 - confidence)

        return VaRResult(
            method="Parametric (Normal)",
            confidence=confidence,
            var=var,
            cvar=cvar,
            n_obs=self.n,
            params={"mu": mu, "sigma": sig},
        )

    # ── 3. Parametric Student-t ──────────────────────────────────────────────

    def parametric_t(self, confidence: float) -> VaRResult:
        """
        Fits a Student-t distribution to returns — better captures fat tails.

        The degrees of freedom (ν) is estimated via MLE.
        When ν → ∞, converges to Normal. Typical equity returns: ν ≈ 3–6.
        """
        df_mle, loc_mle, scale_mle = stats.t.fit(self.returns)

        t_q  = stats.t.ppf(1 - confidence, df=df_mle, loc=loc_mle, scale=scale_mle)
        var  = t_q

        # CVaR for t-distribution (analytical)
        z = (t_q - loc_mle) / scale_mle
        pdf_z = stats.t.pdf(z, df=df_mle)
        cvar = loc_mle - scale_mle * (pdf_z / (1 - confidence)) * (df_mle + z**2) / (df_mle - 1)

        return VaRResult(
            method="Parametric (Student-t)",
            confidence=confidence,
            var=var,
            cvar=cvar,
            n_obs=self.n,
            params={"df": df_mle, "loc": loc_mle, "scale": scale_mle},
        )

    # ── 4. Monte Carlo ───────────────────────────────────────────────────────

    def monte_carlo(
        self,
        confidence: float,
        n_simulations: int = 10_000,
        random_seed: int = 42,
    ) -> VaRResult:
        """
        Parametric Monte Carlo under GBM assumption.
        Simulates n_simulations 1-day returns from fitted Normal distribution.

        For multi-asset portfolios, use PortfolioVaR.monte_carlo() instead,
        which incorporates the full correlation structure via Cholesky.
        """
        rng = np.random.default_rng(random_seed)
        mu  = self.returns.mean()
        sig = self.returns.std(ddof=1)

        simulated = rng.normal(loc=mu, scale=sig, size=n_simulations)
        sorted_sim = np.sort(simulated)
        idx  = int(np.floor((1 - confidence) * n_simulations))
        var  = sorted_sim[idx]
        cvar = sorted_sim[:idx+1].mean()

        return VaRResult(
            method="Monte Carlo",
            confidence=confidence,
            var=var,
            cvar=cvar,
            n_obs=n_simulations,
            params={"n_simulations": n_simulations, "mu": mu, "sigma": sig},
        )

    def all_methods(self, confidence: float, n_simulations: int = 10_000) -> list[VaRResult]:
        """Run all 4 methods for a given confidence level."""
        return [
            self.historical(confidence),
            self.parametric_normal(confidence),
            self.parametric_t(confidence),
            self.monte_carlo(confidence, n_simulations),
        ]

    def report(self, n_simulations: int = 10_000) -> pd.DataFrame:
        """
        Full VaR/CVaR report across all methods and confidence levels.
        Returns a tidy DataFrame.
        """
        rows = []
        for alpha in self.confidence_levels:
            for result in self.all_methods(alpha, n_simulations):
                rows.append({
                    "Method":     result.method,
                    "Confidence": f"{result.confidence:.0%}",
                    "VaR (%)":    round(result.var * 100, 4),
                    "CVaR (%)":   round(result.cvar * 100, 4),
                    "VaR ($1M)":  f"${abs(result.var) * 1_000_000:,.0f}",
                    "CVaR ($1M)": f"${abs(result.cvar) * 1_000_000:,.0f}",
                })
        return pd.DataFrame(rows)


# ─── Multi-Asset Portfolio VaR ────────────────────────────────────────────────

class PortfolioVaR:
    """
    VaR/CVaR engine for a multi-asset portfolio.

    Handles proper correlation structure for Monte Carlo simulations
    using Cholesky decomposition.

    Args:
        returns_df: DataFrame of individual asset returns (dates × assets)
        weights: Asset weights (must sum to 1). None = equal weight.
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray | list | None = None,
    ):
        self.returns_df = returns_df
        n = returns_df.shape[1]

        if weights is None:
            self.weights = np.ones(n) / n
        else:
            w = np.asarray(weights, dtype=float)
            self.weights = w / w.sum()  # normalize

        # Portfolio return series
        self.port_returns = (returns_df * self.weights).sum(axis=1)
        self.engine = VaREngine(self.port_returns)

    def monte_carlo_correlated(
        self,
        confidence: float,
        n_simulations: int = 10_000,
        random_seed: int = 42,
    ) -> VaRResult:
        """
        Monte Carlo with full correlation structure.

        Uses Cholesky decomposition of the covariance matrix to generate
        correlated random returns, then computes portfolio P&L.

        Steps:
          1. Estimate μ vector and Σ covariance matrix from historical data
          2. Cholesky: Σ = L · Lᵀ
          3. Simulate: r = μ + L · z,  z ~ N(0, I)
          4. Portfolio return: rₚ = w · r
          5. VaR/CVaR from simulated distribution
        """
        rng = np.random.default_rng(random_seed)
        rets = self.returns_df.values
        mu   = rets.mean(axis=0)
        cov  = np.cov(rets, rowvar=False)

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Regularize if not positive definite
            eps = 1e-8
            cov += eps * np.eye(cov.shape[0])
            L   = np.linalg.cholesky(cov)

        # Simulate correlated returns
        z = rng.standard_normal((n_simulations, len(mu)))
        sim_returns = mu + z @ L.T    # shape: (n_sim, n_assets)
        port_sim = sim_returns @ self.weights

        sorted_sim = np.sort(port_sim)
        idx  = int(np.floor((1 - confidence) * n_simulations))
        var  = sorted_sim[idx]
        cvar = sorted_sim[:idx+1].mean()

        return VaRResult(
            method="Monte Carlo (Correlated)",
            confidence=confidence,
            var=var,
            cvar=cvar,
            n_obs=n_simulations,
            params={"n_simulations": n_simulations},
        )

    def full_report(
        self,
        confidence_levels: list[float] = None,
        n_simulations: int = 10_000,
    ) -> pd.DataFrame:
        """Complete VaR/CVaR report including correlated Monte Carlo."""
        confidence_levels = confidence_levels or [0.95, 0.99]
        rows = []

        for alpha in confidence_levels:
            # Standard methods
            for result in self.engine.all_methods(alpha, n_simulations):
                rows.append({
                    "Method":     result.method,
                    "Confidence": f"{alpha:.0%}",
                    "VaR (%)":    round(result.var * 100, 4),
                    "CVaR (%)":   round(result.cvar * 100, 4),
                })
            # Correlated MC
            r_mc = self.monte_carlo_correlated(alpha, n_simulations)
            rows.append({
                "Method":     r_mc.method,
                "Confidence": f"{alpha:.0%}",
                "VaR (%)":    round(r_mc.var * 100, 4),
                "CVaR (%)":   round(r_mc.cvar * 100, 4),
            })

        df = pd.DataFrame(rows)
        print("\n" + "═"*60)
        print("  PORTFOLIO VAR / CVaR REPORT")
        print("═"*60)
        print(df.to_string(index=False))
        return df


if __name__ == "__main__":
    # Smoke test with synthetic returns
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.0005, 0.012, 1260), name="Portfolio")

    engine = VaREngine(returns, confidence_levels=[0.95, 0.99])
    report = engine.report()
    print(report.to_string(index=False))
