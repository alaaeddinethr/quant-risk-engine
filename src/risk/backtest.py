"""VaR backtesting: does the model's VaR get breached as often as it should?

A 99 % one-day VaR should be exceeded on about 1 day in 100. Backtesting
re-estimates the VaR every day from the previous window only (no look-ahead),
compares it with the next day's realised return and counts the exceptions.

- Kupiec (1995) proportion-of-failures test: is the exception rate
  statistically compatible with 1 - confidence?
- Methods: historical simulation, normal, Student-t (fitted on the window),
  EWMA volatility (RiskMetrics, lambda = 0.94) and filtered historical
  simulation (Barone-Adesi et al., 1999): returns are divided by their EWMA
  volatility, the empirical quantile of these standardised shocks is taken,
  then rescaled by tomorrow's volatility. Fat tails and volatility clustering
  are both captured.
- Basel traffic light (BCBS, 1996): over 250 days at 99 %, 0-4 exceptions is
  green, 5-9 yellow, 10 or more red.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ("historical", "normal", "student_t", "ewma", "fhs")
EWMA_LAMBDA = 0.94  # RiskMetrics (J.P. Morgan, 1996) decay for daily data


def var_forecast(window: np.ndarray, confidence: float, method: str) -> float:
    """One-day VaR (as a negative return) estimated from a window of past returns."""
    alpha = 1 - confidence
    if method == "historical":
        return float(np.quantile(window, alpha))
    mu, sigma = window.mean(), window.std(ddof=1)
    if method == "normal":
        return float(mu + sigma * stats.norm.ppf(alpha))
    if method == "student_t":
        df, loc, scale = stats.t.fit(window)
        return float(stats.t.ppf(alpha, df, loc, scale))
    if method == "ewma":
        # Recent days weigh more, so volatility reacts within days to a shock.
        w = EWMA_LAMBDA ** np.arange(len(window))[::-1]
        var_ewma = np.sum(w * window**2) / w.sum()
        return float(np.sqrt(var_ewma) * stats.norm.ppf(alpha))
    if method == "fhs":
        sigma = ewma_volatility(window)
        shocks = window / sigma
        sigma_next = np.sqrt(EWMA_LAMBDA * sigma[-1] ** 2 + (1 - EWMA_LAMBDA) * window[-1] ** 2)
        return float(np.quantile(shocks, alpha) * sigma_next)
    raise ValueError(f"Unknown method {method!r}, choose from {METHODS}")


def ewma_volatility(x: np.ndarray, lam: float = EWMA_LAMBDA, seed_days: int = 20) -> np.ndarray:
    """EWMA volatility where sigma[t] only uses returns up to t-1."""
    var = np.empty(len(x))
    var[0] = x[:seed_days].var()
    for t in range(1, len(x)):
        var[t] = lam * var[t - 1] + (1 - lam) * x[t - 1] ** 2
    return np.sqrt(var)


def kupiec_pof(n_obs: int, n_exceptions: int, confidence: float) -> tuple[float, float]:
    """Kupiec likelihood-ratio statistic and its p-value (chi-square, 1 dof)."""
    p = 1 - confidence
    x, n = n_exceptions, n_obs
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        phat = x / n
        lr = -2 * ((n - x) * np.log((1 - p) / (1 - phat)) + x * np.log(p / phat))
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def basel_zone(exceptions_per_250: float) -> str:
    if exceptions_per_250 <= 4:
        return "green"
    if exceptions_per_250 <= 9:
        return "yellow"
    return "red"


@dataclass
class BacktestResult:
    method: str
    confidence: float
    n_obs: int
    n_exceptions: int
    lr_stat: float
    p_value: float

    @property
    def exception_rate(self) -> float:
        return self.n_exceptions / self.n_obs

    @property
    def exceptions_per_250(self) -> float:
        return self.exception_rate * 250

    @property
    def zone(self) -> str:
        return basel_zone(self.exceptions_per_250)

    @property
    def accepted(self) -> bool:
        """Kupiec test not rejected at the 5 % level."""
        return self.p_value >= 0.05


def backtest(returns: pd.Series | np.ndarray, confidence: float = 0.99, method: str = "historical",
             window: int = 250) -> tuple[BacktestResult, pd.DataFrame]:
    """Rolling out-of-sample VaR backtest.

    Returns the summary and a frame with the realised return, the VaR
    forecast and an exception flag for every tested day.
    """
    r = pd.Series(np.asarray(returns, dtype=float),
                  index=returns.index if isinstance(returns, pd.Series) else None)
    if len(r) <= window:
        raise ValueError(f"Need more than {window} observations, got {len(r)}")
    values = r.to_numpy()
    forecasts = np.array([var_forecast(values[t - window:t], confidence, method)
                          for t in range(window, len(values))])
    realised = values[window:]
    exceptions = realised < forecasts
    n, x = len(realised), int(exceptions.sum())
    lr, p = kupiec_pof(n, x, confidence)
    detail = pd.DataFrame({"return": realised, "var": forecasts, "exception": exceptions},
                          index=r.index[window:])
    return BacktestResult(method, confidence, n, x, lr, p), detail
