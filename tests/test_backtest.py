"""Tests for VaR backtesting (no network needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.backtest import (
    backtest,
    basel_zone,
    ewma_volatility,
    kupiec_pof,
    var_forecast,
)


def test_kupiec_accepts_expected_rate():
    lr, p = kupiec_pof(n_obs=1000, n_exceptions=10, confidence=0.99)
    assert lr == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0)


def test_kupiec_rejects_too_many_exceptions():
    _, p = kupiec_pof(n_obs=1000, n_exceptions=30, confidence=0.99)
    assert p < 0.01


def test_kupiec_handles_zero_exceptions():
    lr, p = kupiec_pof(n_obs=1000, n_exceptions=0, confidence=0.99)
    assert lr > 0 and 0 <= p < 0.05


@pytest.mark.parametrize("n,zone", [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")])
def test_basel_traffic_light(n, zone):
    assert basel_zone(n) == zone


def test_var_forecast_matches_normal_quantile():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.01, 100_000)
    assert var_forecast(w, 0.99, "normal") == pytest.approx(-0.02326, abs=2e-4)
    assert var_forecast(w, 0.99, "historical") == pytest.approx(-0.02326, abs=5e-4)


def test_correct_model_passes_on_its_own_distribution():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 2250))
    res, detail = backtest(r, 0.99, "normal", window=250)
    assert res.n_obs == 2000 == len(detail)
    assert res.accepted
    assert res.zone == "green"


def test_normal_var_fails_on_fat_tails():
    """Student-t(3) returns break a normal VaR at 99 %: too many exceptions."""
    rng = np.random.default_rng(2)
    r = pd.Series(rng.standard_t(3, 5250) * 0.01)
    normal, _ = backtest(r, 0.99, "normal", window=250)
    assert normal.exception_rate > 0.01
    assert not normal.accepted


def test_no_look_ahead():
    """A shock on the last day cannot influence any forecast."""
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, 400)
    shocked = base.copy()
    shocked[-1] = -0.5
    _, d1 = backtest(pd.Series(base), 0.99, "historical", window=250)
    _, d2 = backtest(pd.Series(shocked), 0.99, "historical", window=250)
    assert np.allclose(d1["var"], d2["var"])
    assert d2["exception"].iloc[-1]


def test_ewma_reacts_faster_than_equal_weights_after_a_volatility_jump():
    rng = np.random.default_rng(4)
    calm, stressed = rng.normal(0, 0.005, 240), rng.normal(0, 0.03, 10)
    window = np.concatenate([calm, stressed])
    assert var_forecast(window, 0.99, "ewma") < var_forecast(window, 0.99, "normal")


def test_ewma_volatility_uses_only_past_returns():
    x = np.zeros(50)
    x[-1] = 1.0
    assert ewma_volatility(x)[-1] == 0.0


def test_fhs_passes_on_volatility_clustering_where_normal_fails():
    """GARCH(1,1)-like returns with t shocks: FHS adapts, a static normal VaR does not."""
    rng = np.random.default_rng(5)
    n, r, var = 3250, np.zeros(3250), 1e-4
    for t in range(n):
        r[t] = np.sqrt(var) * rng.standard_t(5) / np.sqrt(5 / 3)
        var = 2e-6 + 0.10 * r[t] ** 2 + 0.88 * var
    fhs, _ = backtest(pd.Series(r), 0.99, "fhs")
    normal, _ = backtest(pd.Series(r), 0.99, "normal")
    assert fhs.accepted
    assert abs(fhs.exception_rate - 0.01) < abs(normal.exception_rate - 0.01)
