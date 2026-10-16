"""
market_data.py
--------------
Downloads, caches, and manages market price data via yfinance.

Provides:
  - PriceLoader: downloads adjusted close prices, computes log/simple returns
  - Caches data locally as parquet files to avoid redundant API calls
  - Handles missing data, de-listing, partial histories

Usage:
    loader = PriceLoader(tickers=["AAPL", "MSFT", "JPM"], years=5)
    prices = loader.get_prices()        # DataFrame (dates × tickers)
    returns = loader.get_returns()      # Log returns
    info = loader.get_info()            # Sector, market cap, etc.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ─── Price Loader ────────────────────────────────────────────────────────────

class PriceLoader:
    """
    Downloads and manages historical adjusted close prices.

    Args:
        tickers: List of ticker symbols (Yahoo Finance format)
        years: Number of years of history to fetch
        benchmark: Benchmark ticker (default: S&P 500)
        cache_dir: Directory for local data cache
    """

    def __init__(
        self,
        tickers: list[str],
        years: int = 5,
        benchmark: str = "^GSPC",
        cache_dir: str = "data/cache",
    ):
        self.tickers = [t.upper() for t in tickers]
        self.years = years
        self.benchmark = benchmark
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._prices: pd.DataFrame | None = None
        self._returns: pd.DataFrame | None = None

    @property
    def end_date(self) -> str:
        return datetime.today().strftime("%Y-%m-%d")

    @property
    def start_date(self) -> str:
        return (datetime.today() - timedelta(days=365 * self.years + 10)).strftime("%Y-%m-%d")

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}_{self.years}y.parquet"

    def _download_single(self, ticker: str) -> pd.Series:
        """Download or load from cache a single ticker."""
        path = self._cache_path(ticker)
        # Use cache if less than 24h old
        if path.exists():
            age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
            if age_hours < 24:
                df = pd.read_parquet(path)
                return df["Close"].rename(ticker)

        try:
            raw = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                print(f"  ⚠ No data for {ticker}")
                return pd.Series(dtype=float, name=ticker)

            series = raw["Close"].rename(ticker)
            # Cache it
            series.to_frame().to_parquet(path)
            return series

        except Exception as e:
            print(f"  ✗ Failed to download {ticker}: {e}")
            return pd.Series(dtype=float, name=ticker)

    def get_prices(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Returns adjusted close prices as a DataFrame.

        Shape: (trading_days, n_tickers)
        Drops dates with any missing values after forward-filling max 5 days.
        """
        if self._prices is not None and not force_refresh:
            return self._prices

        all_tickers = self.tickers + ([self.benchmark] if self.benchmark else [])
        print(f"\n📥 Loading prices for {len(all_tickers)} tickers ({self.years}y)...")

        series_list = []
        for ticker in all_tickers:
            s = self._download_single(ticker)
            if not s.empty:
                series_list.append(s)
                print(f"  ✓ {ticker}: {len(s)} rows")

        if not series_list:
            raise ValueError("No price data could be loaded.")

        prices = pd.concat(series_list, axis=1)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()

        # Forward fill up to 5 days (weekends, holidays)
        prices = prices.ffill(limit=5)
        # Drop rows still missing after ffill
        prices = prices.dropna()

        self._prices = prices
        print(f"\n✅ Price matrix: {prices.shape} | {prices.index[0].date()} → {prices.index[-1].date()}")
        return self._prices

    def get_returns(self, method: str = "log") -> pd.DataFrame:
        """
        Computes daily returns (excluding benchmark column).

        Args:
            method: 'log' (ln(P_t/P_{t-1})) or 'simple' ((P_t-P_{t-1})/P_{t-1})

        Returns:
            DataFrame of daily returns (dates × tickers), benchmark excluded
        """
        if self._returns is not None:
            return self._returns

        prices = self.get_prices()
        asset_prices = prices[[c for c in prices.columns if c != self.benchmark]]

        if method == "log":
            returns = np.log(asset_prices / asset_prices.shift(1)).dropna()
        elif method == "simple":
            returns = asset_prices.pct_change().dropna()
        else:
            raise ValueError(f"Unknown return method: {method}")

        self._returns = returns
        return self._returns

    def get_benchmark_returns(self) -> pd.Series:
        """Returns benchmark log returns as a Series."""
        prices = self.get_prices()
        if self.benchmark not in prices.columns:
            raise ValueError(f"Benchmark {self.benchmark} not in prices")
        bench = prices[self.benchmark]
        return np.log(bench / bench.shift(1)).dropna().rename(self.benchmark)

    def get_info(self) -> pd.DataFrame:
        """
        Retrieves fundamental info for each ticker.
        Returns DataFrame with columns: sector, industry, marketCap, beta.
        """
        records = []
        for ticker in self.tickers:
            try:
                info = yf.Ticker(ticker).info
                records.append({
                    "ticker": ticker,
                    "name":       info.get("shortName", ticker),
                    "sector":     info.get("sector", "N/A"),
                    "industry":   info.get("industry", "N/A"),
                    "market_cap": info.get("marketCap", np.nan),
                    "beta":       info.get("beta", np.nan),
                    "pe_ratio":   info.get("trailingPE", np.nan),
                })
            except Exception:
                records.append({"ticker": ticker, "name": ticker})
        return pd.DataFrame(records).set_index("ticker")

    def summary(self) -> None:
        """Print a clean summary of the loaded dataset."""
        returns = self.get_returns()
        ann = returns.mean() * 252
        vol = returns.std() * np.sqrt(252)
        print(f"\n{'─'*55}")
        print(f"  Portfolio Summary ({self.years}y daily data)")
        print(f"{'─'*55}")
        print(f"  {'Ticker':<10} {'Ann.Return':>11} {'Ann.Volatility':>15} {'Skew':>8} {'Kurt':>8}")
        print(f"  {'─'*10} {'─'*11} {'─'*15} {'─'*8} {'─'*8}")
        for ticker in returns.columns:
            r = returns[ticker]
            print(
                f"  {ticker:<10} {ann[ticker]*100:>10.2f}% {vol[ticker]*100:>14.2f}%"
                f" {r.skew():>8.3f} {r.kurtosis():>8.3f}"
            )
        print(f"{'─'*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download market data")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "JPM", "GS"])
    parser.add_argument("--years", type=int, default=5)
    args = parser.parse_args()

    loader = PriceLoader(tickers=args.tickers, years=args.years)
    loader.summary()
