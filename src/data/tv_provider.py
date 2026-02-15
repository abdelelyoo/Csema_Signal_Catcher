"""
Data Provider - TradingView implementation with caching
"""

import logging
import time
import random
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """Simple in-memory cache for market data"""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, datetime]] = {}
        self._ttl = ttl_seconds

    def get(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        key = (symbol, interval)
        if key in self._cache:
            df, timestamp = self._cache[key]
            if (datetime.now() - timestamp).total_seconds() < self._ttl:
                return df.copy()
            else:
                del self._cache[key]
        return None

    def set(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        key = (symbol, interval)
        self._cache[key] = (df.copy(), datetime.now())

    def clear(self) -> None:
        self._cache.clear()


class TradingViewProvider:
    """Implementation using tvdatafeed"""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_cache: bool = True,
    ) -> None:
        self.tv = None
        self._username = username
        self._password = password
        self._initialize_tv(username, password)
        self.exchange = "CSEMA"
        self._cache = DataCache(ttl_seconds=300) if use_cache else None

    def _initialize_tv(self, username: Optional[str], password: Optional[str]) -> None:
        try:
            from tvDatafeed import TvDatafeed

            self.tv = TvDatafeed(username, password)
            logger.info("TvDatafeed initialized successfully")
        except ImportError:
            logger.warning("tvdatafeed library not installed")
        except Exception as e:
            logger.warning(f"Failed to initialize TvDatafeed: {e}")

    def _fetch_with_retries(
        self,
        fetch_func,
        symbol: str,
        interval,
        n_bars: int,
        max_retries: int = 5,
    ) -> Optional[pd.DataFrame]:
        """Fetch data with retries"""
        interval_str = str(interval)

        if self._cache:
            cached = self._cache.get(symbol, interval_str)
            if cached is not None:
                logger.debug(f"Cache hit for {symbol} {interval_str}")
                return cached

        for i in range(max_retries):
            try:
                df = fetch_func(
                    symbol=symbol,
                    exchange=self.exchange,
                    interval=interval,
                    n_bars=n_bars,
                )
                if df is not None and not df.empty:
                    if self._cache:
                        self._cache.set(symbol, interval_str, df)
                    return df
            except Exception as e:
                err_msg = str(e).lower()
                is_conn_error = any(
                    x in err_msg
                    for x in ["10054", "closed", "lost", "reset", "timeout"]
                )

                if is_conn_error:
                    self._initialize_tv(self._username, self._password)
                    time.sleep(2)

                wait_time = (2 ** (i + 1)) + (random.randint(0, 1000) / 1000)
                logger.warning(
                    f"Retry {i + 1}/{max_retries} for {symbol} after {wait_time:.2f}s"
                )
                time.sleep(wait_time)

        return None

    def get_daily_data(self, symbol: str, n_bars: int = 500) -> pd.DataFrame:
        if not self.tv:
            return pd.DataFrame()

        try:
            from tvDatafeed import Interval

            logger.info(f"Fetching daily data for {symbol} from {self.exchange}")
            df = self._fetch_with_retries(
                self.tv.get_hist,
                symbol=symbol,
                interval=Interval.in_daily,
                n_bars=n_bars,
            )

            if df is not None and not df.empty:
                df.columns = [col.capitalize() for col in df.columns]
                logger.info(f"Got {len(df)} bars for {symbol}")
                return df
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in get_daily_data for {symbol}: {e}")
            return pd.DataFrame()

    def get_weekly_data(self, symbol: str, n_bars: int = 200) -> pd.DataFrame:
        if not self.tv:
            return pd.DataFrame()

        try:
            from tvDatafeed import Interval

            logger.info(f"Fetching weekly data for {symbol} from {self.exchange}")
            df = self._fetch_with_retries(
                self.tv.get_hist,
                symbol=symbol,
                interval=Interval.in_weekly,
                n_bars=n_bars,
            )

            if df is not None and not df.empty:
                df.columns = [col.capitalize() for col in df.columns]
                logger.info(f"Got {len(df)} bars for {symbol}")
                return df
            logger.warning(f"No weekly data for {symbol}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in get_weekly_data for {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Optional[float]:
        df = self.get_daily_data(symbol, n_bars=1)
        if not df.empty:
            return float(df["Close"].iloc[-1])
        return None

    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        prices: Dict[str, float] = {}
        for symbol in symbols:
            clean_ticker = symbol.split(":")[-1].split(".")[0]
            price = self.get_current_price(clean_ticker)
            if price is not None:
                prices[symbol] = price
        return prices

    def clear_cache(self) -> None:
        """Clear the data cache"""
        if self._cache:
            self._cache.clear()
            logger.info("Data cache cleared")
