"""
Stage 1: Market Screening Module for CSEMA
Identifies trading candidates using tvscreener library
"""

import logging
import pandas as pd
from typing import Dict, Optional
from pathlib import Path

try:
    import tvscreener as tvs

    TVSCREENER_AVAILABLE = True
except ImportError:
    TVSCREENER_AVAILABLE = False
    tvs = None

logger = logging.getLogger(__name__)


class CSEMAScreenerBase:
    """Base class for CSEMA screeners to avoid code duplication"""

    def __init__(self) -> None:
        if not TVSCREENER_AVAILABLE:
            raise ImportError(
                "tvscreener library is not installed. "
                "Install it with: pip install tvscreener"
            )
        self.screener = tvs.StockScreener()
        self._setup_morocco_market()

    def _setup_morocco_market(self) -> None:
        """Configure screener for Morocco/CSEMA market"""
        try:
            self.screener.set_markets(tvs.Market.MOROCCO)
        except Exception:
            logger.warning("Morocco market not directly available")


class CSEMAPositionScreener(CSEMAScreenerBase):
    """Position trading screener for Moroccan stocks"""

    def get_all_tickers(self, max_results: int = 200) -> pd.DataFrame:
        """Get ALL available CSEMA tickers without filters"""
        logger.info("Fetching ALL CSEMA tickers...")

        ss = self.screener

        ss.select(
            tvs.StockField.NAME,
            tvs.StockField.EXCHANGE,
            tvs.StockField.PRICE,
            tvs.StockField.CHANGE_PERCENT,
            tvs.StockField.VOLUME,
            tvs.StockField.AVERAGE_VOLUME_30_DAY,
            tvs.StockField.MARKET_CAPITALIZATION,
            tvs.StockField.PRICE_TO_EARNINGS_RATIO_TTM,
            tvs.StockField.DEBT_TO_EQUITY_FQ,
            tvs.StockField.RETURN_ON_EQUITY_FQ,
            tvs.StockField.RELATIVE_STRENGTH_INDEX_14,
            tvs.StockField.SMA200_1,
            tvs.StockField.SMA50_1,
            tvs.StockField.SECTOR,
        )

        try:
            ss.set_range(0, max_results)
            df = ss.get()
            if df is None:
                logger.warning("get_all_tickers returned None")
                return pd.DataFrame()
            logger.info(f"Found {len(df)} total tickers")
            return df
        except Exception as e:
            logger.error(f"Error fetching all tickers: {e}")
            return pd.DataFrame()

    def screen_position_candidates(self, max_results: int = 50) -> pd.DataFrame:
        """Screen for position trading candidates"""
        logger.info("Running Position Trading Screener for CSEMA...")

        ss = self.screener

        ss.select(
            tvs.StockField.NAME,
            tvs.StockField.EXCHANGE,
            tvs.StockField.PRICE,
            tvs.StockField.CHANGE_PERCENT,
            tvs.StockField.VOLUME,
            tvs.StockField.AVERAGE_VOLUME_30_DAY,
            tvs.StockField.MARKET_CAPITALIZATION,
            tvs.StockField.PRICE_TO_EARNINGS_RATIO_TTM,
            tvs.StockField.DEBT_TO_EQUITY_FQ,
            tvs.StockField.RETURN_ON_EQUITY_FQ,
            tvs.StockField.RELATIVE_STRENGTH_INDEX_14,
            tvs.StockField.SMA200_1,
            tvs.StockField.SECTOR,
        )

        # Basic filters
        ss.where(tvs.StockField.MARKET_CAPITALIZATION > 1e8)
        ss.where(tvs.StockField.PRICE_TO_EARNINGS_RATIO_TTM.between(10, 25))
        ss.where(tvs.StockField.DEBT_TO_EQUITY_FQ < 1.5)
        ss.where(tvs.StockField.RETURN_ON_EQUITY_FQ > 10.0)
        # Note: Minimum 100k MAD turnover filter applied after fetching data
        # Turnover = Price × Average Volume (30d)

        try:
            ss.set_range(0, max_results)
            df = ss.get()

            # Handle None return from screener
            if df is None:
                logger.warning("Position screener returned None")
                return pd.DataFrame()

            # Apply turnover filter: Price × Average Volume (30d) >= 100,000 MAD
            if (
                not df.empty
                and "Price" in df.columns
                and "Average Volume 30d" in df.columns
            ):
                df["Turnover_MAD"] = df["Price"] * df["Average Volume 30d"]
                df = df[df["Turnover_MAD"] >= 100000]
                logger.info(
                    f"Found {len(df)} position trading candidates (turnover >= 100k MAD)"
                )
            else:
                logger.info(
                    f"Found {len(df)} position trading candidates (no turnover filter applied)"
                )

            return df

        except Exception as e:
            logger.error(f"Error in position screening: {e}")
            return pd.DataFrame()


class CSEMASwingScreener(CSEMAScreenerBase):
    """Swing trading screener for Moroccan stocks"""

    def get_all_tickers(self, max_results: int = 200) -> pd.DataFrame:
        """Get ALL available CSEMA tickers without filters"""
        logger.info("Fetching ALL CSEMA tickers for swing...")

        ss = self.screener

        ss.select(
            tvs.StockField.NAME,
            tvs.StockField.EXCHANGE,
            tvs.StockField.PRICE,
            tvs.StockField.CHANGE_PERCENT,
            tvs.StockField.VOLUME,
            tvs.StockField.AVERAGE_VOLUME_30_DAY,
            tvs.StockField.RELATIVE_STRENGTH_INDEX_14,
            tvs.StockField.MACD_LEVEL_12_26,
            tvs.StockField.MACD_SIGNAL_12_26,
            tvs.StockField.SMA50_1,
            tvs.StockField.SMA200_1,
            tvs.StockField.WEEK_HIGH_52,
            tvs.StockField.WEEK_LOW_52,
            tvs.StockField.AVERAGE_TRUE_RANGE_14,
            tvs.StockField.SECTOR,
        )

        try:
            ss.set_range(0, max_results)
            df = ss.get()
            if df is None:
                logger.warning("get_all_tickers returned None for swing")
                return pd.DataFrame()
            logger.info(f"Found {len(df)} total tickers for swing")
            return df
        except Exception as e:
            logger.error(f"Error fetching all tickers: {e}")
            return pd.DataFrame()

    def screen_swing_candidates(self, max_results: int = 50) -> pd.DataFrame:
        """Screen for swing trading candidates"""
        logger.info("Running Swing Trading Screener for CSEMA...")

        ss = self.screener

        ss.select(
            tvs.StockField.NAME,
            tvs.StockField.EXCHANGE,
            tvs.StockField.PRICE,
            tvs.StockField.CHANGE_PERCENT,
            tvs.StockField.VOLUME,
            tvs.StockField.AVERAGE_VOLUME_30_DAY,
            tvs.StockField.RELATIVE_STRENGTH_INDEX_14,
            tvs.StockField.MACD_LEVEL_12_26,
            tvs.StockField.MACD_SIGNAL_12_26,
            tvs.StockField.SMA50_1,
            tvs.StockField.WEEK_HIGH_52,
            tvs.StockField.WEEK_LOW_52,
            tvs.StockField.AVERAGE_TRUE_RANGE_14,
            tvs.StockField.SECTOR,
        )

        ss.where(tvs.StockField.RELATIVE_STRENGTH_INDEX_14.between(50, 70))
        # Note: Minimum 100k MAD turnover filter applied after fetching data

        try:
            ss.set_range(0, max_results)
            df = ss.get()

            # Handle None return from screener
            if df is None:
                logger.warning("Swing screener returned None")
                return pd.DataFrame()

            # Apply turnover filter: Price × Average Volume (30d) >= 100,000 MAD
            if (
                not df.empty
                and "Price" in df.columns
                and "Average Volume 30d" in df.columns
            ):
                df["Turnover_MAD"] = df["Price"] * df["Average Volume 30d"]
                df = df[df["Turnover_MAD"] >= 100000]

                # Also apply volume surge filter if volume data available
                if "Volume" in df.columns:
                    df["Volume_Ratio"] = df["Volume"] / df["Average Volume 30d"]
                    df = df[df["Volume_Ratio"] >= 1.5]

                logger.info(
                    f"Found {len(df)} swing trading candidates (turnover >= 100k MAD)"
                )
            else:
                logger.info(
                    f"Found {len(df)} swing trading candidates (no turnover filter applied)"
                )

            return df

        except Exception as e:
            logger.error(f"Error in swing screening: {e}")
            return pd.DataFrame()


class CSEMAWatchlistManager:
    """Manages watchlists for both position and swing trading"""

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self.data_dir = data_dir or "./data"
        self.position_screener = CSEMAPositionScreener()
        self.swing_screener = CSEMASwingScreener()

    def generate_full_watchlist(self) -> Dict[str, pd.DataFrame]:
        """Generate watchlist with ALL available tickers (no filters)"""
        logger.info("=" * 60)
        logger.info("GENERATING FULL CSEMA WATCHLIST (ALL TICKERS)")
        logger.info("=" * 60)

        position_df = self.position_screener.get_all_tickers()
        swing_df = self.swing_screener.get_all_tickers()

        result = {
            "position": position_df,
            "swing": swing_df,
            "timestamp": pd.Timestamp.now(),
        }

        total_tickers = len(position_df) + len(swing_df)
        logger.info(f"\n" + "=" * 60)
        logger.info("FULL WATCHLIST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Position Tickers: {len(position_df)}")
        logger.info(f"Total Swing Tickers: {len(swing_df)}")
        logger.info(f"Total Tickers: {total_tickers}")

        return result

    def generate_watchlist(self) -> Dict[str, pd.DataFrame]:
        """Generate complete watchlist with both position and swing candidates"""
        logger.info("=" * 60)
        logger.info("GENERATING CSEMA TRADING WATCHLIST")
        logger.info("=" * 60)

        position_df = self.position_screener.screen_position_candidates()
        swing_df = self.swing_screener.screen_swing_candidates()

        result = {
            "position": position_df,
            "swing": swing_df,
            "timestamp": pd.Timestamp.now(),
        }

        logger.info("\n" + "=" * 60)
        logger.info("WATCHLIST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Position Trading Candidates: {len(position_df)}")
        logger.info(f"Swing Trading Candidates: {len(swing_df)}")

        return result

    def save_watchlist(
        self, watchlist: Dict[str, pd.DataFrame], filename: Optional[str] = None
    ) -> None:
        """Save watchlist to CSV files"""
        if filename is None:
            filename = f"watchlist_{pd.Timestamp.now().strftime('%Y%m%d')}"

        position_path = Path(self.data_dir) / f"{filename}_position.csv"
        swing_path = Path(self.data_dir) / f"{filename}_swing.csv"

        if not watchlist["position"].empty:
            watchlist["position"].to_csv(position_path, index=False)
            logger.info(f"Saved position watchlist to {position_path}")

        if not watchlist["swing"].empty:
            watchlist["swing"].to_csv(swing_path, index=False)
            logger.info(f"Saved swing watchlist to {swing_path}")

    def load_watchlist(self, date_str: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Load watchlist from CSV files"""
        if date_str is None:
            date_str = pd.Timestamp.now().strftime("%Y%m%d")

        position_path = Path(self.data_dir) / f"watchlist_{date_str}_position.csv"
        swing_path = Path(self.data_dir) / f"watchlist_{date_str}_swing.csv"

        result: Dict[str, pd.DataFrame] = {
            "position": pd.DataFrame(),
            "swing": pd.DataFrame(),
        }

        try:
            result["position"] = pd.read_csv(position_path)
            logger.info(f"Loaded position watchlist from {position_path}")
        except FileNotFoundError:
            logger.warning(f"Position watchlist not found for {date_str}")

        try:
            result["swing"] = pd.read_csv(swing_path)
            logger.info(f"Loaded swing watchlist from {swing_path}")
        except FileNotFoundError:
            logger.warning(f"Swing watchlist not found for {date_str}")

        return result


def run_stage1_screening(
    save_results: bool = True,
    data_dir: Optional[str] = None,
    full_watchlist: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Main function to run Stage 1 screening"""
    manager = CSEMAWatchlistManager(data_dir=data_dir)

    if full_watchlist:
        watchlist = manager.generate_full_watchlist()
    else:
        watchlist = manager.generate_watchlist()

    if save_results:
        manager.save_watchlist(watchlist)

    return watchlist
