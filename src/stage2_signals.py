"""
Stage 2: Entry Signal Generation with Risk Management
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Constants
POSITION_RISK_DEFAULT = 0.01
SWING_RISK_DEFAULT = 0.015
ATR_MULTIPLIER_POSITION = 2.0
ATR_MULTIPLIER_SWING = 1.5
POSITION_CRITERIA_MIN = 2
SWING_CRITERIA_MIN = 3
RSI_MIN_SWING = 50
RSI_MAX_SWING = 65
RSI_MAX_POSITION_WEEKLY = 70
RSI_MIN_POSITION_WEEKLY = 40
MOROCCAN_MIN_PROFIT_PERCENT = 3.0

ENTRY_BUFFER = 1.005
STOP_BUFFER_ATR = 0.99
STOP_BUFFER_BREAKOUT = 0.985
BREAKOUT_TOLERANCE = 0.995
RISK_REWARD_POSITION = 2.5
RISK_REWARD_POSITION_2 = 3.5
RISK_REWARD_SWING = 2.0
RISK_REWARD_ENTRY_HIGH = 2.5
RISK_REWARD_ENTRY_MEDIUM = 2.0
RISK_REWARD_ENTRY_LOW = 1.5
VOLUME_RATIO_MIN = 1.5
RSI_FILTER_WEEKLY_UPPER = 70
SMA_PULLBACK_TOLERANCE = 1.02
MIN_DATA_BARS = 50
MIN_WEEKLY_DATA_BARS = 20

# New Indicator Constants
STOCH_OVERBOUGHT = 80
STOCH_OVERSOLD = 20
STOCH_CROSS_BUY = 1.0
ADX_STRONG_TREND = 25
MFI_OVERBOUGHT = 80
MFI_OVERSOLD = 20


@dataclass
class TradeSignal:
    """Represents a trade signal with entry, stop, and target levels"""

    symbol: str
    setup_type: str
    current_price: float
    entry_price: float
    stop_loss: float
    target_price: float
    target_price_2: Optional[float] = None
    position_size: int = 0
    risk_amount: float = 0.0
    risk_reward: float = 0.0
    conviction: str = "medium"
    atr: float = 0.0
    rsi: float = 0.0
    volume_ratio: float = 0.0
    setup_notes: str = ""
    timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


def safe_compare(a: Any, op: str, b: Any) -> bool:
    """Safely compare values that might be None or NaN"""
    if pd.isna(a) or pd.isna(b):
        return False
    if op == "<":
        return bool(a < b)
    elif op == "<=":
        return bool(a <= b)
    elif op == ">":
        return bool(a > b)
    elif op == ">=":
        return bool(a >= b)
    elif op == "==":
        return bool(a == b)
    return False


def validate_price_data(df: Optional[pd.DataFrame], min_bars: int = 50) -> bool:
    """Validate that dataframe has sufficient valid price data"""
    if df is None or df.empty:
        return False
    if len(df) < min_bars:
        return False

    cols_lower = [c.lower() for c in df.columns]
    required_cols_lower = ["close", "open", "high", "low", "volume"]
    if not all(col in cols_lower for col in required_cols_lower):
        return False

    close_col = df.columns[df.columns.str.lower() == "close"][0]
    if df[close_col].isna().all():
        return False
    return True


def safe_get_prev_candle(df: pd.DataFrame) -> Optional[pd.Series]:
    """Safely get previous candle, returns last candle if only one available"""
    if df is None or df.empty:
        return None
    if len(df) < 2:
        return df.iloc[-1] if len(df) == 1 else None
    return df.iloc[-2]


def clean_symbol(symbol: str) -> str:
    """Clean symbol by removing exchange prefix"""
    if ":" in symbol:
        return symbol.split(":")[-1]
    return symbol


class IndicatorCalculator:
    """Calculates technical indicators"""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required technical indicators"""
        if df.empty:
            return df

        df = df.copy()
        df.columns = df.columns.str.lower()

        # Moving Averages
        df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
        df["sma_50"] = df["close"].rolling(window=50, min_periods=1).mean()
        df["sma_200"] = df["close"].rolling(window=200, min_periods=1).mean()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["close"].ewm(span=12, min_periods=1).mean()
        ema_26 = df["close"].ewm(span=26, min_periods=1).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, min_periods=1).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = true_range.rolling(window=14, min_periods=1).mean()

        # Bollinger Bands
        sma = df["close"].rolling(window=20, min_periods=1).mean()
        std = df["close"].rolling(window=20, min_periods=1).std()
        df["bb_lower"] = sma - (std * 2)
        df["bb_middle"] = sma
        df["bb_upper"] = sma + (std * 2)

        # Volume
        df["volume_sma_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)

        # Support/Resistance
        df["swing_low"] = df["low"].rolling(window=5, center=True, min_periods=1).min()
        df["swing_high"] = (
            df["high"].rolling(window=5, center=True, min_periods=1).max()
        )

        # Stochastic Oscillator
        low14 = df["low"].rolling(window=14, min_periods=1).min()
        high14 = df["high"].rolling(window=14, min_periods=1).max()
        stoch_range = high14 - low14
        df["stoch_k"] = 100 * (df["close"] - low14) / stoch_range.replace(0, np.nan)
        df["stoch_d"] = df["stoch_k"].rolling(window=3, min_periods=1).mean()

        # ADX (Average Directional Index)
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        atr = df["atr_14"].replace(0, np.nan)
        plus_di = 100 * (plus_dm.rolling(window=14, min_periods=1).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14, min_periods=1).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        df["adx"] = dx.rolling(window=14, min_periods=1).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        # MFI (Money Flow Index)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        positive_mf = positive_flow.rolling(window=14, min_periods=1).sum()
        negative_mf = negative_flow.rolling(window=14, min_periods=1).sum()
        mfi_ratio = positive_mf / negative_mf.replace(0, np.nan)
        df["mfi"] = 100 - (100 / (1 + mfi_ratio))

        # Pivot Points (Support/Resistance)
        df["pivot"] = (
            df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)
        ) / 3
        df["s1"] = 2 * df["pivot"] - df["high"].shift(1)
        df["s2"] = df["pivot"] - (df["high"].shift(1) - df["low"].shift(1))
        df["r1"] = 2 * df["pivot"] - df["low"].shift(1)
        df["r2"] = df["pivot"] + (df["high"].shift(1) - df["low"].shift(1))

        # VWAP (Volume Weighted Average Price)
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

        return df

    @staticmethod
    def calculate_weekly_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for weekly timeframe"""
        if df.empty:
            return df

        df = df.copy()
        df.columns = df.columns.str.lower()

        df["sma_10"] = df["close"].rolling(window=10, min_periods=1).mean()
        df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Stochastic
        low14 = df["low"].rolling(window=14, min_periods=1).min()
        high14 = df["high"].rolling(window=14, min_periods=1).max()
        stoch_range = high14 - low14
        df["stoch_k"] = 100 * (df["close"] - low14) / stoch_range.replace(0, np.nan)
        df["stoch_d"] = df["stoch_k"].rolling(window=3, min_periods=1).mean()

        # ADX
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = true_range.rolling(window=14, min_periods=1).mean()

        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        atr = df["atr_14"].replace(0, np.nan)
        plus_di = 100 * (plus_dm.rolling(window=14, min_periods=1).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14, min_periods=1).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        df["adx"] = dx.rolling(window=14, min_periods=1).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        return df


class PositionTradingSignals:
    """Generate entry signals for position trading"""

    name = "Position Trading Strategy"

    def __init__(self, data_feed: Any) -> None:
        self.data_feed = data_feed
        self.indicator_calc = IndicatorCalculator()

    def _check_weekly_trend(self, weekly_last: pd.Series, current_price: float) -> bool:
        """Check if weekly trend is bullish"""
        return bool(current_price > weekly_last.get("sma_20", 0))

    def _check_weekly_rsi(self, weekly_last: pd.Series) -> bool:
        """Check if weekly RSI is in acceptable range"""
        weekly_rsi = weekly_last.get("rsi_14", np.nan)
        return bool(
            not pd.isna(weekly_rsi)
            and RSI_MIN_POSITION_WEEKLY <= weekly_rsi <= RSI_MAX_POSITION_WEEKLY
        )

    def _check_entry_criteria(
        self, last: pd.Series, prev: pd.Series
    ) -> tuple[int, Dict[str, bool]]:
        """Check entry criteria and return count + details"""
        criteria = {}

        # Pullback to SMA20
        criteria["pullback_to_sma20"] = bool(
            last.get("low", 0) <= last.get("sma_20", np.inf) <= last.get("high", 0)
            or last.get("close", 0) < last.get("sma_20", 0) * SMA_PULLBACK_TOLERANCE
        )

        # Volume contracting
        criteria["volume_contracting"] = safe_compare(
            prev.get("volume"), "<", prev.get("volume_sma_20")
        )

        # MACD turning positive
        criteria["macd_turning_positive"] = safe_compare(
            last.get("macd_hist"), ">", prev.get("macd_hist")
        ) and safe_compare(last.get("macd_hist"), ">", 0)

        # Bullish engulfing
        criteria["bullish_engulfing"] = (
            safe_compare(prev.get("close"), "<", prev.get("open"))
            and safe_compare(last.get("close"), ">", last.get("open"))
            and safe_compare(last.get("close"), ">", prev.get("open"))
            and safe_compare(last.get("open"), "<", prev.get("close"))
        )

        # Hammer pattern
        open_price = last.get("open", 0)
        close_price = last.get("close", 0)
        low_price = last.get("low", 0)
        criteria["hammer"] = bool(
            close_price > open_price
            and (open_price - low_price) > 2 * (close_price - open_price)
        )

        # New indicators
        criteria["stoch_oversold"] = bool(last.get("stoch_k", 100) < STOCH_OVERSOLD)
        prev_stoch_k = prev.get("stoch_k", 0) if prev is not None else 0
        prev_stoch_d = prev.get("stoch_d", 0) if prev is not None else 0
        criteria["stoch_cross_up"] = bool(
            last.get("stoch_k", 0) > last.get("stoch_d", 0)
            and prev_stoch_k <= prev_stoch_d
        )

        criteria["adx_strong"] = bool(
            last.get("adx", 0) > ADX_STRONG_TREND
            if not pd.isna(last.get("adx"))
            else False
        )

        criteria["mfi_oversold"] = bool(
            last.get("mfi", 100) < MFI_OVERSOLD
            if not pd.isna(last.get("mfi"))
            else False
        )

        # Count criteria met
        criteria_met = sum(
            [
                criteria["pullback_to_sma20"],
                criteria["volume_contracting"],
                criteria["macd_turning_positive"],
                criteria["bullish_engulfing"] or criteria["hammer"],
                criteria["stoch_oversold"] or criteria["stoch_cross_up"],
                criteria["mfi_oversold"],
                criteria["adx_strong"],
            ]
        )

        return criteria_met, criteria

    def _analyze_symbol_impl(
        self,
        symbol: str,
        daily_df: pd.DataFrame,
        weekly_df: pd.DataFrame,
        cash_available: float,
        risk_per_trade: float,
    ) -> Optional[TradeSignal]:
        """Internal implementation of position analysis"""
        if not validate_price_data(daily_df, MIN_DATA_BARS):
            logger.debug(f"{symbol}: Insufficient daily data")
            return None
        if not validate_price_data(weekly_df, MIN_WEEKLY_DATA_BARS):
            logger.debug(f"{symbol}: Insufficient weekly data")
            return None

        daily_df = self.indicator_calc.calculate_indicators(daily_df)
        weekly_df = self.indicator_calc.calculate_weekly_indicators(weekly_df)

        last = daily_df.iloc[-1]
        prev = safe_get_prev_candle(daily_df)
        weekly_last = weekly_df.iloc[-1]

        current_price = float(last.get("close", 0))
        if current_price <= 0:
            return None

        # Weekly trend check
        if not self._check_weekly_trend(weekly_last, current_price):
            logger.debug(
                f"{symbol}: Failed weekly trend ({current_price} <= {weekly_last.get('sma_20', 0)})"
            )
            return None

        # Weekly RSI check
        if not self._check_weekly_rsi(weekly_last):
            weekly_rsi = weekly_last.get("rsi_14", np.nan)
            logger.debug(f"{symbol}: Failed weekly RSI ({weekly_rsi})")
            return None

        atr = last.get("atr_14", np.nan)
        if pd.isna(atr) or atr <= 0:
            logger.debug(f"{symbol}: Invalid ATR ({atr})")
            return None

        # Check entry criteria
        criteria_met, criteria = self._check_entry_criteria(last, prev)

        if criteria_met < POSITION_CRITERIA_MIN:
            adx_val = f"{last.get('adx', np.nan):.1f}"
            mfi_val = f"{last.get('mfi', np.nan):.1f}"
            logger.info(
                f"{symbol}: Only {criteria_met}/{POSITION_CRITERIA_MIN} criteria | "
                f"stoch={last.get('stoch_k', np.nan):.1f}, adx={adx_val}, mfi={mfi_val}"
            )
            return None

        # Entry execution
        entry_price = float(prev.get("high", current_price)) * ENTRY_BUFFER
        stop_atr = current_price - (ATR_MULTIPLIER_POSITION * atr)

        recent_lows = daily_df["swing_low"].dropna().tail(10)
        recent_swing_low = (
            recent_lows.min()
            if not recent_lows.empty
            else daily_df["low"].tail(20).min()
        )
        stop_swing = float(recent_swing_low) * STOP_BUFFER_ATR
        stop_loss = max(stop_atr, stop_swing)

        risk = entry_price - stop_loss
        if risk <= 0:
            return None

        target_price = entry_price + (RISK_REWARD_POSITION * risk)
        target_price_2 = entry_price + (RISK_REWARD_POSITION_2 * risk)

        profit_percent = ((target_price - entry_price) / entry_price) * 100
        if profit_percent < MOROCCAN_MIN_PROFIT_PERCENT:
            target_price = entry_price * (1 + MOROCCAN_MIN_PROFIT_PERCENT / 100)

        risk_reward = (target_price - entry_price) / risk

        if criteria_met >= 3 and risk_reward >= RISK_REWARD_ENTRY_HIGH:
            conviction = "high"
        elif criteria_met >= 2 and risk_reward >= RISK_REWARD_ENTRY_MEDIUM:
            conviction = "medium"
        else:
            conviction = "low"

        risk_amount = cash_available * risk_per_trade
        position_size = max(1, int(risk_amount / risk))

        weekly_rsi = weekly_last.get("rsi_14", np.nan)
        notes = (
            f"Weekly RSI: {weekly_rsi:.1f}, Daily RSI: {last.get('rsi_14', np.nan):.1f}, "
            f"Pullback: {criteria['pullback_to_sma20']}, Volume OK: {criteria['volume_contracting']}"
        )

        return TradeSignal(
            symbol=symbol,
            setup_type="position",
            current_price=current_price,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            target_price=round(target_price, 2),
            target_price_2=round(target_price_2, 2),
            position_size=position_size,
            risk_amount=round(risk_amount, 2),
            risk_reward=round(risk_reward, 2),
            conviction=conviction,
            atr=round(atr, 2),
            rsi=round(last.get("rsi_14", 0), 1),
            volume_ratio=round(last.get("volume_ratio", 0), 2),
            setup_notes=notes,
        )

    def analyze(
        self,
        symbol: str,
        daily_df: pd.DataFrame,
        weekly_df: pd.DataFrame,
        cash_available: float = 10000,
        risk_per_trade: float = POSITION_RISK_DEFAULT,
    ) -> Optional[TradeSignal]:
        """Analyze symbol for position trading setup (for backtesting)"""
        return self._analyze_symbol_impl(
            symbol, daily_df, weekly_df, cash_available, risk_per_trade
        )

    def generate_signals(
        self,
        symbols: List[str],
        cash_available: float = 10000,
        risk_per_trade: float = POSITION_RISK_DEFAULT,
    ) -> List[TradeSignal]:
        """Generate position trading entry signals"""
        if cash_available <= 0:
            logger.error("cash_available must be positive")
            return []
        if risk_per_trade <= 0 or risk_per_trade > 1:
            logger.error("risk_per_trade must be between 0 and 1")
            return []

        signals: List[TradeSignal] = []

        for symbol in symbols:
            try:
                daily_df = self.data_feed.get_daily_data(symbol, n_bars=300)
                weekly_df = self.data_feed.get_weekly_data(symbol, n_bars=100)

                signal = self._analyze_symbol_impl(
                    symbol, daily_df, weekly_df, cash_available, risk_per_trade
                )
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue

        signals.sort(
            key=lambda x: (x.conviction == "high", x.risk_reward), reverse=True
        )
        return signals


class SwingTradingSignals:
    """Generate entry signals for swing trading"""

    name = "Swing Trading Strategy"

    def __init__(self, data_feed: Any) -> None:
        self.data_feed = data_feed
        self.indicator_calc = IndicatorCalculator()

    def _check_entry_criteria(
        self, last: pd.Series, prev: pd.Series, daily_df: pd.DataFrame
    ) -> tuple[int, Dict[str, bool], float]:
        """Check entry criteria and return count + details + recent highs"""
        criteria = {}
        current_price = float(last.get("close", 0))

        # Filters
        criteria["above_sma50"] = bool(current_price > last.get("sma_50", 0))

        daily_rsi = last.get("rsi_14", np.nan)
        criteria["rsi_ok"] = bool(
            not pd.isna(daily_rsi) and RSI_MIN_SWING <= daily_rsi <= RSI_MAX_SWING
        )

        # Entry triggers
        recent_highs = float(daily_df["high"].tail(10).max())
        criteria["breakout"] = bool(current_price > recent_highs * BREAKOUT_TOLERANCE)
        criteria["volume_expansion"] = bool(
            last.get("volume_ratio", 0) > VOLUME_RATIO_MIN
        )
        criteria["macd_cross"] = safe_compare(
            last.get("macd"), ">", last.get("macd_signal")
        ) and safe_compare(prev.get("macd"), "<=", prev.get("macd_signal"))
        criteria["ma_aligned"] = safe_compare(
            last.get("close"), ">", last.get("sma_20")
        ) and safe_compare(last.get("sma_20"), ">", last.get("sma_50"))

        # New indicator triggers
        criteria["stoch_oversold"] = bool(last.get("stoch_k", 100) < STOCH_OVERSOLD)
        prev_stoch_k = prev.get("stoch_k", 0) if prev is not None else 0
        prev_stoch_d = prev.get("stoch_d", 0) if prev is not None else 0
        criteria["stoch_cross_up"] = bool(
            last.get("stoch_k", 0) > last.get("stoch_d", 0)
            and prev_stoch_k <= prev_stoch_d
        )

        criteria["adx_strong"] = bool(
            last.get("adx", 0) > ADX_STRONG_TREND
            if not pd.isna(last.get("adx"))
            else False
        )

        criteria["mfi_oversold"] = bool(
            last.get("mfi", 100) < MFI_OVERSOLD
            if not pd.isna(last.get("mfi"))
            else False
        )

        criteria["near_support"] = bool(
            current_price < last.get("s1", np.inf) * 1.02
            if not pd.isna(last.get("s1"))
            else False
        )

        # Count criteria met
        criteria_met = sum(
            [
                criteria["breakout"],
                criteria["volume_expansion"],
                criteria["macd_cross"],
                criteria["ma_aligned"],
                criteria["stoch_oversold"] or criteria["stoch_cross_up"],
                criteria["mfi_oversold"],
                criteria["adx_strong"],
                criteria["near_support"],
            ]
        )

        return criteria_met, criteria, recent_highs

    def _analyze_symbol_impl(
        self,
        symbol: str,
        daily_df: pd.DataFrame,
        cash_available: float,
        risk_per_trade: float,
    ) -> Optional[TradeSignal]:
        """Internal implementation of swing analysis"""
        if not validate_price_data(daily_df, MIN_DATA_BARS):
            logger.debug(f"{symbol}: Insufficient daily data")
            return None

        daily_df = self.indicator_calc.calculate_indicators(daily_df)

        last = daily_df.iloc[-1]
        prev = safe_get_prev_candle(daily_df)
        current_price = float(last.get("close", 0))

        if current_price <= 0:
            return None

        # Check filters
        if not safe_compare(current_price, ">", last.get("sma_50", 0)):
            return None

        daily_rsi = last.get("rsi_14", np.nan)
        if pd.isna(daily_rsi) or daily_rsi < RSI_MIN_SWING or daily_rsi > RSI_MAX_SWING:
            return None

        # Check entry criteria
        criteria_met, criteria, recent_highs = self._check_entry_criteria(
            last, prev, daily_df
        )

        if criteria_met < SWING_CRITERIA_MIN:
            adx_val = f"{last.get('adx', np.nan):.1f}"
            mfi_val = f"{last.get('mfi', np.nan):.1f}"
            logger.info(
                f"{symbol}: Only {criteria_met}/{SWING_CRITERIA_MIN} criteria | "
                f"stoch={last.get('stoch_k', np.nan):.1f}, adx={adx_val}, mfi={mfi_val}"
            )
            return None

        # Execution
        entry_price = recent_highs * 1.002 if criteria["breakout"] else current_price
        atr = last.get("atr_14", np.nan)

        if pd.isna(atr) or atr <= 0:
            return None

        stop_atr = current_price - (ATR_MULTIPLIER_SWING * atr)
        stop_breakout = recent_highs * STOP_BUFFER_BREAKOUT
        stop_loss = max(stop_atr, stop_breakout)

        risk = entry_price - stop_loss
        if risk <= 0:
            return None

        target_price = entry_price + (RISK_REWARD_SWING * risk)

        profit_percent = ((target_price - entry_price) / entry_price) * 100
        if profit_percent < MOROCCAN_MIN_PROFIT_PERCENT:
            target_price = entry_price * (1 + MOROCCAN_MIN_PROFIT_PERCENT / 100)

        risk_reward = (target_price - entry_price) / risk

        if criteria_met >= 4 and risk_reward >= RISK_REWARD_ENTRY_HIGH:
            conviction = "high"
        elif criteria_met >= 3 and risk_reward >= RISK_REWARD_ENTRY_MEDIUM:
            conviction = "medium"
        else:
            conviction = "low"

        risk_amount = cash_available * risk_per_trade
        position_size = max(1, int(risk_amount / risk))

        notes = f"RSI: {daily_rsi:.1f}, Breakout: {criteria['breakout']}, Volume surge: {criteria['volume_expansion']}"

        return TradeSignal(
            symbol=symbol,
            setup_type="swing",
            current_price=current_price,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            target_price=round(target_price, 2),
            position_size=position_size,
            risk_amount=round(risk_amount, 2),
            risk_reward=round(risk_reward, 2),
            conviction=conviction,
            atr=round(atr, 2),
            rsi=round(daily_rsi, 1),
            volume_ratio=round(last.get("volume_ratio", 0), 2),
            setup_notes=notes,
        )

    def analyze(
        self,
        symbol: str,
        daily_df: pd.DataFrame,
        weekly_df: Optional[pd.DataFrame] = None,
        cash_available: float = 10000,
        risk_per_trade: float = SWING_RISK_DEFAULT,
    ) -> Optional[TradeSignal]:
        """Analyze symbol for swing trading setup (for backtesting)"""
        return self._analyze_symbol_impl(
            symbol, daily_df, cash_available, risk_per_trade
        )

    def generate_signals(
        self,
        symbols: List[str],
        cash_available: float = 10000,
        risk_per_trade: float = SWING_RISK_DEFAULT,
    ) -> List[TradeSignal]:
        """Generate swing trading entry signals"""
        if cash_available <= 0:
            logger.error("cash_available must be positive")
            return []
        if risk_per_trade <= 0 or risk_per_trade > 1:
            logger.error("risk_per_trade must be between 0 and 1")
            return []

        signals: List[TradeSignal] = []

        for symbol in symbols:
            try:
                daily_df = self.data_feed.get_daily_data(symbol, n_bars=200)

                signal = self._analyze_symbol_impl(
                    symbol, daily_df, cash_available, risk_per_trade
                )
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue

        signals.sort(
            key=lambda x: (x.conviction == "high", x.risk_reward), reverse=True
        )
        return signals


def run_stage2_signals(
    data_dir: str = "./data",
    tv_username: Optional[str] = None,
    tv_password: Optional[str] = None,
    cash_available: float = 10000,
) -> Dict[str, List[TradeSignal]]:
    """Main function to run Stage 2 signal generation"""

    from src.data.tv_provider import TradingViewProvider
    from src.stage1_screening import CSEMAWatchlistManager

    logger.info("=" * 60)
    logger.info("GENERATING ENTRY SIGNALS - STAGE 2")
    logger.info("=" * 60)

    # Load watchlist
    manager = CSEMAWatchlistManager(data_dir)
    watchlist = manager.load_watchlist()

    if watchlist["position"].empty and watchlist["swing"].empty:
        logger.warning("No watchlist found. Please run Stage 1 first.")
        return {"position": [], "swing": []}

    # Initialize data provider
    provider = TradingViewProvider(tv_username, tv_password)

    position_signals = PositionTradingSignals(provider)
    swing_signals = SwingTradingSignals(provider)

    results: Dict[str, List[TradeSignal]] = {"position": [], "swing": []}

    # Generate position signals
    if not watchlist["position"].empty:
        symbols = watchlist["position"]["Symbol"].tolist()
        symbols = [clean_symbol(s) for s in symbols]
        logger.info(f"Analyzing {len(symbols)} position candidates: {symbols}")
        results["position"] = position_signals.generate_signals(symbols, cash_available)
        logger.info(f"Generated {len(results['position'])} position signals")

    # Generate swing signals
    if not watchlist["swing"].empty:
        symbols = watchlist["swing"]["Symbol"].tolist()
        symbols = [clean_symbol(s) for s in symbols]
        logger.info(f"Analyzing {len(symbols)} swing candidates: {symbols}")
        results["swing"] = swing_signals.generate_signals(symbols, cash_available)
        logger.info(f"Generated {len(results['swing'])} swing signals")

    return results
