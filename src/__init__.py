"""
Signal Catcher - CSEMA Trading System
"""

from src.stage1_screening import (
    CSEMAWatchlistManager,
    CSEMAPositionScreener,
    CSEMASwingScreener,
    run_stage1_screening,
)
from src.stage2_signals import (
    TradeSignal,
    PositionTradingSignals,
    SwingTradingSignals,
    run_stage2_signals,
)
from src.position_sizing import (
    PositionSizingRecommendation,
    calculate_position_sizing,
    show_position_sizing_recommendations,
)
from src.performance_tracker import (
    TradeRecord,
    TradeJournal,
    PerformanceAnalyzer,
    PerformanceTracker,
    run_performance_analysis,
)
from src.brokerage_fees import (
    BrokerageFees,
    calculate_buy_fees,
    calculate_sell_fees,
    calculate_roundtrip_fees,
    calculate_break_even_price,
    calculate_net_pnl,
)

__all__ = [
    # Stage 1
    "CSEMAWatchlistManager",
    "CSEMAPositionScreener",
    "CSEMASwingScreener",
    "run_stage1_screening",
    # Stage 2
    "TradeSignal",
    "PositionTradingSignals",
    "SwingTradingSignals",
    "run_stage2_signals",
    # Position Sizing
    "PositionSizingRecommendation",
    "calculate_position_sizing",
    "show_position_sizing_recommendations",
    # Performance
    "TradeRecord",
    "TradeJournal",
    "PerformanceAnalyzer",
    "PerformanceTracker",
    "run_performance_analysis",
    # Brokerage Fees
    "BrokerageFees",
    "calculate_buy_fees",
    "calculate_sell_fees",
    "calculate_roundtrip_fees",
    "calculate_break_even_price",
    "calculate_net_pnl",
]

__version__ = "1.0.0"
