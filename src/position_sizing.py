"""
Position Sizing Module - Calculates position sizes based on available cash
Assumes $10,000 cash available (no portfolio required)
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingRecommendation:
    """Position sizing recommendation for a signal"""

    symbol: str
    setup_type: str
    entry_price: float
    stop_loss: float
    target_price: float
    risk_per_share: float
    max_position_size: int
    position_value: float
    risk_amount: float
    risk_reward: float
    conviction: str
    # Fee-related fields
    buy_fees: float = 0.0
    roundtrip_fees: float = 0.0
    break_even_price: float = 0.0
    net_risk_reward: float = 0.0
    total_cost_with_fees: float = 0.0


def _extract_signal_value(signal: Dict[str, Any], keys: List[str]) -> Any:
    """Extract value from signal trying multiple key variants"""
    for key in keys:
        if key in signal:
            return signal[key]
    return None


def calculate_position_sizing(
    signals_data: List[Dict[str, Any]], cash_available: float = 10000
) -> List[PositionSizingRecommendation]:
    """Calculate position sizing for given signals"""
    if cash_available <= 0:
        logger.error("cash_available must be positive")
        return []

    from src.brokerage_fees import (
        calculate_buy_fees,
        calculate_roundtrip_fees,
        calculate_break_even_price,
    )

    recommendations: List[PositionSizingRecommendation] = []

    # Risk levels
    risk_levels = {
        "high": 0.02,  # 2% for high conviction
        "medium": 0.015,  # 1.5% for medium conviction
        "low": 0.01,  # 1% for low conviction
    }

    for signal in signals_data:
        try:
            symbol = _extract_signal_value(
                signal, ["symbol", "Symbol", "ticker", "Ticker"]
            )
            if not symbol:
                logger.warning("Signal missing symbol, skipping")
                continue

            setup_type = _extract_signal_value(
                signal, ["setup_type", "Setup Type", "type", "Type"]
            )
            if not setup_type:
                setup_type = "position"

            entry_price = _extract_signal_value(
                signal, ["entry_price", "Entry Price", "entry", "Entry"]
            )
            stop_loss = _extract_signal_value(
                signal, ["stop_loss", "Stop Loss", "stop", "Stop"]
            )
            target_price = _extract_signal_value(
                signal, ["target_price", "Target Price", "target", "Target"]
            )
            conviction = _extract_signal_value(signal, ["conviction", "Conviction"])

            # Convert to float and validate
            try:
                entry_price = float(entry_price) if entry_price is not None else 0.0
                stop_loss = float(stop_loss) if stop_loss is not None else 0.0
                target_price = float(target_price) if target_price is not None else 0.0
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid numeric values for {symbol}: {e}")
                continue

            if entry_price <= 0 or stop_loss <= 0:
                logger.debug(f"{symbol}: Invalid entry or stop price")
                continue

            risk_per_share = entry_price - stop_loss
            if risk_per_share <= 0:
                logger.debug(f"{symbol}: Risk per share must be positive")
                continue

            # Get risk percentage based on conviction
            conviction_str = str(conviction).lower() if conviction else "medium"
            risk_pct = risk_levels.get(conviction_str, 0.015)
            risk_amount = cash_available * risk_pct

            # Calculate position size
            position_size = max(1, int(risk_amount / risk_per_share))

            # Calculate position value
            position_value = position_size * entry_price

            # Calculate risk/reward (gross)
            if target_price > 0:
                reward_per_share = target_price - entry_price
                risk_reward = reward_per_share / risk_per_share
            else:
                risk_reward = 0.0

            # Calculate fees
            buy_fees_calc = calculate_buy_fees(position_value)
            roundtrip_fees_calc = calculate_roundtrip_fees(position_value)
            break_even = calculate_break_even_price(entry_price, position_size)

            # Calculate net R/R including fees
            total_cost = buy_fees_calc.net_amount
            gross_reward = (target_price - entry_price) * position_size
            net_reward = gross_reward - roundtrip_fees_calc.total_fees
            net_risk_reward = (
                net_reward / (risk_amount + buy_fees_calc.total_fees)
                if (risk_amount + buy_fees_calc.total_fees) > 0
                else 0.0
            )

            recommendation = PositionSizingRecommendation(
                symbol=str(symbol),
                setup_type=str(setup_type),
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                risk_per_share=risk_per_share,
                max_position_size=position_size,
                position_value=position_value,
                risk_amount=risk_amount,
                risk_reward=round(risk_reward, 2),
                conviction=conviction_str,
                buy_fees=round(buy_fees_calc.total_fees, 2),
                roundtrip_fees=round(roundtrip_fees_calc.total_fees, 2),
                break_even_price=round(break_even, 2),
                net_risk_reward=round(net_risk_reward, 2),
                total_cost_with_fees=round(total_cost, 2),
            )

            recommendations.append(recommendation)

        except Exception as e:
            logger.error(f"Error calculating position sizing for signal: {e}")
            continue

    return recommendations


def show_position_sizing_recommendations(
    data_dir: str = "./data", cash_available: float = 10000
) -> None:
    """Show position sizing recommendations based on available cash"""

    import os
    from src.stage1_screening import CSEMAWatchlistManager
    from src.stage2_signals import run_stage2_signals

    logger.info("Calculating position sizing recommendations...")

    # Get TV credentials from environment
    tv_username = os.environ.get("TV_USERNAME")
    tv_password = os.environ.get("TV_PASSWORD")

    # Load watchlist and generate signals
    manager = CSEMAWatchlistManager(data_dir)
    watchlist = manager.load_watchlist()

    print("\n" + "=" * 80)
    print("POSITION SIZING RECOMMENDATIONS")
    print("=" * 80)
    print(f"\nCash Available: {cash_available:,.2f} MAD")

    # Risk levels info
    print("\nRisk Levels:")
    print("  High Conviction:    2% risk per trade")
    print("  Medium Conviction:  1.5% risk per trade")
    print("  Low Conviction:     1% risk per trade")

    print("\n" + "-" * 80)

    if watchlist["position"].empty and watchlist["swing"].empty:
        print("\nNo watchlist found. Please run Stage 1 first:")
        print("  python main.py --stage screening")
        return

    # Generate signals to get detailed info
    print("\nGenerating signals for position sizing...")
    signals = run_stage2_signals(
        data_dir=data_dir,
        tv_username=tv_username,
        tv_password=tv_password,
        cash_available=cash_available,
    )

    all_recommendations: List[PositionSizingRecommendation] = []

    for setup_type, signal_list in signals.items():
        if not signal_list:
            continue

        print(f"\n{setup_type.upper()} SIGNALS:")
        print("-" * 80)

        # Convert TradeSignal objects to dict
        signals_data: List[Dict[str, Any]] = []
        for s in signal_list:
            signals_data.append(
                {
                    "symbol": s.symbol,
                    "setup_type": s.setup_type,
                    "entry_price": s.entry_price,
                    "stop_loss": s.stop_loss,
                    "target_price": s.target_price,
                    "conviction": s.conviction,
                }
            )

        recommendations = calculate_position_sizing(signals_data, cash_available)
        all_recommendations.extend(recommendations)

        for rec in recommendations:
            print(f"\n  {rec.symbol} ({rec.conviction.upper()})")
            print(f"    Entry: {rec.entry_price:.2f} MAD")
            print(f"    Stop:  {rec.stop_loss:.2f} MAD")
            print(f"    Target: {rec.target_price:.2f} MAD")
            print(f"    Risk/Reward (Gross): 1:{rec.risk_reward:.1f}")
            print(f"    Risk/Reward (Net):   1:{rec.net_risk_reward:.1f}")
            print(f"    Max Shares: {rec.max_position_size}")
            print(f"    Position Value: {rec.position_value:,.2f} MAD")
            print(f"    Buy Fees: {rec.buy_fees:,.2f} MAD")
            print(f"    Roundtrip Fees: {rec.roundtrip_fees:,.2f} MAD")
            print(f"    Total Cost (with fees): {rec.total_cost_with_fees:,.2f} MAD")
            print(f"    Break-Even Price: {rec.break_even_price:.2f} MAD")
            risk_pct = (
                (rec.risk_amount / cash_available * 100) if cash_available > 0 else 0
            )
            print(f"    Risk Amount: {rec.risk_amount:,.2f} MAD ({risk_pct:.1f}%)")

    # Summary
    if all_recommendations:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        total_risk = sum(r.risk_amount for r in all_recommendations)
        total_value = sum(r.position_value for r in all_recommendations)
        total_buy_fees = sum(r.buy_fees for r in all_recommendations)
        total_roundtrip_fees = sum(r.roundtrip_fees for r in all_recommendations)
        total_with_fees = sum(r.total_cost_with_fees for r in all_recommendations)

        print(f"\nTotal Signals: {len(all_recommendations)}")
        print(f"Total Risk Required: {total_risk:,.2f} MAD")
        print(f"Total Position Value: {total_value:,.2f} MAD")
        print(f"Total Buy Fees: {total_buy_fees:,.2f} MAD")
        print(f"Total Roundtrip Fees: {total_roundtrip_fees:,.2f} MAD")
        print(f"Total Capital Required (with fees): {total_with_fees:,.2f} MAD")
        print(f"Cash Available: {cash_available:,.2f} MAD")

        if total_with_fees > cash_available:
            print(
                f"\nWARNING: Signals require {total_with_fees:,.2f} MAD (with fees) but only {cash_available:,.2f} MAD available"
            )
            print(
                "Recommendation: Select top 3-5 signals by conviction and net risk/reward"
            )
        else:
            print(
                f"\nOK: All signals can be taken with available cash (including fees)"
            )
    else:
        print("\nNo signals generated for position sizing.")

    print("\n" + "=" * 80)
