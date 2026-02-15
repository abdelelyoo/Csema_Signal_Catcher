#!/usr/bin/env python3
"""
Signal Catcher - Minimal CSEMA Trading System
Only includes: screening, signals, position-sizing, performance commands
Cash assumed: $10,000 (no portfolio required)
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import pandas as pd

from src.stage1_screening import run_stage1_screening
from src.stage2_signals import run_stage2_signals
from src.position_sizing import show_position_sizing_recommendations
from src.performance_tracker import run_performance_analysis


def setup_logging() -> None:
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def get_tv_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Get TradingView credentials from environment variables"""
    username = os.environ.get("TV_USERNAME")
    password = os.environ.get("TV_PASSWORD")
    return username, password


def main() -> int:
    """Main entry point for the application"""
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Signal Catcher - Minimal CSEMA Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stage screening
  python main.py --stage signals
  python main.py --position-sizing
  python main.py --performance

Credentials (optional):
  Set TV_USERNAME and TV_PASSWORD environment variables for TradingView data
        """,
    )

    parser.add_argument(
        "--stage",
        choices=["screening", "signals"],
        help="Run specific stage only",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Get ALL tickers in watchlist (no filters)",
    )
    parser.add_argument(
        "--position-sizing",
        action="store_true",
        help="Show position sizing recommendations",
    )
    parser.add_argument(
        "--performance", action="store_true", help="Generate performance report"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory path (default: ./data)",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=10000,
        help="Available cash in MAD (default: 10000)",
    )

    args = parser.parse_args()

    # Create data directory
    Path(args.data_dir).mkdir(exist_ok=True)

    # Get TV credentials from environment
    tv_username, tv_password = get_tv_credentials()

    # Execute based on arguments
    try:
        if args.stage == "screening":
            run_stage1_screening_command(args)
        elif args.stage == "signals":
            run_stage2_signals_command(args, tv_username, tv_password)
        elif args.position_sizing:
            show_position_sizing_recommendations(args.data_dir, args.cash)
        elif args.performance:
            run_performance_analysis(args.data_dir)
        else:
            parser.print_help()
            return 0
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return 1

    return 0


def run_stage1_screening_command(args: argparse.Namespace) -> None:
    """Run only Stage 1 screening"""
    logger = logging.getLogger(__name__)
    logger.info("Running Stage 1: Market Screening...")

    watchlist = run_stage1_screening(
        save_results=True, data_dir=args.data_dir, full_watchlist=args.full
    )

    print("\n" + "=" * 80)
    print("STAGE 1 RESULTS")
    print("=" * 80)
    print(f"Position Trading Candidates: {len(watchlist['position'])}")
    print(f"Swing Trading Candidates: {len(watchlist['swing'])}")

    if not watchlist["position"].empty:
        print("\nTop Position Candidates:")
        display_cols = ["Symbol", "Name", "Price", "Change %"]
        available_cols = [c for c in display_cols if c in watchlist["position"].columns]
        if available_cols:
            print(watchlist["position"][available_cols].head().to_string(index=False))

    if not watchlist["swing"].empty:
        print("\nTop Swing Candidates:")
        cols = ["Symbol", "Name", "Price", "Change %"]
        available_cols = [c for c in cols if c in watchlist["swing"].columns]
        if available_cols:
            print(watchlist["swing"][available_cols].head().to_string(index=False))


def run_stage2_signals_command(
    args: argparse.Namespace,
    tv_username: Optional[str] = None,
    tv_password: Optional[str] = None,
) -> None:
    """Run only Stage 2: Signal Generation"""
    logger = logging.getLogger(__name__)
    logger.info("Running Stage 2: Signal Generation...")

    signals = run_stage2_signals(
        data_dir=args.data_dir,
        tv_username=tv_username,
        tv_password=tv_password,
        cash_available=args.cash,
    )

    print("\n" + "=" * 80)
    print("STAGE 2 RESULTS")
    print("=" * 80)

    for setup_type, signal_list in signals.items():
        if signal_list:
            print(f"\n{setup_type.upper()} SIGNALS:")

            # Build display data
            data: list[Dict[str, Any]] = []
            for s in signal_list:
                data.append(
                    {
                        "Symbol": s.symbol,
                        "Entry Price": s.entry_price,
                        "Stop Loss": s.stop_loss,
                        "Target Price": s.target_price,
                        "Risk Reward": s.risk_reward,
                        "Conviction": s.conviction,
                    }
                )

            df = pd.DataFrame(data)
            print(df.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
