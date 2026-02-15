"""
Signal Catcher Web Application
Flask-based web interface for CSEMA Trading System
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.stage1_screening import run_stage1_screening, CSEMAWatchlistManager
from src.stage2_signals import run_stage2_signals
from src.position_sizing import calculate_position_sizing
from src.performance_tracker import PerformanceTracker
from src.brokerage_fees import calculate_buy_fees, calculate_roundtrip_fees

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "signal-catcher-secret-key")
app.config["DATA_DIR"] = "./data"
app.config["CASH_DEFAULT"] = 100000

# Ensure data directory exists
Path(app.config["DATA_DIR"]).mkdir(exist_ok=True)


@app.route("/")
def index():
    """Home page"""
    return render_template("index.html")


@app.route("/screening", methods=["GET", "POST"])
def screening():
    """Stage 1: Market Screening"""
    if request.method == "POST":
        try:
            full_watchlist = request.form.get("full_watchlist") == "on"

            # Run screening
            watchlist = run_stage1_screening(
                save_results=True,
                data_dir=app.config["DATA_DIR"],
                full_watchlist=full_watchlist,
            )

            # Handle None watchlist
            if watchlist is None:
                flash(
                    "Screening returned no data. Please check your internet connection and try again.",
                    "error",
                )
                return redirect(url_for("screening"))

            # Get DataFrames with null checks
            position_df = watchlist.get("position")
            swing_df = watchlist.get("swing")

            # Handle None DataFrames
            if position_df is None:
                position_df = pd.DataFrame()
            if swing_df is None:
                swing_df = pd.DataFrame()

            # Convert to display format
            position_data = (
                position_df.to_dict("records") if not position_df.empty else []
            )
            swing_data = swing_df.to_dict("records") if not swing_df.empty else []

            flash(
                f"Screening complete! Found {len(position_data)} position and {len(swing_data)} swing candidates",
                "success",
            )

            return render_template(
                "screening.html",
                position_candidates=position_data,
                swing_candidates=swing_data,
                total_position=len(position_data),
                total_swing=len(swing_data),
                results=True,
            )

        except ImportError as e:
            logger.error(f"Screening import error: {e}")
            flash(
                f"Missing dependency: {str(e)}. Please install required libraries.",
                "error",
            )
            return redirect(url_for("screening"))
        except Exception as e:
            logger.error(f"Screening error: {e}")
            flash(f"Error during screening: {str(e)}", "error")
            return redirect(url_for("screening"))

    return render_template("screening.html", results=False)


@app.route("/signals", methods=["GET", "POST"])
def signals():
    """Stage 2: Signal Generation"""
    if request.method == "POST":
        try:
            cash = float(request.form.get("cash", app.config["CASH_DEFAULT"]))

            # Get TV credentials from environment
            tv_username = os.environ.get("TV_USERNAME")
            tv_password = os.environ.get("TV_PASSWORD")

            # Run signal generation
            signals_data = run_stage2_signals(
                data_dir=app.config["DATA_DIR"],
                tv_username=tv_username,
                tv_password=tv_password,
                cash_available=cash,
            )

            # Convert TradeSignal objects to dict
            position_signals = []
            for s in signals_data.get("position", []):
                position_signals.append(
                    {
                        "symbol": s.symbol,
                        "setup_type": s.setup_type,
                        "entry_price": s.entry_price,
                        "stop_loss": s.stop_loss,
                        "target_price": s.target_price,
                        "target_price_2": s.target_price_2,
                        "position_size": s.position_size,
                        "risk_reward": s.risk_reward,
                        "conviction": s.conviction,
                        "atr": s.atr,
                        "rsi": s.rsi,
                        "setup_notes": s.setup_notes,
                    }
                )

            swing_signals = []
            for s in signals_data.get("swing", []):
                swing_signals.append(
                    {
                        "symbol": s.symbol,
                        "setup_type": s.setup_type,
                        "entry_price": s.entry_price,
                        "stop_loss": s.stop_loss,
                        "target_price": s.target_price,
                        "position_size": s.position_size,
                        "risk_reward": s.risk_reward,
                        "conviction": s.conviction,
                        "atr": s.atr,
                        "rsi": s.rsi,
                        "setup_notes": s.setup_notes,
                    }
                )

            flash(
                f"Signal generation complete! Found {len(position_signals)} position and {len(swing_signals)} swing signals",
                "success",
            )

            return render_template(
                "signals.html",
                position_signals=position_signals,
                swing_signals=swing_signals,
                total_position=len(position_signals),
                total_swing=len(swing_signals),
                cash=cash,
                results=True,
            )

        except Exception as e:
            logger.error(f"Signals error: {e}")
            flash(f"Error generating signals: {str(e)}", "error")
            return redirect(url_for("signals"))

    return render_template("signals.html", results=False)


@app.route("/position-sizing", methods=["GET", "POST"])
def position_sizing():
    """Position Sizing with Fees"""
    if request.method == "POST":
        try:
            cash = float(request.form.get("cash", app.config["CASH_DEFAULT"]))

            # Get TV credentials
            tv_username = os.environ.get("TV_USERNAME")
            tv_password = os.environ.get("TV_PASSWORD")

            # Generate signals first
            signals_data = run_stage2_signals(
                data_dir=app.config["DATA_DIR"],
                tv_username=tv_username,
                tv_password=tv_password,
                cash_available=cash,
            )

            all_recommendations = []

            # Calculate position sizing for both types
            for setup_type, signal_list in signals_data.items():
                if not signal_list:
                    continue

                # Convert to dict format
                signals_dict = []
                for s in signal_list:
                    signals_dict.append(
                        {
                            "symbol": s.symbol,
                            "setup_type": s.setup_type,
                            "entry_price": s.entry_price,
                            "stop_loss": s.stop_loss,
                            "target_price": s.target_price,
                            "conviction": s.conviction,
                        }
                    )

                recommendations = calculate_position_sizing(signals_dict, cash)
                all_recommendations.extend(recommendations)

            # Convert to display format
            display_recs = []
            for rec in all_recommendations:
                display_recs.append(
                    {
                        "symbol": rec.symbol,
                        "setup_type": rec.setup_type,
                        "conviction": rec.conviction,
                        "entry_price": rec.entry_price,
                        "stop_loss": rec.stop_loss,
                        "target_price": rec.target_price,
                        "risk_reward": rec.risk_reward,
                        "net_risk_reward": rec.net_risk_reward,
                        "max_position_size": rec.max_position_size,
                        "position_value": rec.position_value,
                        "risk_amount": rec.risk_amount,
                        "buy_fees": rec.buy_fees,
                        "roundtrip_fees": rec.roundtrip_fees,
                        "break_even_price": rec.break_even_price,
                        "total_cost_with_fees": rec.total_cost_with_fees,
                    }
                )

            # Calculate totals
            total_risk = sum(r["risk_amount"] for r in display_recs)
            total_value = sum(r["position_value"] for r in display_recs)
            total_buy_fees = sum(r["buy_fees"] for r in display_recs)
            total_roundtrip_fees = sum(r["roundtrip_fees"] for r in display_recs)
            total_with_fees = sum(r["total_cost_with_fees"] for r in display_recs)

            flash(
                f"Position sizing complete for {len(display_recs)} signals", "success"
            )

            return render_template(
                "position_sizing.html",
                recommendations=display_recs,
                total_signals=len(display_recs),
                total_risk=total_risk,
                total_value=total_value,
                total_buy_fees=total_buy_fees,
                total_roundtrip_fees=total_roundtrip_fees,
                total_with_fees=total_with_fees,
                cash=cash,
                sufficient_funds=total_with_fees <= cash,
                results=True,
            )

        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            flash(f"Error calculating position sizing: {str(e)}", "error")
            return redirect(url_for("position_sizing"))

    return render_template("position_sizing.html", results=False)


@app.route("/performance")
def performance():
    """Performance Tracking"""
    try:
        cash = float(request.args.get("cash", app.config["CASH_DEFAULT"]))

        tracker = PerformanceTracker(app.config["DATA_DIR"])
        report = tracker.analyzer.generate_report(cash)

        return render_template("performance.html", report=report, cash=cash)

    except Exception as e:
        logger.error(f"Performance error: {e}")
        flash(f"Error generating performance report: {str(e)}", "error")
        return render_template("performance.html", report=None)


@app.route("/api/screening", methods=["POST"])
def api_screening():
    """API endpoint for screening"""
    try:
        data = request.get_json()
        full_watchlist = data.get("full_watchlist", False)

        watchlist = run_stage1_screening(
            save_results=True,
            data_dir=app.config["DATA_DIR"],
            full_watchlist=full_watchlist,
        )

        return jsonify(
            {
                "success": True,
                "position_count": len(watchlist["position"]),
                "swing_count": len(watchlist["swing"]),
                "position_candidates": watchlist["position"].to_dict("records")
                if not watchlist["position"].empty
                else [],
                "swing_candidates": watchlist["swing"].to_dict("records")
                if not watchlist["swing"].empty
                else [],
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/signals", methods=["POST"])
def api_signals():
    """API endpoint for signals"""
    try:
        data = request.get_json()
        cash = float(data.get("cash", app.config["CASH_DEFAULT"]))

        tv_username = os.environ.get("TV_USERNAME")
        tv_password = os.environ.get("TV_PASSWORD")

        signals_data = run_stage2_signals(
            data_dir=app.config["DATA_DIR"],
            tv_username=tv_username,
            tv_password=tv_password,
            cash_available=cash,
        )

        return jsonify(
            {
                "success": True,
                "position_signals": [vars(s) for s in signals_data.get("position", [])],
                "swing_signals": [vars(s) for s in signals_data.get("swing", [])],
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/about")
def about():
    """About page"""
    return render_template("about.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 70)
    print("SIGNAL CATCHER WEB APP")
    print("=" * 70)
    print(f"Access the app at: http://127.0.0.1:{port}")
    print(f"Data directory: {app.config['DATA_DIR']}")
    print("=" * 70)
    app.run(debug=False, host="0.0.0.0", port=port)
