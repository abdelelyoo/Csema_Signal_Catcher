"""
Performance Tracking Module - Tracks trading performance metrics
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of an executed trade"""

    trade_id: str
    date_entered: datetime
    date_exited: Optional[datetime] = None
    symbol: str = ""
    setup_type: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    position_size: int = 0
    stop_loss: float = 0.0
    target_price: float = 0.0
    risk_amount: float = 0.0
    # Gross metrics (before fees)
    pnl: float = field(default=0.0, init=False)
    pnl_percent: float = field(default=0.0, init=False)
    r_multiple: float = field(default=0.0, init=False)
    # Net metrics (after fees)
    net_pnl: float = field(default=0.0, init=False)
    net_pnl_percent: float = field(default=0.0, init=False)
    total_fees: float = field(default=0.0, init=False)
    exit_reason: str = ""
    conviction: str = ""
    setup_notes: str = ""

    def calculate_metrics(self) -> None:
        """Calculate trade metrics including brokerage fees"""
        from src.brokerage_fees import calculate_net_pnl

        if self.exit_price > 0 and self.entry_price > 0:
            # Gross P&L (before fees)
            self.pnl = (self.exit_price - self.entry_price) * self.position_size
            self.pnl_percent = (
                (self.exit_price - self.entry_price) / self.entry_price
            ) * 100

            risk_per_share = self.entry_price - self.stop_loss
            if risk_per_share > 0:
                actual_pnl_per_share = self.exit_price - self.entry_price
                self.r_multiple = actual_pnl_per_share / risk_per_share

            # Net P&L (after fees)
            pnl_data = calculate_net_pnl(
                self.entry_price, self.exit_price, self.position_size
            )
            self.net_pnl = pnl_data["net_pnl"]
            self.net_pnl_percent = (
                self.net_pnl / pnl_data["gross_entry"] * 100
                if pnl_data["gross_entry"] > 0
                else 0.0
            )
            self.total_fees = pnl_data["total_fees"]


class TradeJournal:
    """Journal for tracking all trades"""

    TRADE_COLUMNS = [
        "trade_id",
        "date_entered",
        "date_exited",
        "symbol",
        "setup_type",
        "entry_price",
        "exit_price",
        "position_size",
        "stop_loss",
        "target_price",
        "risk_amount",
        "pnl",  # Gross P&L
        "pnl_percent",  # Gross P&L %
        "r_multiple",
        "net_pnl",  # Net P&L (after fees)
        "net_pnl_percent",  # Net P&L %
        "total_fees",  # Total brokerage fees
        "exit_reason",
        "conviction",
        "setup_notes",
    ]

    def __init__(self, data_dir: str = "./data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.trades_file = self.data_dir / "trade_journal.csv"
        self.trades: List[TradeRecord] = []
        self._load_trades()

    def _load_trades(self) -> None:
        """Load existing trades from file"""
        if not self.trades_file.exists():
            return

        try:
            df = pd.read_csv(self.trades_file)
            # Validate required columns
            missing_cols = set(self.TRADE_COLUMNS) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing columns in trade file: {missing_cols}")
                return

            for _, row in df.iterrows():
                trade = TradeRecord(
                    trade_id=str(row["trade_id"]),
                    date_entered=pd.to_datetime(row["date_entered"]),
                    date_exited=pd.to_datetime(row["date_exited"])
                    if pd.notna(row["date_exited"])
                    else None,
                    symbol=str(row["symbol"]),
                    setup_type=str(row["setup_type"]),
                    entry_price=float(row["entry_price"]),
                    exit_price=float(row["exit_price"]),
                    position_size=int(row["position_size"]),
                    stop_loss=float(row["stop_loss"]),
                    target_price=float(row["target_price"]),
                    risk_amount=float(row["risk_amount"]),
                    exit_reason=str(row["exit_reason"]),
                    conviction=str(row["conviction"]),
                    setup_notes=str(row["setup_notes"]),
                )
                self.trades.append(trade)
            logger.info(f"Loaded {len(self.trades)} trades from journal")
        except Exception as e:
            logger.error(f"Error loading trades: {e}")

    def add_trade(self, trade: TradeRecord) -> None:
        """Add a new trade to the journal"""
        self.trades.append(trade)
        self._save_trades()
        logger.info(f"Added trade {trade.trade_id} for {trade.symbol}")

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_date: datetime,
        exit_reason: str,
    ) -> bool:
        """Close an open trade"""
        for trade in self.trades:
            if trade.trade_id == trade_id and trade.date_exited is None:
                trade.exit_price = exit_price
                trade.date_exited = exit_date
                trade.exit_reason = exit_reason
                trade.calculate_metrics()
                self._save_trades()
                logger.info(
                    f"Closed trade {trade_id}: Gross P&L = {trade.pnl:.2f} MAD, "
                    f"Net P&L = {trade.net_pnl:.2f} MAD, Fees = {trade.total_fees:.2f} MAD"
                )
                return True
        return False

    def _save_trades(self) -> None:
        """Save all trades to file"""
        if not self.trades:
            return

        data: List[Dict[str, Any]] = []
        for trade in self.trades:
            data.append(
                {
                    "trade_id": trade.trade_id,
                    "date_entered": trade.date_entered,
                    "date_exited": trade.date_exited,
                    "symbol": trade.symbol,
                    "setup_type": trade.setup_type,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "position_size": trade.position_size,
                    "stop_loss": trade.stop_loss,
                    "target_price": trade.target_price,
                    "risk_amount": trade.risk_amount,
                    "pnl": trade.pnl,
                    "pnl_percent": trade.pnl_percent,
                    "r_multiple": trade.r_multiple,
                    "net_pnl": trade.net_pnl,
                    "net_pnl_percent": trade.net_pnl_percent,
                    "total_fees": trade.total_fees,
                    "exit_reason": trade.exit_reason,
                    "conviction": trade.conviction,
                    "setup_notes": trade.setup_notes,
                }
            )

        try:
            df = pd.DataFrame(data)
            df.to_csv(self.trades_file, index=False)
        except Exception as e:
            logger.error(f"Error saving trades: {e}")

    def get_open_trades(self) -> List[TradeRecord]:
        """Get all open trades"""
        return [t for t in self.trades if t.date_exited is None]

    def get_closed_trades(self) -> List[TradeRecord]:
        """Get all closed trades"""
        return [t for t in self.trades if t.date_exited is not None]


class PerformanceAnalyzer:
    """Analyzes trading performance and calculates metrics"""

    def __init__(self, trade_journal: TradeJournal) -> None:
        self.journal = trade_journal

    def calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall performance metrics including gross and net P&L"""
        closed_trades = self.journal.get_closed_trades()

        if not closed_trades:
            return {"message": "No closed trades to analyze"}

        total_trades = len(closed_trades)

        # Gross metrics (before fees)
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        total_pnl = sum(t.pnl for t in closed_trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        avg_win = sum(t.pnl for t in winning_trades) / win_count if win_count > 0 else 0
        avg_loss = (
            sum(t.pnl for t in losing_trades) / loss_count if loss_count > 0 else 0
        )

        # Net metrics (after fees)
        net_winning_trades = [t for t in closed_trades if t.net_pnl > 0]
        net_losing_trades = [t for t in closed_trades if t.net_pnl <= 0]
        net_win_count = len(net_winning_trades)
        net_loss_count = len(net_losing_trades)
        net_win_rate = (net_win_count / total_trades * 100) if total_trades > 0 else 0

        total_net_pnl = sum(t.net_pnl for t in closed_trades)
        avg_net_pnl = total_net_pnl / total_trades if total_trades > 0 else 0
        avg_net_win = (
            sum(t.net_pnl for t in net_winning_trades) / net_win_count
            if net_win_count > 0
            else 0
        )
        avg_net_loss = (
            sum(t.net_pnl for t in net_losing_trades) / net_loss_count
            if net_loss_count > 0
            else 0
        )

        # Calculate profit factor with safety check
        total_losses = sum(t.pnl for t in losing_trades)
        if total_losses != 0:
            profit_factor = abs(sum(t.pnl for t in winning_trades) / total_losses)
        else:
            profit_factor = (
                float("inf") if sum(t.pnl for t in winning_trades) > 0 else 0
            )

        # Net profit factor
        total_net_losses = sum(t.net_pnl for t in net_losing_trades)
        if total_net_losses != 0:
            net_profit_factor = abs(
                sum(t.net_pnl for t in net_winning_trades) / total_net_losses
            )
        else:
            net_profit_factor = (
                float("inf") if sum(t.net_pnl for t in net_winning_trades) > 0 else 0
            )

        r_multiples = [t.r_multiple for t in closed_trades if t.r_multiple != 0]
        avg_r = np.mean(r_multiples) if r_multiples else 0

        # Gross expectancy
        win_pct = win_rate / 100
        loss_pct = (100 - win_rate) / 100
        expectancy = (
            (win_pct * avg_win) - (loss_pct * abs(avg_loss)) if total_trades > 0 else 0
        )

        # Net expectancy
        net_win_pct = net_win_rate / 100
        net_loss_pct = (100 - net_win_rate) / 100
        net_expectancy = (
            (net_win_pct * avg_net_win) - (net_loss_pct * abs(avg_net_loss))
            if total_trades > 0
            else 0
        )

        # Total fees
        total_fees = sum(t.total_fees for t in closed_trades)
        avg_fees_per_trade = total_fees / total_trades if total_trades > 0 else 0

        return {
            # Gross metrics
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_per_trade": round(avg_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_r_multiple": round(avg_r, 2),
            "expectancy": round(expectancy, 3),
            # Net metrics (after fees)
            "net_winning_trades": net_win_count,
            "net_losing_trades": net_loss_count,
            "net_win_rate": round(net_win_rate, 2),
            "total_net_pnl": round(total_net_pnl, 2),
            "avg_net_pnl_per_trade": round(avg_net_pnl, 2),
            "avg_net_win": round(avg_net_win, 2),
            "avg_net_loss": round(avg_net_loss, 2),
            "net_profit_factor": round(net_profit_factor, 2),
            "net_expectancy": round(net_expectancy, 3),
            # Fees
            "total_fees_paid": round(total_fees, 2),
            "avg_fees_per_trade": round(avg_fees_per_trade, 2),
        }

    def calculate_by_setup_type(self) -> Dict[str, Dict[str, Any]]:
        """Calculate metrics by setup type"""
        closed_trades = self.journal.get_closed_trades()
        result: Dict[str, Dict[str, Any]] = {}

        for setup_type in ["position", "swing"]:
            trades = [t for t in closed_trades if t.setup_type == setup_type]
            if not trades:
                result[setup_type] = {"message": f"No {setup_type} trades"}
                continue

            total = len(trades)
            wins = len([t for t in trades if t.pnl > 0])
            win_rate = (wins / total * 100) if total > 0 else 0
            total_pnl = sum(t.pnl for t in trades)
            avg_pnl = total_pnl / total if total > 0 else 0

            r_multiples = [t.r_multiple for t in trades if t.r_multiple != 0]
            avg_r = np.mean(r_multiples) if r_multiples else 0

            result[setup_type] = {
                "total_trades": total,
                "win_rate": round(win_rate, 2),
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(avg_pnl, 2),
                "avg_r_multiple": round(avg_r, 2),
            }

        return result

    def calculate_drawdown(self, account_size: float = 10000) -> Dict[str, Any]:
        """Calculate maximum drawdown"""
        if account_size <= 0:
            return {"message": "Invalid account size"}

        closed_trades = self.journal.get_closed_trades()

        if not closed_trades:
            return {"message": "No trades to analyze"}

        trades_sorted = sorted(closed_trades, key=lambda x: x.date_exited)
        cumulative_pnl: List[float] = []
        running_total = 0.0

        for trade in trades_sorted:
            running_total += trade.pnl
            cumulative_pnl.append(running_total)

        peak = account_size
        max_drawdown = 0.0
        max_drawdown_percent = 0.0

        for pnl in cumulative_pnl:
            current_equity = account_size + pnl
            if current_equity > peak:
                peak = current_equity
            drawdown = peak - current_equity
            drawdown_percent = (drawdown / peak) * 100 if peak > 0 else 0

            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_percent = drawdown_percent

        return {
            "max_drawdown_mad": round(max_drawdown, 2),
            "max_drawdown_percent": round(max_drawdown_percent, 2),
            "current_equity": round(account_size + cumulative_pnl[-1], 2),
            "total_return": round(cumulative_pnl[-1], 2),
            "total_return_percent": round((cumulative_pnl[-1] / account_size) * 100, 2),
        }

    def generate_report(self, account_size: float = 10000) -> str:
        """Generate a comprehensive performance report"""
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("SIGNAL CATCHER - PERFORMANCE REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        overall = self.calculate_overall_metrics()

        # GROSS PERFORMANCE (Before Fees)
        lines.append("GROSS PERFORMANCE (Before Fees)")
        lines.append("-" * 80)
        if "message" in overall:
            lines.append(str(overall["message"]))
        else:
            lines.append(f"Total Trades: {overall['total_trades']}")
            lines.append(f"Winning Trades: {overall['winning_trades']}")
            lines.append(f"Losing Trades: {overall['losing_trades']}")
            lines.append(f"Win Rate: {overall['win_rate']}%")
            lines.append(f"Total P&L: {overall['total_pnl']:.2f} MAD")
            lines.append(
                f"Average P&L per Trade: {overall['avg_pnl_per_trade']:.2f} MAD"
            )
            lines.append(f"Average Win: {overall['avg_win']:.2f} MAD")
            lines.append(f"Average Loss: {overall['avg_loss']:.2f} MAD")
            lines.append(f"Profit Factor: {overall['profit_factor']}")
            lines.append(f"Average R-Multiple: {overall['avg_r_multiple']:.2f}R")
            lines.append(f"Expectancy: {overall['expectancy']:.3f}")
        lines.append("")

        # NET PERFORMANCE (After Fees)
        if "message" not in overall:
            lines.append("NET PERFORMANCE (After Brokerage Fees)")
            lines.append("-" * 80)
            lines.append(f"Winning Trades: {overall['net_winning_trades']}")
            lines.append(f"Losing Trades: {overall['net_losing_trades']}")
            lines.append(f"Win Rate: {overall['net_win_rate']}%")
            lines.append(f"Total Net P&L: {overall['total_net_pnl']:.2f} MAD")
            lines.append(
                f"Average Net P&L per Trade: {overall['avg_net_pnl_per_trade']:.2f} MAD"
            )
            lines.append(f"Average Net Win: {overall['avg_net_win']:.2f} MAD")
            lines.append(f"Average Net Loss: {overall['avg_net_loss']:.2f} MAD")
            lines.append(f"Net Profit Factor: {overall['net_profit_factor']}")
            lines.append(f"Net Expectancy: {overall['net_expectancy']:.3f}")
            lines.append("")

            # BROKERAGE FEES
            lines.append("BROKERAGE FEES IMPACT")
            lines.append("-" * 80)
            lines.append(f"Total Fees Paid: {overall['total_fees_paid']:.2f} MAD")
            lines.append(
                f"Average Fees per Trade: {overall['avg_fees_per_trade']:.2f} MAD"
            )
            fee_impact = (
                (overall["total_pnl"] - overall["total_net_pnl"])
                / overall["total_pnl"]
                * 100
                if overall["total_pnl"] != 0
                else 0
            )
            lines.append(f"Fees Impact on P&L: {fee_impact:.1f}%")
        lines.append("")

        by_setup = self.calculate_by_setup_type()
        lines.append("PERFORMANCE BY SETUP TYPE")
        lines.append("-" * 80)
        for setup_type, metrics in by_setup.items():
            lines.append(f"\n{setup_type.upper()}:")
            if "message" in metrics:
                lines.append(f"  {metrics['message']}")
            else:
                lines.append(f"  Trades: {metrics['total_trades']}")
                lines.append(f"  Win Rate: {metrics['win_rate']}%")
                lines.append(f"  Total P&L: {metrics['total_pnl']:.2f} MAD")
                lines.append(f"  Avg R: {metrics['avg_r_multiple']:.2f}R")
        lines.append("")

        drawdown = self.calculate_drawdown(account_size)
        lines.append("DRAWDOWN ANALYSIS")
        lines.append("-" * 80)
        if "message" in drawdown:
            lines.append(str(drawdown["message"]))
        else:
            lines.append(f"Current Equity: {drawdown['current_equity']:.2f} MAD")
            lines.append(
                f"Total Return: {drawdown['total_return']:.2f} MAD ({drawdown['total_return_percent']}%)"
            )
            lines.append(
                f"Max Drawdown: {drawdown['max_drawdown_mad']:.2f} MAD ({drawdown['max_drawdown_percent']}%)"
            )
        lines.append("")

        open_trades = self.journal.get_open_trades()
        lines.append("OPEN POSITIONS")
        lines.append("-" * 80)
        if open_trades:
            lines.append(f"Total Open Positions: {len(open_trades)}")
            for trade in open_trades:
                lines.append(
                    f"  {trade.symbol}: Entry {trade.entry_price}, Size {trade.position_size}"
                )
        else:
            lines.append("No open positions")
        lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)


class PerformanceTracker:
    """Main class for tracking all performance metrics"""

    def __init__(self, data_dir: str = "./data") -> None:
        self.journal = TradeJournal(data_dir)
        self.analyzer = PerformanceAnalyzer(self.journal)

    def print_performance_report(self, account_size: float = 10000) -> None:
        """Print performance report to console"""
        report = self.analyzer.generate_report(account_size)
        print(report)

    def export_report(
        self, account_size: float = 10000, filename: Optional[str] = None
    ) -> None:
        """Export performance report to file"""
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d')}.txt"

        report = self.analyzer.generate_report(account_size)
        filepath = self.journal.data_dir / filename

        try:
            with open(filepath, "w") as f:
                f.write(report)
            logger.info(f"Performance report saved to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting report: {e}")


def run_performance_analysis(
    data_dir: str = "./data", account_size: float = 10000
) -> None:
    """Run complete performance analysis"""
    tracker = PerformanceTracker(data_dir)
    tracker.print_performance_report(account_size)
    tracker.export_report(account_size)
