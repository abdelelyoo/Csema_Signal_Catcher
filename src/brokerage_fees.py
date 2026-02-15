"""
Brokerage Fee Calculator for CSEMA Trading
Calculates all trading fees and commissions for Moroccan stocks
"""

from dataclasses import dataclass
from typing import Dict

# Moroccan Brokerage Fee Constants
BROKERAGE_RATE_HT = 0.006  # 0.60%
BROKERAGE_MIN_HT = 7.50
SETTLEMENT_RATE_HT = 0.002  # 0.20%
SETTLEMENT_MIN_HT = 2.50
SBVC_RATE_HT = 0.001  # 0.10%
VAT_RATE = 0.10  # 10%


@dataclass
class BrokerageFees:
    """Complete breakdown of trading fees"""

    gross_amount: float  # Position value without fees
    brokerage_ht: float  # Brokerage commission (before VAT)
    settlement_ht: float  # Settlement fee (before VAT)
    sbvc_ht: float  # SBVC fee (before VAT)
    vat: float  # VAT on all fees
    total_fees: float  # All fees including VAT
    net_amount: float  # Total with fees (gross + fees)
    fees_percent: float  # Fees as % of gross amount


def calculate_buy_fees(gross_amount: float) -> BrokerageFees:
    """
    Calculate all fees for buying stocks

    Args:
        gross_amount: Total position value (price × quantity) in MAD

    Returns:
        BrokerageFees object with complete fee breakdown
    """
    if gross_amount <= 0:
        return BrokerageFees(
            gross_amount=0.0,
            brokerage_ht=0.0,
            settlement_ht=0.0,
            sbvc_ht=0.0,
            vat=0.0,
            total_fees=0.0,
            net_amount=0.0,
            fees_percent=0.0,
        )

    # Calculate fees (Hors Tax)
    brokerage_ht = max(gross_amount * BROKERAGE_RATE_HT, BROKERAGE_MIN_HT)
    settlement_ht = max(gross_amount * SETTLEMENT_RATE_HT, SETTLEMENT_MIN_HT)
    sbvc_ht = gross_amount * SBVC_RATE_HT

    # Calculate VAT (10% on all fees)
    total_ht = brokerage_ht + settlement_ht + sbvc_ht
    vat = total_ht * VAT_RATE

    # Total fees and net amount
    total_fees = total_ht + vat
    net_amount = gross_amount + total_fees
    fees_percent = (total_fees / gross_amount * 100) if gross_amount > 0 else 0.0

    return BrokerageFees(
        gross_amount=round(gross_amount, 2),
        brokerage_ht=round(brokerage_ht, 2),
        settlement_ht=round(settlement_ht, 2),
        sbvc_ht=round(sbvc_ht, 2),
        vat=round(vat, 2),
        total_fees=round(total_fees, 2),
        net_amount=round(net_amount, 2),
        fees_percent=round(fees_percent, 4),
    )


def calculate_sell_fees(gross_amount: float) -> BrokerageFees:
    """
    Calculate all fees for selling stocks
    Same calculation as buying for Moroccan market

    Args:
        gross_amount: Total position value (price × quantity) in MAD

    Returns:
        BrokerageFees object with complete fee breakdown
    """
    return calculate_buy_fees(gross_amount)


def calculate_roundtrip_fees(gross_amount: float) -> BrokerageFees:
    """
    Calculate total fees for both buy and sell (roundtrip)

    Args:
        gross_amount: Total position value (same for entry and exit)

    Returns:
        BrokerageFees object with combined fee breakdown
    """
    buy_fees = calculate_buy_fees(gross_amount)
    sell_fees = calculate_sell_fees(gross_amount)

    return BrokerageFees(
        gross_amount=round(gross_amount, 2),
        brokerage_ht=round(buy_fees.brokerage_ht + sell_fees.brokerage_ht, 2),
        settlement_ht=round(buy_fees.settlement_ht + sell_fees.settlement_ht, 2),
        sbvc_ht=round(buy_fees.sbvc_ht + sell_fees.sbvc_ht, 2),
        vat=round(buy_fees.vat + sell_fees.vat, 2),
        total_fees=round(buy_fees.total_fees + sell_fees.total_fees, 2),
        net_amount=round(buy_fees.net_amount + sell_fees.net_amount - gross_amount, 2),
        fees_percent=round(
            (buy_fees.total_fees + sell_fees.total_fees) / gross_amount * 100, 4
        ),
    )


def calculate_break_even_price(
    entry_price: float, position_size: int, target_profit_percent: float = 1.0
) -> float:
    """
    Calculate the minimum exit price needed to break even including all fees

    Args:
        entry_price: Entry price per share
        position_size: Number of shares
        target_profit_percent: Desired profit % after fees (default 1%)

    Returns:
        Minimum exit price to achieve target profit
    """
    gross_entry = entry_price * position_size

    # Buy fees
    buy_fees = calculate_buy_fees(gross_entry)

    # Target gross amount after sell (entry + fees + profit target)
    target_gross = gross_entry * (1 + target_profit_percent / 100)

    # Iterate to find exact break-even considering sell fees
    tolerance = 0.01
    max_iterations = 50
    exit_price = entry_price * (1 + target_profit_percent / 100)

    for _ in range(max_iterations):
        gross_exit = exit_price * position_size
        sell_fees = calculate_sell_fees(gross_exit)

        total_cost = buy_fees.net_amount
        total_revenue = gross_exit - sell_fees.total_fees
        actual_profit = (total_revenue - total_cost) / gross_entry * 100

        if abs(actual_profit - target_profit_percent) < tolerance:
            break

        # Adjust exit price
        adjustment = (target_profit_percent - actual_profit) / 100 * entry_price
        exit_price += adjustment

    return round(exit_price, 2)


def calculate_net_pnl(
    entry_price: float, exit_price: float, position_size: int
) -> Dict:
    """
    Calculate net P&L after all fees

    Args:
        entry_price: Entry price per share
        exit_price: Exit price per share
        position_size: Number of shares

    Returns:
        Dictionary with gross and net P&L details
    """
    gross_entry = entry_price * position_size
    gross_exit = exit_price * position_size

    buy_fees = calculate_buy_fees(gross_entry)
    sell_fees = calculate_sell_fees(gross_exit)

    total_fees = buy_fees.total_fees + sell_fees.total_fees
    gross_pnl = (exit_price - entry_price) * position_size
    net_pnl = gross_pnl - total_fees

    return {
        "gross_entry": round(gross_entry, 2),
        "gross_exit": round(gross_exit, 2),
        "buy_fees": buy_fees.total_fees,
        "sell_fees": sell_fees.total_fees,
        "total_fees": round(total_fees, 2),
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "fees_impact_percent": round(total_fees / gross_entry * 100, 4)
        if gross_entry > 0
        else 0,
    }


def print_fee_example(position_value: float = 10000) -> None:
    """Print a detailed fee calculation example"""
    print("=" * 70)
    print("CSEMA BROKERAGE FEE CALCULATION")
    print("=" * 70)
    print(f"\nPosition Value: {position_value:,.2f} MAD")
    print("-" * 70)

    # Buy fees
    buy_fees = calculate_buy_fees(position_value)
    print("\nBUY SIDE FEES:")
    print(f"  Gross Amount:        {buy_fees.gross_amount:>12,.2f} MAD")
    print(f"  Brokerage (0.60%):   {buy_fees.brokerage_ht:>12,.2f} MAD")
    print(f"  Settlement (0.20%):  {buy_fees.settlement_ht:>12,.2f} MAD")
    print(f"  SBVC (0.10%):        {buy_fees.sbvc_ht:>12,.2f} MAD")
    print(f"  VAT (10%):           {buy_fees.vat:>12,.2f} MAD")
    print(
        f"  Total Buy Fees:      {buy_fees.total_fees:>12,.2f} MAD ({buy_fees.fees_percent:.4f}%)"
    )
    print(f"  Net Buy Amount:      {buy_fees.net_amount:>12,.2f} MAD")

    # Sell fees
    sell_fees = calculate_sell_fees(position_value * 1.05)  # 5% profit
    print("\nSELL SIDE FEES (with 5% gain):")
    print(f"  Gross Amount:        {sell_fees.gross_amount:>12,.2f} MAD")
    print(
        f"  Total Sell Fees:     {sell_fees.total_fees:>12,.2f} MAD ({sell_fees.fees_percent:.4f}%)"
    )
    print(f"  Net Sell Amount:     {sell_fees.net_amount:>12,.2f} MAD")

    # Roundtrip
    roundtrip = calculate_roundtrip_fees(position_value)
    print("\nROUNDTRIP TOTALS:")
    print(f"  Total Fees:          {roundtrip.total_fees:>12,.2f} MAD")
    print(f"  Fees as % of Trade:  {roundtrip.fees_percent:>12,.4f}%")

    # Break-even example
    be_price = calculate_break_even_price(100, 100)
    print(f"\nBREAK-EVEN EXAMPLE:")
    print(f"  Entry: 100 MAD × 100 shares = 10,000 MAD")
    print(f"  Min Exit for 1% profit: {be_price:.2f} MAD")
    print(f"  (Need {(be_price / 100 - 1) * 100:.2f}% gain to net 1%)")

    # Net P&L example
    pnl = calculate_net_pnl(100, 105, 100)  # 5% gain
    print(f"\nNET P&L EXAMPLE (Entry 100, Exit 105, 100 shares):")
    print(f"  Gross P&L:           {pnl['gross_pnl']:>12,.2f} MAD (5.00%)")
    print(f"  Total Fees:          {pnl['total_fees']:>12,.2f} MAD")
    print(f"  Net P&L:             {pnl['net_pnl']:>12,.2f} MAD")
    print(
        f"  Net Return:          {pnl['net_pnl'] / pnl['gross_entry'] * 100:>12,.2f}%"
    )
    print("=" * 70)


if __name__ == "__main__":
    print_fee_example(10000)
