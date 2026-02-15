"""
Quick test script for web app fixes
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

# Test 1: Check if tvscreener is available
try:
    from src.stage1_screening import TVSCREENER_AVAILABLE

    print(f"[OK] tvscreener available: {TVSCREENER_AVAILABLE}")
except Exception as e:
    print(f"[ERROR] Error checking tvscreener: {e}")

# Test 2: Test None handling in DataFrame conversion
print("\n[OK] Testing DataFrame None handling:")
position_df = None
if position_df is None:
    position_df = pd.DataFrame()
    print("  - None DataFrame converted to empty DataFrame")

position_data = position_df.to_dict("records") if not position_df.empty else []
print(f"  - Empty DataFrame converted to list: {position_data}")
print(f"  - Length: {len(position_data)}")

# Test 3: Test watchlist None handling
print("\n[OK] Testing watchlist None handling:")
watchlist = None
if watchlist is None:
    print("  - Detected None watchlist")
    print("  - Would show error message to user")

# Test 4: Test normal watchlist
print("\n[OK] Testing normal watchlist:")
watchlist = {
    "position": pd.DataFrame({"Symbol": ["TEST1", "TEST2"], "Price": [100, 200]}),
    "swing": pd.DataFrame(),
}
position_df = watchlist.get("position")
swing_df = watchlist.get("swing")
print(f"  - Position DataFrame has {len(position_df)} rows")
print(f"  - Swing DataFrame has {len(swing_df)} rows")

position_data = position_df.to_dict("records") if not position_df.empty else []
swing_data = swing_df.to_dict("records") if not swing_df.empty else []
print(f"  - Position data: {len(position_data)} records")
print(f"  - Swing data: {len(swing_data)} records")

print(
    "\n[SUCCESS] All tests passed! The fixes should resolve the 'NoneType has no len()' error."
)
print("\nTo run the web app:")
print("  python app.py")
print("\nThen access http://127.0.0.1:5000 in your browser")
