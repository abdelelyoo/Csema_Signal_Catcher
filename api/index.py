import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import app, handle missing dependencies gracefully
try:
    from app import app
except ImportError as e:
    print(f"Import error: {e}")
    # Create minimal Flask app to show error
    from flask import Flask, jsonify

    app = Flask(__name__)

    @app.route("/")
    def error_page():
        return jsonify(
            {
                "error": "Missing dependencies",
                "message": f"{e}. This app requires pandas, numpy, tvscreener, and tvdatafeed.",
            }
        ), 500


# Vercel needs the app variable
application = app
