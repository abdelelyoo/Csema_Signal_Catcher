import os
import sys

# Ensure dependencies are installed before importing
if not os.path.exists("/tmp/.deps_installed"):
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "pandas", "numpy", "--quiet"]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/deepentropy/tvscreener.git",
            "--quiet",
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/rongardF/tvdatafeed.git",
            "--quiet",
        ]
    )
    os.makedirs("/tmp/.deps_installed", exist_ok=True)

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app as application

app = application
