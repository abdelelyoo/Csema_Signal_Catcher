#!/bin/bash
echo "Installing dependencies..."
pip install pandas numpy python-dotenv
pip install git+https://github.com/deepentropy/tvscreener.git
pip install git+https://github.com/rongardF/tvdatafeed.git
echo "Dependencies installed successfully"
