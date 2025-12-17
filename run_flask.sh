#!/bin/bash
# ============================================
# Steel Industry Energy AI - Flask Application
# ============================================
# This script starts the Flask web application
# 
# Usage: ./run_flask.sh
# Then open: http://localhost:5000
# ============================================

echo "================================================"
echo "  Steel Industry Energy AI - Web Application"
echo "================================================"
echo ""

# Change to flask_app directory
cd "$(dirname "$0")/flask_app"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Install dependencies if needed
echo "[1/2] Checking dependencies..."
pip install flask pandas numpy scikit-learn xgboost matplotlib seaborn --quiet --break-system-packages 2>/dev/null || pip install flask pandas numpy scikit-learn xgboost matplotlib seaborn --quiet

echo "[2/2] Starting Flask server..."
echo ""
echo "=============================================="
echo "  Server starting at: http://localhost:5000"
echo "  Press Ctrl+C to stop the server"
echo "=============================================="
echo ""

# Run Flask app
python3 app.py
