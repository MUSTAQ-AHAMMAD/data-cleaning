#!/bin/bash

# AI CRM Data Cleaning - Startup Script
# This script starts the Streamlit application

echo "=================================================="
echo "   AI CRM - Data Cleaning System"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo "Please install Python 3.8 or higher."
    exit 1
fi

# Recommend virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Recommendation: Use a virtual environment"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo ""
    read -p "Continue without virtual environment? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Starting the application..."
echo ""

# Check if dependencies are installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Start Streamlit
echo "Starting Streamlit application..."
echo "The application will open in your browser automatically."
echo ""
echo "If it doesn't open automatically, navigate to:"
echo "  http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application."
echo ""

streamlit run app.py
