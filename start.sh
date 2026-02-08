#!/bin/bash

echo "Setting up MolDelta Demo..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Starting Streamlit app..."
echo "The app will open at: http://localhost:8501"
echo ""

streamlit run app.py
