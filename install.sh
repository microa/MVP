#!/bin/bash

# Installation script for MVP detector

echo "Installing MVP: Motion Vector Propagation for Zero-Shot Object Detection"
echo "========================================================================"

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "Python version: $python_version ✓"

# Create virtual environment (optional)
read -p "Do you want to create a virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    echo "Creating virtual environment..."
    python3 -m venv mvp_env
    source mvp_env/bin/activate
    echo "Virtual environment created and activated."
fi

# Install requirements
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install motion vector extractor
echo "Installing motion vector extractor..."
cd mv-extractor
pip install -e .
cd ..

# Install MVP package
echo "Installing MVP package..."
pip install -e .

echo ""
echo "Installation completed successfully!"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment (if created): source mvp_env/bin/activate"
echo "2. Run the basic example: python examples/basic_usage.py"
echo "3. Check the README.md for more detailed usage instructions"
echo ""
echo "For evaluation, make sure you have:"
echo "- Video files in your dataset directory"
echo "- Motion vectors extracted using the mv-extractor tool"
echo "- Ground truth annotations in the correct format"
