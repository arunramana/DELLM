#!/bin/bash

echo "=========================================="
echo "Recursive Coding Platform - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "✗ Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Check Docker
echo "Checking Docker..."
docker --version

if [ $? -ne 0 ]; then
    echo "✗ Docker not found. Please install Docker"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create models directory
echo ""
echo "Creating models directory..."
mkdir -p models

# Download model
echo ""
echo "=========================================="
echo "MODEL DOWNLOAD"
echo "=========================================="
echo "The model is ~1.5GB and will be downloaded to models/"
echo "This may take several minutes depending on your connection."
echo ""
read -p "Download Qwen2.5-1.5B model now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading model..."
    wget -q --show-progress https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf -O models/qwen2.5-1.5b-instruct-q8_0.gguf
    
    if [ $? -eq 0 ]; then
        echo "✓ Model downloaded successfully"
    else
        echo "✗ Model download failed"
        echo "You can download manually from:"
        echo "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    fi
else
    echo "Skipping model download."
    echo "Download manually from:"
    echo "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    echo "Place the file in: models/qwen2.5-1.5b-instruct-q8_0.gguf"
fi

echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "To start the network:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Launch nodes: python network.py"
echo "  3. In another terminal, submit tasks: python submit_task.py \"<task>\""
echo ""
echo "See README.md for more details"
