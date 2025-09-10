#!/bin/bash

# Qwen3 Knowledge Distillation Setup Script
# This script helps you set up the virtual environment and dependencies

set -e

echo "🚀 Qwen3 Knowledge Distillation Setup"
echo "======================================"

# Check if Python 3.8+ is available
python_cmd=""
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    python_cmd="python3"
elif command -v python &> /dev/null; then
    python_version=$(python --version 2>&1 | awk '{print $2}')
    python_cmd="python"
else
    echo "❌ Error: Python is not installed or not in PATH"
    exit 1
fi

echo "🐍 Found Python: $python_version"

# Check Python version is 3.8+
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || [ "$python_major" -eq 3 -a "$python_minor" -lt 8 ]; then
    echo "❌ Error: Python 3.8+ required, found $python_version"
    exit 1
fi

echo "✅ Python version is compatible"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment 'venv' already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf venv
    else
        echo "📂 Using existing virtual environment"
    fi
fi

if [ ! -d "venv" ]; then
    $python_cmd -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

# Verify installation
echo "🔍 Verifying installation..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')"
python -c "import peft; print(f'✅ PEFT {peft.__version__}')"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Verify project works:"
echo "     python verify_project.py"
echo ""
echo "  3. Try quick inference test:"
echo "     python test_kd_inference_v2.py --quick_test"
echo ""
echo "  4. Train your own model:"
echo "     python scripts/run_improved_kd.py"
echo ""
echo "💡 Remember to activate the virtual environment before each session!"