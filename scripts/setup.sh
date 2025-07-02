#!/bin/bash
# scripts/setup.sh
# Quick setup script for AutoML Builder

set -e

echo "🤖 AutoML Builder Setup"
echo "======================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if (( $(echo "$PYTHON_VERSION >= 3.10" | bc -l) )); then
        print_status "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.10+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check Docker
echo ""
echo "Checking Docker..."
if command -v docker &> /dev/null; then
    print_status "Docker found"
else
    print_error "Docker not found. Please install Docker"
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
echo ""
echo "Checking Docker Compose..."
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    print_status "Docker Compose found"
else
    print_error "Docker Compose not found. Please install Docker Compose"
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Create directories
echo ""
echo "Creating project directories..."
mkdir -p data/{uploads,processed,artifacts}
mkdir -p logs
mkdir -p mlruns
mkdir -p configs
print_status "Directories created"

# Create .env file if it doesn't exist
echo ""
if [ ! -f .env ]; then
    echo "Creating .env file..."
    if [ -f .env.example ]; then
        cp .env.example .env
        print_status ".env file created from .env.example"
        print_warning "Please update .env with your OpenAI API key and other credentials"
    else
        print_error ".env.example not found"
        exit 1
    fi
else
    print_status ".env file already exists"
fi

# Create virtual environment
echo ""
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate || . venv/Scripts/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
print_status "pip upgraded"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
if [ -f "requirements/dev.txt" ]; then
    pip install -r requirements/dev.txt
    print_status "Python dependencies installed"
else
    print_error "requirements/dev.txt not found"
    exit 1
fi

# Create .gitkeep files
touch data/uploads/.gitkeep
touch data/processed/.gitkeep
touch data/artifacts/.gitkeep

# Setup complete
echo ""
echo "======================================"
echo -e "${GREEN}✅ Setup completed successfully!${NC}"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Update .env file with your OpenAI API key"
echo "   - Edit: .env"
echo "   - Set: OPENAI_API_KEY=your-api-key-here"
echo ""
echo "2. Start the services:"
echo "   - Run: make docker-up"
echo "   - Run: make dev"
echo ""
echo "3. Access the application:"
echo "   - Streamlit UI: http://localhost:8501"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - MLflow UI: http://localhost:5000"
echo ""
echo "For more information, see README.md"
echo ""