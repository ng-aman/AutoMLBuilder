#!/bin/bash
# scripts/run_local.sh
# Script to run AutoML Builder locally

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "🚀 Starting AutoML Builder..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found!${NC}"
    echo "Please run: ./scripts/setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${RED}.env file not found!${NC}"
    echo "Please copy .env.example to .env and configure it"
    exit 1
fi

# Export environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-openai-api-key" ]; then
    echo -e "${YELLOW}Warning: OPENAI_API_KEY not configured in .env${NC}"
    echo "The application will not work properly without it"
    echo ""
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    
    # Kill API server
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
    fi
    
    # Kill Streamlit
    if [ ! -z "$STREAMLIT_PID" ]; then
        kill $STREAMLIT_PID 2>/dev/null || true
    fi
    
    # Kill MLflow
    if [ ! -z "$MLFLOW_PID" ]; then
        kill $MLFLOW_PID 2>/dev/null || true
    fi
    
    echo "Services stopped"
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Check if Docker services are running
echo "Checking Docker services..."
if ! docker ps | grep -q automl_postgres; then
    echo -e "${YELLOW}PostgreSQL not running. Starting Docker services...${NC}"
    cd docker && docker-compose up -d postgres redis && cd ..
    sleep 5
fi

# Start MLflow server
echo ""
echo "Starting MLflow server..."
mlflow server \
    --backend-store-uri $DATABASE_URL \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000 \
    > logs/mlflow.log 2>&1 &
MLFLOW_PID=$!
echo -e "${GREEN}✓${NC} MLflow started (PID: $MLFLOW_PID)"

# Wait for MLflow to start
sleep 3

# Start API server
echo ""
echo "Starting API server..."
python -m uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-config configs/logging.yaml \
    > logs/api.log 2>&1 &
API_PID=$!
echo -e "${GREEN}✓${NC} API server started (PID: $API_PID)"

# Wait for API to start
sleep 3

# Start Streamlit
echo ""
echo "Starting Streamlit UI..."
streamlit run src/frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --theme.base dark \
    > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo -e "${GREEN}✓${NC} Streamlit started (PID: $STREAMLIT_PID)"

# Wait a bit for everything to start
sleep 3

# Display access information
echo ""
echo "======================================"
echo -e "${GREEN}✅ AutoML Builder is running!${NC}"
echo "======================================"
echo ""
echo "Access the services at:"
echo "  • Streamlit UI:  http://localhost:8501"
echo "  • API Docs:      http://localhost:8000/docs"
echo "  • MLflow UI:     http://localhost:5000"
echo ""
echo "Logs are available in:"
echo "  • API:       logs/api.log"
echo "  • Streamlit: logs/streamlit.log"
echo "  • MLflow:    logs/mlflow.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Keep script running
while true; do
    sleep 1
    
    # Check if services are still running
    if ! kill -0 $API_PID 2>/dev/null; then
        echo -e "${RED}API server stopped unexpectedly${NC}"
        exit 1
    fi
    
    if ! kill -0 $STREAMLIT_PID 2>/dev/null; then
        echo -e "${RED}Streamlit stopped unexpectedly${NC}"
        exit 1
    fi
done