#!/bin/bash
# scripts/setup_local.sh

set -e

echo "🚀 AutoML Builder - Local Development Setup"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{uploads,processed,artifacts}
mkdir -p mlruns
mkdir -p logs

# Create .env file from example
if [ ! -f .env ]; then
    echo "📄 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please update .env with your OpenAI API key and other credentials"
fi

# Create PostgreSQL initialization script
echo "📝 Creating database initialization script..."
cat > docker/init.sql << 'EOF'
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tables
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    oauth_provider VARCHAR(50),
    oauth_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    dataset_id UUID,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT,
    rows_count INTEGER,
    columns_count INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES chat_sessions(id),
    mlflow_run_id VARCHAR(255),
    status VARCHAR(50),
    results JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX idx_datasets_user_id ON datasets(user_id);
CREATE INDEX idx_experiments_session_id ON experiments(session_id);
EOF

# Start services with Docker Compose
echo "🐳 Starting Docker services..."
cd docker
docker-compose down -v  # Clean any existing containers
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🔍 Checking service status..."
docker-compose ps

# Verify PostgreSQL connection
echo "🗄️  Verifying PostgreSQL connection..."
docker exec automl_postgres pg_isready -U automl_user

# Verify Redis connection
echo "🔴 Verifying Redis connection..."
docker exec automl_redis redis-cli ping

# Install Python dependencies (optional - for local development)
if command -v python3 &> /dev/null; then
    echo "🐍 Setting up Python environment..."
    python3 -m venv venv
    source venv/bin/activate || . venv/Scripts/activate
    pip install -r ../requirements/dev.txt
fi

echo "✅ Setup complete!"
echo ""
echo "Services running:"
echo "  - PostgreSQL: localhost:5432"
echo "  - Redis: localhost:6379"
echo "  - MLflow: http://localhost:5000"
echo ""
echo "Next steps:"
echo "1. Update .env file with your OpenAI API key"
echo "2. Run 'make dev' to start the API and frontend"
echo "3. Access the application at http://localhost:8501"
echo ""
echo "To stop services: cd docker && docker-compose down"
echo "To view logs: cd docker && docker-compose logs -f"