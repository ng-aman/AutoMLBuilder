# Makefile for AutoML Builder

.PHONY: help setup dev test clean docker-up docker-down format lint

# Default target
help:
	@echo "AutoML Builder - Available Commands"
	@echo "=================================="
	@echo "make setup      - Set up local development environment"
	@echo "make dev        - Run development servers (API + Frontend)"
	@echo "make test       - Run test suite"
	@echo "make format     - Format code with black and isort"
	@echo "make lint       - Run linting checks"
	@echo "make docker-up  - Start Docker services"
	@echo "make docker-down - Stop Docker services"
	@echo "make clean      - Clean up generated files and caches"

# Setup local development environment
setup:
	@echo "Setting up local development environment..."
	@chmod +x scripts/setup_local.sh
	@./scripts/setup_local.sh

# Run development servers
dev:
	@echo "Starting development servers..."
	@if [ ! -f venv/bin/activate ]; then \
		echo "Virtual environment not found. Running setup..."; \
		make setup; \
	fi
	@echo "Starting API server..."
	@. venv/bin/activate && python src/api/main.py &
	@echo "Starting Streamlit frontend..."
	@. venv/bin/activate && streamlit run src/frontend/app.py

# Run tests
test:
	@echo "Running tests..."
	@. venv/bin/activate && pytest tests/ -v --cov=src --cov-report=html

# Run specific test categories
test-unit:
	@. venv/bin/activate && pytest tests/unit/ -v

test-integration:
	@. venv/bin/activate && pytest tests/integration/ -v

test-e2e:
	@. venv/bin/activate && pytest tests/e2e/ -v

# Code formatting
format:
	@echo "Formatting code..."
	@. venv/bin/activate && black src/ tests/
	@. venv/bin/activate && isort src/ tests/

# Linting
lint:
	@echo "Running linting checks..."
	@. venv/bin/activate && flake8 src/ tests/
	@. venv/bin/activate && mypy src/

# Docker commands
docker-up:
	@echo "Starting Docker services..."
	@cd docker && docker-compose up -d

docker-down:
	@echo "Stopping Docker services..."
	@cd docker && docker-compose down

docker-logs:
	@cd docker && docker-compose logs -f

docker-restart:
	@make docker-down
	@make docker-up

# Database operations
db-migrate:
	@echo "Running database migrations..."
	@. venv/bin/activate && alembic upgrade head

db-reset:
	@echo "Resetting database..."
	@cd docker && docker-compose exec postgres psql -U automl_user -d automl -f /docker-entrypoint-initdb.d/init.sql

# Clean up
clean:
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@rm -rf .pytest_cache
	@rm -rf htmlcov
	@rm -rf .coverage
	@rm -rf .mypy_cache
	@echo "Clean complete!"

# Development utilities
shell-api:
	@docker exec -it automl_api /bin/bash

shell-postgres:
	@docker exec -it automl_postgres psql -U automl_user -d automl

shell-redis:
	@docker exec -it automl_redis redis-cli

# MLflow UI
mlflow-ui:
	@echo "Opening MLflow UI..."
	@open http://localhost:5000 || xdg-open http://localhost:5000

# Install pre-commit hooks
install-hooks:
	@. venv/bin/activate && pre-commit install

# Run pre-commit on all files
pre-commit:
	@. venv/bin/activate && pre-commit run --all-files