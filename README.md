# 🤖 AutoML Builder

An intelligent, production-ready AutoML platform powered by LangGraph multi-agent architecture and OpenAI. Build, train, and optimize machine learning models through natural language conversations.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-latest-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

### 🎯 Core Capabilities
- **Conversational ML**: Build ML models through natural language interactions
- **Multi-Agent System**: Specialized AI agents for different ML tasks
- **Auto Mode**: Fully automated workflow from data to deployed model
- **Interactive Mode**: Human-in-the-loop for critical decisions
- **Real-time Debug Console**: Monitor agent activities and LLM calls

### 🛠️ Technical Features
- **Intelligent Data Analysis**: Automatic EDA and problem type detection
- **Smart Preprocessing**: Automated data cleaning and feature engineering
- **Model Training**: Multiple algorithms with cross-validation
- **Hyperparameter Optimization**: Optuna integration for tuning
- **MLflow Integration**: Comprehensive experiment tracking
- **Production Ready**: Docker, authentication, monitoring, and scaling

## 🏗️ Architecture

### Multi-Agent System
```
┌─────────────────────────────────────────────────┐
│              Supervisor Agent                    │
│         (Orchestration & Routing)                │
└────────────────┬────────────────────────────────┘
                 │
    ┌────────────┴────────────┬─────────────┬──────────────┐
    │                         │             │              │
┌───▼────────┐   ┌───────────▼──┐   ┌─────▼─────┐   ┌────▼──────┐
│  Analysis  │   │ Preprocessing │   │   Model   │   │Optimization│
│   Agent    │   │    Agent      │   │   Agent   │   │   Agent    │
└────────────┘   └───────────────┘   └───────────┘   └────────────┘
```

### Tech Stack
- **Backend**: FastAPI, LangGraph, LangChain, SQLAlchemy
- **Frontend**: Streamlit
- **ML**: Scikit-learn, XGBoost, Optuna
- **Infrastructure**: Docker, PostgreSQL, Redis, MLflow
- **LLM**: OpenAI GPT-4

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- OpenAI API Key

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/automl-builder.git
cd automl-builder
```

### 2. Set Up Environment
```bash
# Copy environment variables
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your-api-key-here
```

### 3. Start Services
```bash
# Using Make (recommended)
make setup
make docker-up
make dev

# Or manually
chmod +x scripts/setup_local.sh
./scripts/setup_local.sh
```

### 4. Access the Application
- **Streamlit UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000

## 📖 Usage Guide

### 1. Upload Dataset
Navigate to the Chat page and upload your dataset (CSV, Excel, or JSON).

### 2. Describe Your Goal
Tell the AI what you want to predict:
```
"I want to predict customer churn based on the provided features"
```

### 3. Review Analysis
The AI will analyze your data and identify:
- Problem type (classification/regression)
- Data quality issues
- Feature importance
- Preprocessing recommendations

### 4. Approve Steps (Interactive Mode)
Review and approve each preprocessing step:
- Missing value handling
- Feature encoding
- Outlier treatment
- Feature scaling

### 5. Model Training
Multiple models are automatically trained:
- Random Forest
- XGBoost
- Logistic Regression / Linear Regression
- Support Vector Machines
- Gradient Boosting

### 6. Optimization
Hyperparameters are tuned using Optuna with:
- Bayesian optimization
- Cross-validation
- Early stopping

### 7. Results & Export
- View model comparisons
- Download trained models
- Export predictions
- Access MLflow for detailed tracking

## 🔧 Configuration

### Workflow Modes
- **Auto Mode**: Fully automated pipeline
- **Interactive Mode**: Approve each major step
- **Debug Mode**: See all agent activities

### Environment Variables
Key configurations in `.env`:
```bash
# API Configuration
API_SECRET_KEY=your-secret-key
API_ENVIRONMENT=development

# OpenAI Configuration
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4

# Feature Flags
ENABLE_DEBUG_MODE=true
ENABLE_AUTO_MODE=true
MAX_UPLOAD_SIZE_MB=100
```

## 📊 API Endpoints

### Authentication
```
POST   /api/auth/login          # Email login
POST   /api/auth/logout         # Logout
GET    /api/auth/me            # Current user
```

### Chat & Workflow
```
POST   /api/chat/message        # Send message
GET    /api/chat/sessions       # List sessions
POST   /api/chat/approve        # Approve action
```

### Datasets
```
POST   /api/datasets/upload     # Upload dataset
GET    /api/datasets/           # List datasets
GET    /api/datasets/{id}       # Dataset details
```

### Experiments
```
GET    /api/experiments/        # List experiments
GET    /api/experiments/{id}    # Experiment details
POST   /api/experiments/compare # Compare models
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test suites
make test-unit
make test-integration
make test-e2e
```

## 📦 Deployment

### Local Deployment
```bash
make docker-up
```

### EC2 Deployment
```bash
./scripts/deploy_ec2.sh
```

### Production Considerations
- Use environment-specific configs
- Enable SSL/TLS
- Set up monitoring (Prometheus/Grafana)
- Configure backup strategies
- Implement rate limiting

## 🛡️ Security

- OAuth 2.0 authentication (Google, GitHub)
- JWT token-based sessions
- Input validation and sanitization
- Rate limiting
- Secure file upload handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain & LangGraph teams for the amazing framework
- OpenAI for GPT-4 API
- MLflow for experiment tracking
- Streamlit for the intuitive UI framework

## 📞 Support

- Documentation: [docs.automl-builder.com](https://docs.automl-builder.com)
- Issues: [GitHub Issues](https://github.com/yourusername/automl-builder/issues)
- Email: support@automl-builder.com

---

Built with ❤️ by the AutoML Builder Team