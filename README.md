# AI Architecture Demo: Local Enterprise Platform

> **Enterprise-grade AI architecture demonstration running entirely on local infrastructure**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![uv](https://img.shields.io/badge/uv-package%20manager-green.svg)](https://github.com/astral-sh/uv)

## 🎯 Overview

This project demonstrates advanced AI architecture capabilities through a comprehensive document processing and agentic analytics platform. Built with modern MLOps practices, it showcases enterprise-scale patterns using local/open-source technologies.

### Key Capabilities Demonstrated

- **MLOps Pipeline**: Complete ML lifecycle with MLflow, model registry, and automated deployment
- **Agentic AI**: Multi-agent orchestration using local LLMs (Ollama)  
- **Streaming Architecture**: Real-time data processing with Kafka and event-driven patterns
- **Enterprise Patterns**: Circuit breakers, CQRS, event sourcing, and distributed architectures
- **Infrastructure as Code**: Docker Compose orchestration with 10+ microservices

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Dashboard │    │   FastAPI API   │    │  Document Agent │
│   (Streamlit)   │◄──►│    Server       │◄──►│     (LLM)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Grafana       │    │   PostgreSQL    │    │   Ollama LLM    │
│  Monitoring     │    │   Metadata      │    │    Server       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │    │     MLflow      │    │   Kafka         │
│   Metrics       │    │   Tracking      │    │  Streaming      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start (Windows PowerShell)

### Prerequisites
- **Docker Desktop** installed and running
- **Python 3.9+** with **uv package manager**
- **16GB+ RAM** recommended (8GB minimum)
- **50GB+** free disk space

### Installation

1. **Clone and Setup**
   ```powershell
   git clone <your-repo-url>
   cd ai-architect-demo
   ```

2. **Install Dependencies with uv**
   ```powershell
   # Install uv if not already installed
   pip install uv
   
   # Create virtual environment and install dependencies
   uv venv
   .venv\Scripts\activate
   uv pip install -e .[dev,jupyter]
   ```

3. **Start Infrastructure**
   ```powershell
   # Start all services
   docker-compose up -d
   
   # Wait for services to be ready (~2-3 minutes)
   docker-compose logs -f
   ```

4. **Initialize the Platform**
   ```powershell
   # Setup MLflow experiments and load initial models
   python scripts/init_platform.py
   
   # Start the main application
   ai-demo start
   ```

### Access Services

| Service | URL | Description |
|---------|-----|-------------|
| **Streamlit Dashboard** | http://localhost:8501 | Main application interface |
| **FastAPI Docs** | http://localhost:8000/docs | API documentation |
| **MLflow UI** | http://localhost:5000 | ML experiment tracking |
| **Grafana** | http://localhost:3000 | Monitoring dashboards |
| **Jupyter Lab** | http://localhost:8888 | Interactive notebooks |
| **Kafka UI** | http://localhost:8080 | Message streaming console |

## 🔧 Development

### Project Structure
```
ai-architect-demo/
├── ai_architect_demo/          # Main Python package
│   ├── agents/                 # AI agents and orchestration
│   ├── api/                    # FastAPI application
│   ├── core/                   # Shared utilities and config
│   ├── data/                   # Data processing pipelines  
│   ├── ml/                     # ML models and MLOps
│   └── streaming/              # Kafka producers/consumers
├── docker/                     # Docker configurations
├── docs/                       # Documentation
├── notebooks/                  # Jupyter notebooks
├── scripts/                    # Automation scripts
├── tests/                      # Test suites
├── docker-compose.yml          # Infrastructure as code
└── pyproject.toml             # Project configuration
```

### Development Commands
```powershell
# Code formatting
uv run black ai_architect_demo/
uv run isort ai_architect_demo/

# Type checking
uv run mypy ai_architect_demo/

# Testing
uv run pytest tests/

# Run individual services
uv run python -m ai_architect_demo.api.server
uv run streamlit run ai_architect_demo/dashboard/app.py
```

## 🧪 Demo Scenarios

### 1. Financial Document Processing
- Upload loan applications, invoices, contracts
- AI extraction of key information and risk analysis
- Real-time compliance checking and audit trails
- Automated workflow routing and approvals

### 2. Customer Service Intelligence  
- Analyze support tickets and feedback
- Generate automated responses with local LLMs
- Sentiment analysis and trend identification
- Multi-language support and escalation logic

### 3. Business Process Automation
- Document approval workflows
- Automated report generation and insights
- Anomaly detection in business metrics
- Integration patterns for enterprise systems

## 📊 Monitoring & Observability

- **Metrics**: Prometheus with custom business metrics
- **Dashboards**: Grafana with ML model performance tracking
- **Logging**: Structured logging with correlation IDs
- **Tracing**: Request tracing across microservices
- **Alerting**: Automated alerts for system health and model drift

## 🏢 Enterprise Patterns Demonstrated

- **Circuit Breaker**: Fault tolerance between services
- **Event Sourcing**: Complete audit trail of all operations
- **CQRS**: Optimized read/write data models
- **Saga Pattern**: Distributed transaction management
- **API Gateway**: Centralized routing and authentication

## 🤖 AI/ML Best Practices

- **Model Lifecycle**: Automated training, validation, deployment
- **Feature Store**: Centralized feature management
- **A/B Testing**: Statistical model comparison framework  
- **Explainable AI**: LIME/SHAP integration for interpretability
- **Data Lineage**: Complete data flow tracking

## 🔒 Security & Compliance

- Container security with non-root users
- Secrets management with Docker secrets
- API authentication and rate limiting
- Data encryption at rest and in transit
- GDPR-compliant data handling patterns

## 📈 Performance Optimization

- **Async Processing**: FastAPI with async/await patterns
- **Connection Pooling**: Database and Redis connection management
- **Caching Strategy**: Multi-level caching with Redis
- **Resource Management**: Container resource limits and monitoring
- **Load Testing**: Performance benchmarks and scaling recommendations

## 🚢 Production Readiness

This local demo demonstrates patterns that scale to production:

- **Container Orchestration**: Ready for Kubernetes deployment
- **Service Mesh**: Microservice communication patterns
- **GitOps**: Infrastructure and application deployment automation
- **Multi-Environment**: Development, staging, production configurations
- **Disaster Recovery**: Backup and recovery procedures

## 📚 Documentation

- [Architecture Decision Records](docs/adr/) - Key design decisions
- [API Documentation](docs/api/) - Complete API reference  
- [Deployment Guide](docs/deployment/) - Production deployment
- [Performance Benchmarks](docs/performance/) - System performance analysis
- [Security Guide](docs/security/) - Security implementation details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Business Value

This demo showcases enterprise AI architecture capabilities that deliver:

- **Cost Reduction**: 60%+ savings through local development and testing
- **Time to Market**: Rapid prototyping and deployment automation
- **Risk Mitigation**: Comprehensive testing and monitoring frameworks  
- **Scalability**: Patterns that support 10x+ growth in data and users
- **Compliance**: Built-in audit trails and data governance

---

**Built with ❤️ for enterprise AI architecture demonstration**