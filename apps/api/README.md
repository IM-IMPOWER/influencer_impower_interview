# KOL Platform API

FastAPI backend for the KOL (Key Opinion Leader) platform with GraphQL gateway, REST endpoints, and AI-powered matching algorithms.

## 🚀 Features

### Core POCs Implementation

- **POC1**: Scalable KOL discovery with Go scraper integration
- **POC2**: AI-powered multi-factor KOL matching and scoring
- **POC3**: Campaign workflow and KOL collaboration management  
- **POC4**: Algorithmic budget optimization and allocation

### Technology Stack

- **FastAPI**: Modern, fast web framework for APIs
- **PostgreSQL + pgvector**: Vector database for semantic search
- **Redis**: Caching and background task queue
- **GraphQL**: Primary API interface via Strawberry GraphQL
- **Celery**: Background task processing
- **SQLAlchemy**: Async ORM with full type hints

### Key Features

- Multi-factor KOL scoring algorithm
- Vector similarity search for content matching
- Budget optimization using algorithmic approaches
- Real-time performance analytics
- Comprehensive audit logging
- Rate limiting and security middleware
- Docker containerization ready

## 📋 Prerequisites

- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- Redis 7+
- Docker and Docker Compose (for containerized deployment)

## 🛠️ Installation

### Local Development Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd apps/api
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -e .
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Set up database**
```bash
# Start PostgreSQL with pgvector
docker run --name postgres-kol \
  -e POSTGRES_DB=kol_platform \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  -d pgvector/pgvector:pg15

# Run migrations
alembic upgrade head
```

6. **Start Redis**
```bash
docker run --name redis-kol -p 6379:6379 -d redis:7-alpine
```

7. **Run the application**
```bash
uvicorn kol_api.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in background
docker-compose up -d
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:port/db

# Redis
REDIS_URL=redis://localhost:6379/0

# Authentication
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# External Services
GO_SCRAPER_SERVICE_URL=http://localhost:8080
```

### Scoring Weights (POC2)

Configure multi-factor scoring weights:

```json
{
  "engagement_rate": 0.3,
  "follower_quality": 0.25, 
  "content_relevance": 0.2,
  "brand_safety": 0.15,
  "posting_consistency": 0.1
}
```

### Budget Tiers (POC4)

Configure KOL tier pricing:

```json
{
  "nano": {"min_followers": 1000, "max_followers": 10000, "avg_cost_per_post": 50},
  "micro": {"min_followers": 10000, "max_followers": 100000, "avg_cost_per_post": 500}
}
```

## 📊 API Documentation

### GraphQL Endpoint

- **URL**: `http://localhost:8000/api/graphql`
- **Playground**: Available in development mode
- **Schema**: Auto-generated from Strawberry types

### REST Endpoints

- **Health Check**: `GET /api/health`
- **Authentication**: `POST /api/v1/auth/login`
- **File Upload**: `POST /api/v1/upload`
- **Documentation**: `http://localhost:8000/docs`

### Key GraphQL Operations

#### Query KOLs with Filters
```graphql
query GetKOLs($filters: KOLFilterInput) {
  kols(filters: $filters, limit: 20) {
    id
    username
    displayName
    platform
    tier
    primaryCategory
    metrics {
      followerCount
      engagementRate
    }
    score {
      overallScore
      brandSafetyScore
    }
  }
}
```

#### AI-Powered KOL Matching
```graphql
query MatchKOLs($campaignId: String!) {
  matchKolsForCampaign(campaignId: $campaignId, useAiScoring: true) {
    matchedKols {
      id
      username
      score {
        overallScore
      }
    }
    matchCriteria
    similarityScores
  }
}
```

#### Budget Optimization
```graphql
query OptimizeBudget($campaignId: String!, $objective: String!) {
  optimizeBudget(campaignId: $campaignId, optimizationObjective: $objective) {
    optimizedPlan {
      id
      totalBudget
      predictedReach
      allocations {
        allocationName
        allocatedAmount
        expectedReach
      }
    }
    optimizationMetadata
  }
}
```

## 🎯 POC Implementation Details

### POC1: KOL Discovery at Scale

- **Go Service Integration**: REST endpoints to trigger data collection
- **Batch Processing**: Celery tasks for large-scale data ingestion  
- **Data Quality Scoring**: Automated assessment of scraped data
- **Real-time Monitoring**: Health checks and progress tracking

### POC2: AI-Powered KOL Matching

- **Multi-Factor Scoring**: Engagement rate, follower quality, content relevance, brand safety, posting consistency
- **Vector Similarity**: pgvector for semantic content matching
- **Campaign Context**: Audience matching and cost efficiency analysis
- **Machine Learning Ready**: Extensible scoring framework

### POC3: Campaign Workflow (Foundation)

- **Campaign Management**: Create, manage, and track campaigns
- **KOL Collaboration**: Invitation, negotiation, and content approval workflow
- **Communication Tracking**: Audit trail for all interactions
- **Performance Monitoring**: Real-time campaign metrics

### POC4: Budget Optimization

- **Algorithmic Allocation**: Greedy and linear programming approaches
- **Multi-Objective Optimization**: Maximize reach, engagement, or conversions
- **Constraint Handling**: Tier requirements, budget limits, risk tolerance
- **Scenario Analysis**: Alternative budget allocations

## 🔐 Security Features

- **JWT Authentication**: HTTP-only cookies and Bearer tokens
- **Rate Limiting**: Redis-backed distributed rate limiting
- **Input Validation**: Pydantic models with comprehensive validation
- **Audit Logging**: Structured logging with user context
- **CORS Protection**: Configurable origin restrictions
- **SQL Injection Prevention**: Parameterized queries via SQLAlchemy

## 📈 Monitoring & Observability

- **Health Checks**: Kubernetes/Docker-ready endpoints
- **Metrics**: Prometheus-compatible metrics export
- **Structured Logging**: JSON logging with correlation IDs
- **Performance Tracking**: Request timing and database query metrics
- **Error Tracking**: Comprehensive error logging and reporting

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests  
pytest tests/integration/

# Load tests
pytest tests/load/

# Coverage report
pytest --cov=kol_api --cov-report=html
```

### Database Migrations

```bash
# Generate new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## 🚀 Deployment

### Production Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Configure secure `SECRET_KEY`
- [ ] Set up PostgreSQL with pgvector
- [ ] Configure Redis cluster
- [ ] Set up SSL certificates
- [ ] Configure monitoring and logging
- [ ] Set up backup procedures
- [ ] Configure rate limiting
- [ ] Test disaster recovery

### Scaling Considerations

- **Database**: Read replicas for query scaling
- **Application**: Horizontal scaling with load balancer
- **Background Tasks**: Multiple Celery workers
- **Caching**: Redis cluster for high availability
- **Vector Search**: Consider specialized vector databases for scale

## 🤝 Integration Points

### Go Scraper Service

The API integrates with a Go-based scraper service for data collection:

```bash
# Trigger KOL data refresh
POST /api/v1/kols/refresh
{
  "platform": "tiktok",
  "kol_ids": ["123", "456"]
}
```

### Frontend Integration

Designed to work with React/Next.js frontend:

- GraphQL subscriptions for real-time updates
- File upload endpoints for media assets
- Authentication flow with HTTP-only cookies
- CORS configured for development and production

## 📝 Development Notes

### Database Schema

- **Users**: Authentication and authorization
- **KOLs**: Influencer profiles and metrics
- **Campaigns**: Campaign management and requirements
- **Budget Plans**: Optimization results and allocations
- **Scores**: Multi-factor scoring history

### Vector Embeddings

- **Content Embeddings**: Sentence transformers for semantic search
- **Similarity Search**: Cosine similarity via pgvector
- **Embedding Models**: Configurable transformer models

### Background Tasks

- **Data Refresh**: Periodic KOL data updates
- **Score Calculation**: Batch KOL scoring jobs
- **Performance Analytics**: Aggregation and reporting
- **Cleanup Tasks**: Data retention and archival

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection**: Ensure PostgreSQL is running with pgvector extension
2. **Redis Connection**: Verify Redis is accessible for caching and tasks
3. **Memory Usage**: Monitor memory usage with large vector operations
4. **Performance**: Check database indexes and query optimization

### Debugging

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Database query logging
export DATABASE_ECHO=true

# Check service health
curl http://localhost:8000/api/health/ready
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions and support:

- **Documentation**: `/docs` endpoint for API documentation
- **GraphQL Playground**: `/api/graphql` for interactive queries
- **Health Monitoring**: `/api/health` for system status

---

**Note**: This is a proof-of-concept implementation focused on demonstrating core KOL platform capabilities. Production deployment should include additional security hardening, monitoring, and scalability considerations.