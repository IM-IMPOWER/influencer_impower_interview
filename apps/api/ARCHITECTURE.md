# KOL Platform FastAPI Architecture

## 🏗️ Architecture Overview

This document describes the complete FastAPI backend architecture for the KOL platform, designed to replace tRPC and provide scalable, AI-powered influencer campaign management.

## 📁 Project Structure

```
apps/api/
├── src/kol_api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py              # Application configuration
│   │
│   ├── database/
│   │   ├── connection.py      # Database session management
│   │   └── models/            # SQLAlchemy models
│   │       ├── __init__.py
│   │       ├── base.py        # Base model classes
│   │       ├── auth.py        # User authentication models
│   │       ├── kol.py         # KOL and metrics models
│   │       ├── campaign.py    # Campaign management models
│   │       ├── budget.py      # Budget optimization models
│   │       └── scoring.py     # Multi-factor scoring models
│   │
│   ├── graphql/
│   │   ├── schema.py          # Main GraphQL schema
│   │   ├── types.py           # GraphQL type definitions
│   │   └── resolvers/         # GraphQL resolvers
│   │       ├── base.py        # Base resolver functionality
│   │       ├── kol_resolvers.py
│   │       ├── campaign_resolvers.py
│   │       ├── budget_resolvers.py
│   │       └── scoring_resolvers.py
│   │
│   ├── routers/               # REST API endpoints
│   │   ├── __init__.py
│   │   ├── auth.py           # Authentication endpoints
│   │   ├── health.py         # Health check endpoints
│   │   ├── kols.py           # KOL management endpoints
│   │   ├── campaigns.py      # Campaign endpoints
│   │   ├── budget_optimizer.py # Budget optimization endpoints
│   │   ├── scoring.py        # Scoring endpoints
│   │   └── upload.py         # File upload endpoints
│   │
│   ├── services/              # Business logic services
│   │   ├── kol_matching.py   # POC2: AI-powered KOL matching
│   │   └── budget_optimizer.py # POC4: Budget optimization
│   │
│   └── middleware/            # Custom middleware
│       ├── auth.py           # JWT authentication middleware
│       ├── logging.py        # Request/response logging
│       └── rate_limit.py     # Redis-based rate limiting
│
├── migrations/                # Alembic database migrations
├── tests/                    # Test suites
├── docker-compose.yml        # Development environment
├── Dockerfile               # Production container
├── pyproject.toml           # Python dependencies
├── alembic.ini             # Migration configuration
└── README.md               # Documentation
```

## 🎯 POC Implementation Status

### ✅ Completed Architecture Components

1. **Database Models**: Complete schema design for all POCs
   - **KOL Models**: KOL profiles, metrics, content, with pgvector embeddings
   - **Campaign Models**: Campaign management, briefs, KOL collaborations
   - **Budget Models**: Optimization plans, allocations, constraints
   - **Scoring Models**: Multi-factor scores, history tracking
   - **Auth Models**: User management with role-based access

2. **FastAPI Application Structure**
   - **Main App**: Complete setup with middleware, CORS, monitoring
   - **GraphQL Gateway**: Strawberry GraphQL with comprehensive schema
   - **REST Endpoints**: Specialized endpoints for file uploads, health checks
   - **Authentication**: JWT-based auth with HTTP-only cookies

3. **Core Services**
   - **KOL Matching Service**: Multi-factor scoring algorithm for POC2
   - **Budget Optimizer Service**: Algorithmic allocation for POC4
   - **Vector Similarity**: pgvector integration for semantic search

4. **Infrastructure**
   - **Docker Configuration**: Multi-stage builds, development compose
   - **Database Setup**: PostgreSQL with pgvector, Redis caching
   - **Monitoring**: Health checks, structured logging, metrics
   - **Security**: Rate limiting, input validation, audit trails

### 🚧 Implementation Required (Next Steps)

#### POC1: Scalable KOL Discovery
- [ ] **Go Scraper Integration**: HTTP client for data collection service
- [ ] **Background Tasks**: Celery workers for data ingestion
- [ ] **Data Quality Assessment**: Automated scoring of scraped data
- [ ] **Batch Processing**: Large-scale data import workflows

#### POC2: AI-Powered KOL Matching
- [ ] **Sentence Transformers**: Content embedding generation
- [ ] **Similarity Search**: Vector database queries optimization
- [ ] **Score Calculation**: Complete multi-factor algorithm implementation
- [ ] **Machine Learning**: Model training for improved scoring

#### POC3: Campaign Workflow
- [ ] **Communication System**: Notification and messaging integration
- [ ] **Content Approval**: Workflow management for deliverables
- [ ] **Performance Tracking**: Real-time metrics collection
- [ ] **Payment Processing**: Integration with payment systems

#### POC4: Budget Optimization
- [ ] **Advanced Algorithms**: Linear programming, genetic algorithms
- [ ] **Constraint Handling**: Complex business rule implementation
- [ ] **Scenario Analysis**: Multi-budget comparison tools
- [ ] **ROI Prediction**: Machine learning models for performance prediction

## 🔧 Technical Architecture

### Database Design

#### Core Tables
```sql
-- KOL Management
kols                    # Main KOL profiles
kol_profiles           # Extended profile data
kol_metrics            # Performance metrics
kol_content            # Sample content for analysis
kol_scores             # Multi-factor scoring results

-- Campaign Management
campaigns              # Campaign definitions
campaign_briefs        # Detailed requirements
campaign_kols          # KOL-campaign relationships

-- Budget Optimization
budget_plans           # Optimization results
budget_allocations     # Individual allocations

-- System
users                  # Authentication
score_history          # Scoring trends
```

#### Key Indexes
```sql
-- Performance indexes
CREATE INDEX idx_kol_tier_category ON kols (tier, primary_category);
CREATE INDEX idx_kol_metrics_engagement ON kol_metrics (engagement_rate, follower_count);
CREATE INDEX idx_kol_brand_safe ON kols (is_brand_safe, is_active);

-- Vector similarity
CREATE INDEX idx_kol_content_embedding ON kols USING gin (content_embedding);
```

### GraphQL Schema Design

#### Query Operations
```graphql
# KOL Discovery and Matching
kols(filters: KOLFilterInput): [KOL!]!
matchKolsForCampaign(campaignId: ID!): KOLMatchingResult!
similarKols(kolId: ID!): [KOL!]!

# Budget Optimization  
optimizeBudget(campaignId: ID!, objective: String!): BudgetOptimizationResult!
budgetPlans(campaignId: ID): [BudgetPlan!]!

# Analytics
kolPerformanceAnalytics(kolIds: [ID!]!): AnalyticsResult!
campaignPerformanceSummary(campaignId: ID!): PerformanceSummary!
```

#### Mutation Operations
```graphql
# Campaign Management
createCampaign(input: CampaignCreateInput!): OperationResult!
inviteKolToCampaign(campaignId: ID!, kolId: ID!): OperationResult!

# Budget Planning
createBudgetPlan(input: BudgetPlanCreateInput!): OperationResult!
approveBudgetPlan(planId: ID!): OperationResult!

# Scoring
rescoreKol(kolId: ID!, campaignId: ID): OperationResult!
bulkRescoreKols(kolIds: [ID!]!): OperationResult!
```

### Service Architecture

#### KOL Matching Service (POC2)
```python
class KOLMatchingService:
    async def find_matching_kols(campaign_id, filters) -> List[KOL]
    async def calculate_kol_score_for_campaign(kol, campaign) -> Decimal
    async def find_similar_kols(reference_kol_id) -> List[KOL]
```

**Multi-Factor Scoring Algorithm:**
- **Engagement Rate** (30%): Likes, comments, shares per follower
- **Follower Quality** (25%): Fake follower detection, audience demographics
- **Content Relevance** (20%): Category matching, semantic similarity
- **Brand Safety** (15%): Content analysis, reputation scoring  
- **Posting Consistency** (10%): Frequency, schedule reliability

#### Budget Optimizer Service (POC4)
```python
class BudgetOptimizerService:
    async def optimize_campaign_budget(campaign, budget, objective) -> OptimizationResult
    async def generate_alternative_scenarios(campaign, budgets) -> List[OptimizationResult]
    async def create_budget_plan_from_optimization(result) -> BudgetPlan
```

**Optimization Objectives:**
- **Maximize Reach**: Optimize for total audience size
- **Maximize Engagement**: Focus on interaction rates
- **Maximize Conversions**: Target conversion-focused KOLs
- **Minimize Cost**: Best value within constraints
- **Balanced**: Multi-objective optimization

## 🔒 Security Architecture

### Authentication Flow
1. **JWT Authentication**: Access + refresh token pattern
2. **HTTP-Only Cookies**: Secure token storage for web clients
3. **Role-Based Access**: Admin, Manager, Analyst, Viewer roles
4. **Rate Limiting**: Redis-backed distributed limiting

### Data Protection
- **Input Validation**: Pydantic models with comprehensive validation
- **SQL Injection Prevention**: Parameterized queries via SQLAlchemy
- **Audit Logging**: Structured logging with user context
- **File Upload Security**: Type validation, size limits, virus scanning

## 📊 Monitoring & Observability

### Health Checks
```bash
GET /api/health          # Basic health check
GET /api/health/ready    # Readiness for load balancer
GET /api/health/live     # Liveness for container orchestration
```

### Metrics & Logging
- **Prometheus Metrics**: `/metrics` endpoint for monitoring
- **Structured Logging**: JSON logs with correlation IDs
- **Performance Tracking**: Request timing, database query metrics
- **Error Tracking**: Comprehensive error logging and alerting

## 🚀 Deployment Architecture

### Development Environment
```bash
docker-compose up --build
```
- **API Server**: FastAPI with hot reload
- **PostgreSQL**: With pgvector extension
- **Redis**: Caching and task queue
- **Celery Workers**: Background task processing

### Production Deployment
```bash
docker build -t kol-platform-api .
kubectl apply -f k8s/
```
- **Multi-stage Docker**: Optimized production images
- **Kubernetes Ready**: Health checks, resource limits
- **Horizontal Scaling**: Stateless application design
- **Database Scaling**: Read replicas, connection pooling

## 🔄 Integration Points

### Go Scraper Service
```http
POST /api/v1/kols/refresh
{
  "platform": "tiktok",
  "kol_ids": ["123", "456"]
}
```

### Frontend Integration (React/Next.js)
- **GraphQL Subscriptions**: Real-time updates
- **File Upload Endpoints**: Media asset handling
- **Authentication Flow**: JWT + HTTP-only cookies
- **CORS Configuration**: Development and production origins

## 📈 Performance Considerations

### Database Optimization
- **Connection Pooling**: SQLAlchemy async pool management  
- **Query Optimization**: Proper indexing, query analysis
- **Vector Operations**: pgvector performance tuning
- **Caching Strategy**: Redis for frequently accessed data

### Application Scaling
- **Async/Await**: Non-blocking I/O throughout
- **Background Tasks**: CPU-intensive operations in Celery
- **Rate Limiting**: Prevent API abuse and overload
- **Memory Management**: Efficient vector operations

## 🧪 Testing Strategy

### Test Coverage
```bash
pytest tests/unit/           # Unit tests for services
pytest tests/integration/    # Database integration tests  
pytest tests/graphql/        # GraphQL endpoint tests
pytest tests/load/           # Performance and load tests
```

### Quality Assurance
- **Type Safety**: Full type hints with mypy validation
- **Code Quality**: Black formatting, Ruff linting
- **Security Scanning**: Dependency vulnerability checks
- **Performance Testing**: Load testing for critical endpoints

## 📋 Next Implementation Steps

### Phase 1: Core Functionality (Week 1-2)
1. **Database Migration**: Create initial migration with all tables
2. **Authentication System**: Complete JWT auth implementation
3. **Basic GraphQL Queries**: KOL and campaign retrieval
4. **Health Monitoring**: Complete health check endpoints

### Phase 2: POC2 Implementation (Week 3-4)  
1. **Scoring Algorithm**: Complete multi-factor scoring
2. **Vector Similarity**: pgvector integration for content matching
3. **KOL Matching**: AI-powered recommendation system
4. **Performance Optimization**: Query optimization and caching

### Phase 3: POC4 Implementation (Week 5-6)
1. **Budget Algorithms**: Linear programming optimization
2. **Constraint Handling**: Business rule implementation  
3. **Scenario Generation**: Multi-budget analysis tools
4. **Plan Execution**: Budget plan approval and execution

### Phase 4: Integration (Week 7-8)
1. **Go Scraper Integration**: HTTP client implementation
2. **Background Tasks**: Celery worker setup
3. **Frontend Integration**: GraphQL client setup
4. **Production Deployment**: Kubernetes configuration

## 🎯 Success Metrics

### Technical Metrics
- **API Response Time**: < 200ms for 95% of requests
- **Database Query Performance**: < 100ms for complex queries
- **Vector Similarity Search**: < 500ms for similarity queries
- **Background Task Processing**: < 30s for scoring operations

### Business Metrics  
- **KOL Matching Accuracy**: > 85% relevance score
- **Budget Optimization Efficiency**: > 20% cost savings
- **User Adoption**: Complete workflow POC demonstration
- **System Reliability**: 99.9% uptime in production

---

**Note**: This architecture provides a solid foundation for all four POCs while maintaining production-ready quality, security, and scalability. The implementation prioritizes POC2 (KOL Matching) and POC4 (Budget Optimization) as the highest-value features.