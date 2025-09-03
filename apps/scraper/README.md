# KOL Data Discovery Service (POC1)

A high-performance Go microservice for scalable KOL (Key Opinion Leader) data collection and processing. This service provides the foundation for POC1 (KOL Data Discovery) while integrating seamlessly with POC2 (KOL Matching) and POC4 (Budget Optimizer).

## Features

### 🚀 High-Performance Scraping
- **TikTok Profile Scraping**: Extract comprehensive profile data, metrics, and content
- **Rate Limiting**: Intelligent rate limiting to respect platform limits
- **Concurrent Processing**: Handle 100+ concurrent scraping operations
- **Retry Logic**: Exponential backoff with configurable retry attempts
- **Error Recovery**: Graceful error handling and recovery mechanisms

### 📊 Data Management
- **PostgreSQL Integration**: Efficient data storage with pgvector support
- **Bulk Operations**: Process 1000+ records efficiently
- **Data Quality**: Automated data validation and quality scoring
- **Transaction Safety**: ACID-compliant operations with proper rollback

### 🔄 Job Queue System
- **Redis-backed Queue**: Scalable job processing with Redis
- **Priority Queuing**: Priority-based job execution
- **Worker Management**: Configurable worker pools
- **Job Status Tracking**: Real-time job status and progress monitoring
- **Dead Letter Queue**: Failed job handling and retry management

### 🌐 REST API
- **Comprehensive Endpoints**: Full CRUD operations for KOL data
- **Real-time Status**: Job status and progress tracking
- **Bulk Operations**: Efficient bulk scraping and updates
- **Health Monitoring**: Service health checks and metrics
- **Integration Ready**: Webhook support for external integrations

### 🔧 Production Ready
- **Docker Support**: Full containerization with docker-compose
- **Configuration Management**: Environment-based configuration
- **Logging**: Structured logging with contextual information
- **Monitoring**: Health checks and performance metrics
- **Security**: Non-root user, proper permissions, and input validation

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone and navigate to the scraper directory
cd apps/scraper

# Start all services (scraper, PostgreSQL, Redis)
docker-compose up -d

# Check service health
curl http://localhost:8080/api/v1/health

# View logs
docker-compose logs -f kol-scraper
```

### Manual Setup

```bash
# Install dependencies
go mod download

# Set required environment variables
export DATABASE_URL="postgres://user:password@localhost:5432/koldb?sslmode=disable"
export REDIS_URL="redis://localhost:6379"

# Build and run
go build -o kol-scraper main.go
./kol-scraper
```

### Using Build Scripts

```bash
# Build binaries for all platforms
./scripts/build.sh

# Start service in background
./scripts/start.sh --background

# Stop service
./scripts/stop.sh
```

## API Documentation

### Base URL
```
http://localhost:8080/api/v1
```

### Health & Monitoring

#### Health Check
```http
GET /health
```

#### Service Metrics
```http
GET /metrics
```

### KOL Scraping

#### Scrape Single KOL
```http
POST /scrape/kol/{platform}/{username}
Content-Type: application/json

{
  "priority": 50,
  "timeout": 120,
  "params": {
    "include_content": true,
    "content_limit": 10
  }
}
```

#### Bulk Scraping
```http
POST /scrape/bulk
Content-Type: application/json

{
  "platform": "tiktok",
  "usernames": ["user1", "user2", "user3"],
  "priority": 75,
  "timeout": 300
}
```

#### Check Job Status
```http
GET /scrape/status/{job_id}
```

#### Cancel Job
```http
DELETE /scrape/job/{job_id}
```

### Data Updates

#### Update Metrics
```http
POST /update/metrics
Content-Type: application/json

{
  "platform": "tiktok",
  "usernames": ["user1", "user2"],
  "force_refresh": true
}
```

#### Update Profile
```http
POST /update/profile/{kol_id}
```

#### Update Content
```http
POST /update/content/{kol_id}
```

### Data Queries

#### Get KOLs with Filtering
```http
GET /data/kols?platform=tiktok&tier=micro&min_followers=10000&limit=20
```

Query Parameters:
- `platform`: Filter by platform (tiktok, instagram, youtube, etc.)
- `tier`: Filter by tier (nano, micro, mid, macro, mega)
- `category`: Filter by primary category
- `min_followers`, `max_followers`: Follower count range
- `min_engagement`, `max_engagement`: Engagement rate range
- `is_verified`: Filter verified accounts
- `is_brand_safe`: Filter brand-safe accounts
- `location`: Filter by location (partial match)
- `languages`: Filter by languages (array)
- `last_scraped_days`: Filter by recent scraping activity
- `limit`, `offset`: Pagination
- `sort_by`, `sort_order`: Sorting options

#### Get Single KOL
```http
GET /data/kol/{id}?include_metrics=true&include_content=true&include_profile=true
```

#### Get Statistics
```http
GET /data/stats
```

### Integration

#### Webhook Endpoint
```http
POST /integration/webhook
Content-Type: application/json
X-Webhook-Signature: signature

{
  "type": "scrape_completed",
  "job_id": "job-123",
  "success": true,
  "data": {...}
}
```

#### Integration Status
```http
GET /integration/status
```

## Configuration

### Environment Variables

#### Required
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string

#### Server Configuration
- `ENVIRONMENT`: Environment (development/production)
- `PORT`: Server port (default: 8080)
- `READ_TIMEOUT`: HTTP read timeout in seconds (default: 30)
- `WRITE_TIMEOUT`: HTTP write timeout in seconds (default: 30)
- `IDLE_TIMEOUT`: HTTP idle timeout in seconds (default: 120)

#### Database Configuration
- `MAX_DB_CONNECTIONS`: Maximum database connections (default: 25)
- `MAX_IDLE_DB_CONNECTIONS`: Maximum idle connections (default: 5)
- `DB_CONN_MAX_LIFETIME`: Connection max lifetime in seconds (default: 300)
- `AUTO_MIGRATE`: Run migrations on startup (default: false)

#### Scraping Configuration
- `MAX_CONCURRENT_SCRAPE`: Maximum concurrent scraping operations (default: 10)
- `RATE_LIMIT_RPS`: Global rate limit requests per second (default: 2)
- `RETRY_ATTEMPTS`: Number of retry attempts (default: 3)
- `REQUEST_TIMEOUT`: HTTP request timeout (default: 30s)

#### Queue Configuration
- `QUEUE_WORKERS`: Number of queue workers (default: 5)
- `JOB_TIMEOUT`: Job execution timeout (default: 300s)
- `MAX_QUEUE_SIZE`: Maximum queue size (default: 1000)

#### TikTok Configuration
- `TIKTOK_ENABLED`: Enable TikTok scraping (default: true)
- `TIKTOK_RATE_LIMIT_RPS`: TikTok-specific rate limit (default: 1)
- `TIKTOK_PROXY_URL`: Proxy URL for TikTok requests (optional)
- `TIKTOK_SESSION_COOKIE`: Session cookie for authenticated requests (optional)

#### Logging
- `LOG_LEVEL`: Log level (trace/debug/info/warn/error/fatal/panic)
- `LOG_JSON`: Use JSON logging format (default: false)

#### Integration
- `FASTAPI_URL`: FastAPI backend URL for integration
- `WEBHOOK_SECRET`: Secret for webhook signature verification
- `API_TIMEOUT`: API request timeout in seconds (default: 30)

### Configuration File

You can also use a `config/config.yaml` file:

```yaml
environment: development
port: 8080
database_url: postgres://user:pass@localhost:5432/koldb
redis_url: redis://localhost:6379

scraping:
  max_concurrent: 10
  rate_limit_rps: 2
  retry_attempts: 3
  request_timeout: 30s

queue:
  workers: 5
  job_timeout: 300s
  max_queue_size: 1000

tiktok:
  enabled: true
  rate_limit_rps: 1
  proxy_url: ""
  session_cookie: ""

logging:
  level: info
  json: false
```

## Architecture

### Components

1. **HTTP Server**: Gin-based REST API server
2. **Scraper Manager**: Multi-platform scraper coordination
3. **TikTok Scraper**: Specialized TikTok data extraction
4. **Job Queue**: Redis-backed job processing system
5. **Database Layer**: PostgreSQL with pgvector support
6. **Configuration**: Environment-based configuration management
7. **Logging**: Structured logging with contextual information

### Data Flow

1. **API Request** → HTTP handlers validate and queue jobs
2. **Job Queue** → Redis stores jobs with priority and metadata
3. **Workers** → Process jobs concurrently with retry logic
4. **Scrapers** → Extract data from platforms with rate limiting
5. **Database** → Store structured data with quality validation
6. **Integration** → Notify external systems via webhooks

### Database Schema

The service uses the existing PostgreSQL schema from the main KOL API:

- **kols**: Main KOL profiles and metadata
- **kol_metrics**: Performance metrics and statistics
- **kol_content**: Sample content and analysis
- **kol_profiles**: Extended profile information from various sources

## Development

### Prerequisites

- Go 1.21 or later
- PostgreSQL 12+ with pgvector extension
- Redis 6+
- Docker and docker-compose (for containerized development)

### Local Development

```bash
# Start dependencies
docker-compose up postgres redis -d

# Install dependencies
go mod download

# Run tests
go test ./...

# Start development server with hot reload (using air)
go install github.com/cosmtrek/air@latest
air

# Or run directly
go run main.go
```

### Testing

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Run specific test package
go test ./internal/scrapers/

# Run benchmarks
go test -bench=. ./...
```

### Building

```bash
# Build for current platform
go build -o kol-scraper main.go

# Build for all platforms using script
./scripts/build.sh

# Build Docker image
docker build -t kol-scraper .
```

## Deployment

### Docker Deployment

```bash
# Build and deploy with docker-compose
docker-compose up -d

# Scale workers
docker-compose up -d --scale kol-scraper=3

# Update service
docker-compose pull kol-scraper
docker-compose up -d kol-scraper
```

### Production Deployment

1. **Environment Setup**:
   - Set all required environment variables
   - Configure proper database credentials
   - Set up monitoring and logging

2. **Security**:
   - Use non-root user (handled in Dockerfile)
   - Set up proper network security
   - Configure webhook secret for integrations
   - Use TLS for external communications

3. **Monitoring**:
   - Monitor service health endpoints
   - Set up alerts for job failures
   - Monitor database and Redis connections
   - Track scraping success rates and performance

4. **Scaling**:
   - Scale horizontally by running multiple instances
   - Adjust worker count based on load
   - Monitor queue depth and processing times

## Integration with FastAPI Backend

The Go scraper service integrates with the existing FastAPI backend through:

1. **Database Sharing**: Uses the same PostgreSQL database
2. **Webhook Notifications**: Sends updates to FastAPI endpoints
3. **API Compatibility**: Maintains compatible data structures
4. **Job Coordination**: Coordinates with existing job systems

### Integration Flow

```
FastAPI → Triggers scraping jobs → Go Service
Go Service → Scrapes data → Updates database
Go Service → Sends webhook → FastAPI (notifications)
FastAPI → Reads updated data → POC2/POC4 processing
```

## Performance Characteristics

### Throughput
- **Single KOL**: ~2-5 seconds per profile (TikTok)
- **Bulk Operations**: 100+ concurrent operations
- **Queue Processing**: 1000+ jobs per hour (depends on platform limits)

### Resource Usage
- **Memory**: ~50-100MB base + ~5MB per concurrent job
- **CPU**: Low usage, I/O bound operations
- **Database**: ~10-50 connections depending on configuration
- **Network**: Varies based on scraping volume and rate limits

### Scalability
- **Horizontal**: Multiple service instances with shared queue
- **Vertical**: Configurable worker pools and connection limits
- **Storage**: PostgreSQL with proper indexing for large datasets

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check database connectivity
   psql $DATABASE_URL -c "SELECT 1"
   
   # Verify extensions
   psql $DATABASE_URL -c "SELECT * FROM pg_extension"
   ```

2. **Redis Connection Failed**
   ```bash
   # Test Redis connection
   redis-cli -u $REDIS_URL ping
   ```

3. **Scraping Failures**
   - Check rate limiting configuration
   - Verify platform accessibility
   - Review proxy settings if used
   - Check for IP blocking

4. **Job Queue Issues**
   ```bash
   # Check queue stats
   curl http://localhost:8080/api/v1/metrics
   
   # Monitor Redis queues
   redis-cli -u $REDIS_URL keys "queue:*"
   ```

### Debugging

Enable debug logging:
```bash
export LOG_LEVEL=debug
./kol-scraper
```

Monitor specific components:
```bash
# Database operations
grep "database" logs/service.log

# Queue operations  
grep "queue" logs/service.log

# Scraping operations
grep "scraper" logs/service.log
```

## Contributing

1. Follow Go best practices and effective Go guidelines
2. Write tests for new functionality
3. Update documentation for API changes
4. Use structured logging with appropriate context
5. Follow the existing error handling patterns

## License

This project is part of the KOL Platform and follows the same licensing terms.