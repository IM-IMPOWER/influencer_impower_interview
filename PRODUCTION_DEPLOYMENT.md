# KOL Platform - Production Deployment Guide

## 🚀 Overview

This guide covers the complete production deployment setup for the KOL Platform, including infrastructure, monitoring, CI/CD, and operational procedures.

## 📋 Prerequisites

### Required Software
- Docker 24.0+ and Docker Compose 2.20+
- Node.js 18+ (for local development)
- Go 1.21+ (for local development)
- Python 3.11+ (for local development)

### Required Accounts
- **Railway**: Backend services deployment (Go scraper + FastAPI)
- **Vercel**: Frontend deployment (Next.js)
- **GitHub**: Source code and CI/CD
- **Slack**: Alert notifications (optional)

### Infrastructure Requirements
- **PostgreSQL with pgvector**: Railway PostgreSQL or external
- **Redis**: Railway Redis or external
- **Domain**: Custom domain for production

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Next.js Web   │    │   FastAPI API   │    │   Go Scraper    │
│   (Vercel)      │◄──►│   (Railway)     │◄──►│   (Railway)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │      Redis      │    │   Monitoring    │
│   (Railway)     │    │   (Railway)     │    │   (Optional)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### 1. Environment Setup

Copy the example environment file and configure it:

```bash
cp .env.production.example .env.production
```

### 2. Required Environment Variables

#### Database Configuration
```env
DATABASE_URL=postgresql://username:password@host:port/database
REDIS_URL=redis://host:port
REDIS_PASSWORD=your_redis_password
```

#### Security Configuration
```env
JWT_SECRET_KEY=your_jwt_secret_key_change_in_production
WEBHOOK_SECRET=your_webhook_secret
NEXTAUTH_SECRET=your_nextauth_secret
```

#### Service URLs
```env
NEXT_PUBLIC_API_URL=https://your-api-domain.com
CORS_ORIGINS=https://yourapp.com,https://www.yourapp.com
```

#### Monitoring (Optional)
```env
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/your/slack/webhook
GRAFANA_PASSWORD=your_grafana_password
```

### 3. Service-Specific Configuration

#### Railway Services
- **API Service**: `apps/api/` - FastAPI backend
- **Scraper Service**: `apps/scraper/` - Go microservice

#### Vercel Deployment
- **Web Service**: `apps/web/` - Next.js frontend

## 🚀 Deployment Methods

### Method 1: Automated CI/CD (Recommended)

The platform includes GitHub Actions workflows for automated deployment:

1. **Push to `develop`** → Deploys to staging
2. **Push to `main`** → Deploys to production

#### Setup CI/CD:

1. Configure GitHub Secrets:
```
RAILWAY_TOKEN=your_railway_token
VERCEL_TOKEN=your_vercel_token
VERCEL_ORG_ID=your_org_id
VERCEL_PROJECT_ID=your_project_id
PROD_DATABASE_URL=your_prod_database_url
STAGING_DATABASE_URL=your_staging_database_url
```

2. Push to trigger deployment:
```bash
git push origin main  # Production deployment
git push origin develop  # Staging deployment
```

### Method 2: Manual Railway Deployment

#### Deploy API Service:
```bash
cd apps/api
railway up
```

#### Deploy Scraper Service:
```bash
cd apps/scraper
railway up
```

### Method 3: Manual Vercel Deployment

```bash
cd apps/web
vercel --prod
```

### Method 4: Self-Hosted Docker (Full Stack)

For complete control or on-premises deployment:

```bash
# Production deployment
./scripts/deploy-production.sh production deploy

# Check status
./scripts/deploy-production.sh production status

# Health check
./scripts/deploy-production.sh production health
```

## 📊 Monitoring & Observability

### Included Monitoring Stack (LGTM)

- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards
- **Loki**: Log aggregation
- **Tempo**: Distributed tracing (optional)

### Health Endpoints

| Service | Endpoint | Purpose |
|---------|----------|---------|
| API | `/api/health` | Overall health |
| API | `/api/health/ready` | Readiness probe |
| Scraper | `/api/v1/health` | Overall health |
| Scraper | `/api/v1/health/ready` | Readiness probe |
| Web | `/api/health` | Frontend health |

### Key Metrics

- **Business Metrics**: Scraping jobs, queue size, error rates
- **System Metrics**: Memory, CPU, disk usage
- **Application Metrics**: Response times, throughput, errors

### Alerting

Alerts are configured for:
- High error rates (>5%)
- System resource usage (>80%)
- Service downtime
- Database connectivity issues
- Queue size growth

## 🔐 Security

### Security Features Implemented

- **Authentication**: JWT tokens with secure storage
- **Authorization**: Role-based access control
- **CORS**: Configured for production domains
- **Rate Limiting**: API and scraper rate limits
- **Input Validation**: Request validation and sanitization
- **Security Headers**: HSTS, CSP, etc.
- **Secrets Management**: Environment variables only

### Security Checklist

- [ ] Change all default passwords
- [ ] Configure CORS for production domains only
- [ ] Enable HTTPS with valid certificates
- [ ] Set up secure headers
- [ ] Configure rate limiting
- [ ] Enable audit logging
- [ ] Regular security updates

## 🚨 Operations

### Backup Strategy

#### Automated Backups
```bash
# Database backup (daily)
pg_dump -U $POSTGRES_USER -h $POSTGRES_HOST $POSTGRES_DB > backup_$(date +%Y%m%d).sql

# Redis backup
redis-cli --rdb backup_$(date +%Y%m%d).rdb
```

### Scaling

#### Horizontal Scaling (Railway)
1. Go to Railway dashboard
2. Select service
3. Adjust replicas in settings
4. Monitor performance

#### Vertical Scaling
1. Increase memory/CPU limits
2. Update environment variables if needed
3. Monitor resource usage

### Troubleshooting

#### Common Issues

1. **Service Won't Start**
   ```bash
   # Check logs
   railway logs
   
   # Check environment variables
   railway variables
   ```

2. **Database Connection Issues**
   ```bash
   # Test connection
   psql $DATABASE_URL
   
   # Check connection pool
   curl http://your-api/api/health
   ```

3. **High Memory Usage**
   ```bash
   # Check metrics
   curl http://localhost:9090/metrics
   
   # Restart service if needed
   railway service restart
   ```

### Rollback Procedures

#### Automatic Rollback (Railway)
Railway provides automatic rollbacks on deployment failure.

#### Manual Rollback
```bash
# Via Railway CLI
railway rollback [deployment-id]

# Via Docker (self-hosted)
./scripts/deploy-production.sh production rollback
```

## 📈 Performance Optimization

### Database Optimization

- **Connection Pooling**: Configured for optimal concurrency
- **Query Optimization**: Indexes on frequently queried fields
- **Connection Limits**: Environment-specific limits

### Application Optimization

- **Caching**: Redis for session and query caching
- **Compression**: GZIP compression enabled
- **Image Optimization**: Next.js image optimization
- **Bundle Optimization**: Code splitting and tree shaking

### Infrastructure Optimization

- **CDN**: Vercel Edge Network for static assets
- **Load Balancing**: Railway automatic load balancing
- **Auto-scaling**: Based on CPU and memory usage

## 🔧 Maintenance

### Regular Maintenance Tasks

#### Weekly
- Review error logs and metrics
- Check disk space usage
- Verify backup integrity
- Update security patches

#### Monthly
- Database maintenance and optimization
- Log rotation and cleanup
- Performance review
- Security audit

#### Quarterly
- Dependency updates
- Infrastructure review
- Disaster recovery testing
- Performance benchmarking

### Monitoring Dashboards

Access monitoring dashboards at:
- **Grafana**: http://your-domain:3000
- **Prometheus**: http://your-domain:9090
- **Application Logs**: Via Loki/Grafana

### Support and Debugging

#### Log Locations
- **API Logs**: Railway logs or `/app/logs/`
- **Scraper Logs**: Railway logs or `/app/logs/`
- **Web Logs**: Vercel function logs

#### Debug Commands
```bash
# Check service health
curl -I http://your-api/api/health

# View live logs
railway logs --follow

# Check metrics
curl http://your-api/metrics
```

## 📞 Emergency Procedures

### Service Outage Response

1. **Immediate**: Check status dashboard
2. **Escalate**: Alert on-call engineer
3. **Communicate**: Update status page
4. **Resolve**: Follow runbook procedures
5. **Post-mortem**: Document incident

### Contact Information

- **On-call Engineer**: [Your contact]
- **Railway Support**: Via dashboard
- **Vercel Support**: Via dashboard

---

## 🎯 Quick Start Checklist

For immediate production deployment:

- [ ] Clone repository
- [ ] Configure `.env.production`
- [ ] Set up Railway services
- [ ] Set up Vercel project  
- [ ] Configure GitHub secrets
- [ ] Deploy via CI/CD or manually
- [ ] Verify all health endpoints
- [ ] Configure monitoring alerts
- [ ] Test backup procedures
- [ ] Document custom configurations

---

*Last updated: September 3, 2025*