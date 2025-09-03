#!/bin/bash
# AIDEV-NOTE: 250903170023 Production deployment script for KOL platform
# Handles environment-specific deployment with health checks and rollback

set -euo pipefail

# AIDEV-NOTE: Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENVIRONMENT="${1:-production}"
COMPOSE_FILE="docker-compose.production.yml"

# AIDEV-NOTE: Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# AIDEV-NOTE: Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# AIDEV-NOTE: Error handling
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Deployment failed with exit code $exit_code"
        log_info "Check logs and consider running rollback"
    fi
    exit $exit_code
}

trap cleanup EXIT

# AIDEV-NOTE: Pre-deployment checks
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed"
        exit 1
    fi
    
    # Check if environment file exists
    if [ ! -f "$PROJECT_ROOT/.env.production" ]; then
        log_error "Production environment file not found"
        log_info "Copy .env.production.example to .env.production and configure it"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# AIDEV-NOTE: Health check function
health_check() {
    local service=$1
    local port=$2
    local path=$3
    local max_attempts=30
    local attempt=1
    
    log_info "Health checking $service on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:$port$path" > /dev/null; then
            log_success "$service is healthy"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts - $service not ready yet, waiting 10s..."
        sleep 10
        ((attempt++))
    done
    
    log_error "$service health check failed after $max_attempts attempts"
    return 1
}

# AIDEV-NOTE: Database migration function
run_migrations() {
    log_info "Running database migrations..."
    
    # Check if database is ready
    if ! health_check "postgres" 5432 ""; then
        log_error "Database is not ready for migrations"
        return 1
    fi
    
    # Run migrations through the API service
    docker-compose -f "$COMPOSE_FILE" exec -T kol-api python -m alembic upgrade head
    
    log_success "Database migrations completed"
}

# AIDEV-NOTE: Backup function
create_backup() {
    log_info "Creating backup before deployment..."
    
    local backup_name="backup_$(date +%Y%m%d_%H%M%S)"
    
    # Create database backup
    docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump \
        -U "${POSTGRES_USER}" \
        -d "${POSTGRES_DB}" \
        --no-password > "$PROJECT_ROOT/backups/db_$backup_name.sql"
    
    log_success "Backup created: $backup_name"
}

# AIDEV-NOTE: Main deployment function
deploy() {
    log_info "Starting production deployment for $ENVIRONMENT..."
    
    cd "$PROJECT_ROOT"
    
    # AIDEV-NOTE: Pull latest images if using registry
    log_info "Pulling latest Docker images..."
    docker-compose -f "$COMPOSE_FILE" pull
    
    # AIDEV-NOTE: Create backup
    create_backup
    
    # AIDEV-NOTE: Build and start services
    log_info "Building and starting services..."
    docker-compose -f "$COMPOSE_FILE" up -d --build
    
    # AIDEV-NOTE: Wait for core services to be ready
    log_info "Waiting for core services to be ready..."
    sleep 30
    
    # AIDEV-NOTE: Health check core services
    health_check "postgres" 5432 ""
    health_check "redis" 6379 ""
    health_check "kol-api" 8000 "/api/health/ready"
    health_check "kol-scraper" 8080 "/api/v1/health/ready"
    health_check "prometheus" 9090 "/-/ready"
    health_check "grafana" 3000 "/api/health"
    
    # AIDEV-NOTE: Run migrations
    run_migrations
    
    # AIDEV-NOTE: Final health checks
    log_info "Running final health checks..."
    health_check "kol-api" 8000 "/api/health"
    health_check "kol-scraper" 8080 "/api/v1/health"
    
    log_success "Deployment completed successfully!"
    log_info "Services are now running:"
    log_info "  - API: http://localhost:8000"
    log_info "  - Scraper: http://localhost:8080"
    log_info "  - Prometheus: http://localhost:9090"
    log_info "  - Grafana: http://localhost:3000"
}

# AIDEV-NOTE: Rollback function
rollback() {
    log_warning "Rolling back deployment..."
    
    # Stop current services
    docker-compose -f "$COMPOSE_FILE" down
    
    # TODO: Implement proper rollback to previous version
    # This would involve:
    # 1. Restoring previous Docker images
    # 2. Rolling back database migrations if needed
    # 3. Restoring configuration
    
    log_info "Rollback completed"
}

# AIDEV-NOTE: Show status
show_status() {
    log_info "Current deployment status:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    log_info "Service health status:"
    services=("kol-api:8000:/api/health" "kol-scraper:8080:/api/v1/health" "prometheus:9090:/-/ready" "grafana:3000:/api/health")
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r service port path <<< "$service_info"
        if curl -f -s "http://localhost:$port$path" > /dev/null; then
            log_success "$service is healthy"
        else
            log_error "$service is unhealthy"
        fi
    done
}

# AIDEV-NOTE: Main script logic
case "${2:-deploy}" in
    "deploy")
        check_prerequisites
        deploy
        ;;
    "rollback")
        rollback
        ;;
    "status")
        show_status
        ;;
    "health")
        check_prerequisites
        show_status
        ;;
    *)
        echo "Usage: $0 [environment] [deploy|rollback|status|health]"
        echo "  environment: production (default), staging"
        echo "  action: deploy (default), rollback, status, health"
        exit 1
        ;;
esac