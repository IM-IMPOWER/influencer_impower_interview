# AIDEV-NOTE: Multi-stage Docker build for FastAPI KOL platform API

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# AIDEV-NOTE: Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# AIDEV-NOTE: Copy dependency files first for better Docker layer caching
COPY pyproject.toml .
COPY README.md .

# AIDEV-NOTE: Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

WORKDIR /app

# AIDEV-NOTE: Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser appuser

# AIDEV-NOTE: Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# AIDEV-NOTE: Copy application code
COPY src/ ./src/
COPY alembic.ini .
COPY migrations/ ./migrations/

# AIDEV-NOTE: Create directories for uploads and logs
RUN mkdir -p /app/uploads /app/logs && \
    chown -R appuser:appuser /app

# AIDEV-NOTE: Switch to non-root user for security
USER appuser

# AIDEV-NOTE: Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# AIDEV-NOTE: Expose port
EXPOSE 8000

# AIDEV-NOTE: Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# AIDEV-NOTE: Use exec form for better signal handling
CMD ["python", "-m", "uvicorn", "kol_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]