# AIDEV-NOTE: Multi-stage Docker build for Go KOL scraper service
# Uses Alpine Linux for small production image size

# Build stage
FROM golang:1.21-alpine AS builder

# AIDEV-NOTE: Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# AIDEV-NOTE: Set working directory
WORKDIR /app

# AIDEV-NOTE: Copy go mod files first for better cache utilization
COPY go.mod go.sum ./

# AIDEV-NOTE: Download dependencies
RUN go mod download

# AIDEV-NOTE: Copy source code
COPY . .

# AIDEV-NOTE: Build the binary with optimizations
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags='-w -s -extldflags "-static"' \
    -a -installsuffix cgo \
    -o kol-scraper \
    ./main.go

# Production stage
FROM alpine:latest

# AIDEV-NOTE: Install runtime dependencies
RUN apk --no-cache add ca-certificates tzdata curl

# AIDEV-NOTE: Create non-root user for security
RUN adduser -D -s /bin/sh scraper

# AIDEV-NOTE: Set working directory
WORKDIR /home/scraper

# AIDEV-NOTE: Copy binary from builder stage
COPY --from=builder /app/kol-scraper .

# AIDEV-NOTE: Copy configuration files if they exist
COPY --from=builder /app/config/ ./config/ 2>/dev/null || true

# AIDEV-NOTE: Set ownership and permissions
RUN chown -R scraper:scraper /home/scraper && \
    chmod +x ./kol-scraper

# AIDEV-NOTE: Switch to non-root user
USER scraper

# AIDEV-NOTE: Expose default port
EXPOSE 8080

# AIDEV-NOTE: Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# AIDEV-NOTE: Set default command
CMD ["./kol-scraper"]