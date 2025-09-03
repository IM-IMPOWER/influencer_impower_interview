// AIDEV-NOTE: Main entry point for KOL Data Discovery microservice (POC1)
// High-performance Go service for scalable KOL data collection and processing
package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"kol-scraper/internal/database"
	"kol-scraper/internal/fastapi"
	"kol-scraper/internal/handlers"
	"kol-scraper/internal/monitoring"
	"kol-scraper/internal/queue"
	"kol-scraper/internal/scrapers"
	"kol-scraper/pkg/config"
	"kol-scraper/pkg/logger"
)

func main() {
	// AIDEV-NOTE: Initialize configuration and logging
	cfg, err := config.Load()
	if err != nil {
		fmt.Printf("Failed to load configuration: %v\n", err)
		os.Exit(1)
	}

	log := logger.New(cfg.LogLevel, cfg.Environment)
	log.Info("Starting KOL Data Discovery Service")

	// AIDEV-NOTE: 250903170520 Initialize enhanced database connection with optimization features
	db, err := database.NewConnection(cfg, log)
	if err != nil {
		log.Fatal("Failed to connect to database", "error", err)
	}
	defer db.Close()

	// AIDEV-NOTE: Run database migrations if enabled
	if cfg.AutoMigrate {
		if err := database.Migrate(db, cfg.MigrationsPath); err != nil {
			log.Fatal("Failed to run migrations", "error", err)
		}
		log.Info("Database migrations completed successfully")
	}

	// AIDEV-NOTE: Initialize job queue system for bulk operations
	jobQueue, err := queue.NewJobQueue(cfg, log)
	if err != nil {
		log.Fatal("Failed to initialize job queue", "error", err)
	}
	defer jobQueue.Close()

	// AIDEV-NOTE: Start background workers for processing scraping jobs
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := jobQueue.StartWorkers(ctx); err != nil {
		log.Fatal("Failed to start job queue workers", "error", err)
	}

	// AIDEV-NOTE: Setup HTTP server with Gin router
	if cfg.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()
	
	// AIDEV-NOTE: Add middleware for logging, recovery, CORS, and monitoring
	router.Use(gin.Logger())
	router.Use(gin.Recovery())
	router.Use(corsMiddleware())
	router.Use(rateLimitMiddleware())
	router.Use(metricsCollector.HTTPMetricsMiddleware())

	// AIDEV-NOTE: Initialize scraper manager
	scraperMgr := scrapers.NewManager(cfg, log)

	// AIDEV-NOTE: Initialize FastAPI client for POC2/POC4 integration
	var fastapiClient *fastapi.Client
	if cfg.FastAPIURL != "" {
		fastapiClient, err = fastapi.NewClient(cfg, log)
		if err != nil {
			log.Warn("Failed to initialize FastAPI client - integration features will be disabled", "error", err)
		} else {
			log.Info("FastAPI client initialized successfully", "url", cfg.FastAPIURL)
			defer fastapiClient.Close()
		}
	} else {
		log.Warn("FastAPI URL not configured - integration features will be disabled")
	}

	// AIDEV-NOTE: 250903170020 Initialize monitoring system
	metricsCollector := monitoring.NewMetricsCollector(log)
	healthChecker := monitoring.NewHealthChecker(db.DB, nil, log) // TODO: Add Redis client
	
	// AIDEV-NOTE: Start system metrics collection
	go metricsCollector.UpdateSystemMetrics(ctx)
	
	// AIDEV-NOTE: Initialize handlers with dependencies
	h := handlers.New(db, jobQueue, log)
	h.SetScraperManager(scraperMgr)
	h.SetMetricsCollector(metricsCollector)
	if fastapiClient != nil {
		h.SetFastAPIClient(fastapiClient)
	}

	// AIDEV-NOTE: Register job handlers
	jobHandlerFactory := handlers.NewJobHandlerFactory(db, scraperMgr, log)
	jobHandlerFactory.RegisterHandlers(jobQueue)

	// AIDEV-NOTE: Register API routes
	setupRoutes(router, h, metricsCollector, healthChecker)

	// AIDEV-NOTE: Start monitoring server for Prometheus metrics (separate port)
	if cfg.Monitoring.Enabled {
		go func() {
			metricsServer := &http.Server{
				Addr:    fmt.Sprintf(":%d", cfg.Monitoring.PrometheusPort),
				Handler: metricsCollector.ServeMetrics(),
			}
			log.Info("Metrics server starting", "port", cfg.Monitoring.PrometheusPort)
			if err := metricsServer.ListenAndServe(); err != nil {
				log.Error("Metrics server failed", "error", err)
			}
		}()
	}
	
	// AIDEV-NOTE: Create HTTP server with timeouts
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Port),
		Handler:      router,
		ReadTimeout:  time.Duration(cfg.ReadTimeout) * time.Second,
		WriteTimeout: time.Duration(cfg.WriteTimeout) * time.Second,
		IdleTimeout:  time.Duration(cfg.IdleTimeout) * time.Second,
	}

	// AIDEV-NOTE: Start server in goroutine for graceful shutdown
	go func() {
		log.Info("Server starting", "port", cfg.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal("Failed to start server", "error", err)
		}
	}()

	// AIDEV-NOTE: Wait for interrupt signal for graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Info("Server shutting down...")

	// AIDEV-NOTE: Graceful shutdown with timeout
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Error("Server forced to shutdown", "error", err)
	}

	log.Info("Server shutdown completed")
}

// AIDEV-NOTE: 250903170021 Setup API routes with monitoring integration
func setupRoutes(router *gin.Engine, h *handlers.Handler, metrics *monitoring.MetricsCollector, health *monitoring.HealthChecker) {
	api := router.Group("/api/v1")
	{
		// AIDEV-NOTE: Health and status endpoints with comprehensive monitoring
		api.GET("/health", gin.WrapF(health.HealthHandler()))
		api.GET("/health/ready", gin.WrapF(health.ReadinessHandler()))
		api.GET("/health/live", gin.WrapF(health.LivenessHandler()))
		api.GET("/metrics", h.Metrics) // Legacy endpoint
		api.GET("/metrics/business", h.BusinessMetrics) // Business-specific metrics

		// AIDEV-NOTE: KOL scraping endpoints
		scrape := api.Group("/scrape")
		{
			scrape.POST("/kol/:platform/:username", h.ScrapeKOL)
			scrape.POST("/bulk", h.BulkScrape)
			scrape.GET("/status/:job_id", h.ScrapeStatus)
			scrape.DELETE("/job/:job_id", h.CancelScrape)
		}

		// AIDEV-NOTE: Data update endpoints
		update := api.Group("/update")
		{
			update.POST("/metrics", h.UpdateMetrics)
			update.POST("/profile/:kol_id", h.UpdateProfile)
			update.POST("/content/:kol_id", h.UpdateContent)
		}

		// AIDEV-NOTE: Data query endpoints for integration
		data := api.Group("/data")
		{
			data.GET("/kols", h.GetKOLs)
			data.GET("/kol/:id", h.GetKOL)
			data.GET("/stats", h.GetStats)
		}

		// AIDEV-NOTE: Integration endpoints for FastAPI backend
		integration := api.Group("/integration")
		{
			integration.POST("/webhook", h.WebhookCallback)
			integration.GET("/status", h.IntegrationStatus)
			integration.POST("/match-kols", h.MatchKOLs)
			integration.POST("/optimize-budget", h.OptimizeBudget)
		}
	}
}

// AIDEV-NOTE: CORS middleware for cross-origin requests
func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Credentials", "true")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With")
		c.Header("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}

// AIDEV-NOTE: Rate limiting middleware to respect platform limits
func rateLimitMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement rate limiting logic based on endpoint and client
		c.Next()
	}
}