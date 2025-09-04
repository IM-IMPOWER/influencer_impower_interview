// AIDEV-NOTE: Configuration management for KOL scraper service
// Loads settings from environment variables and config files
package config

import (
	"fmt"
	"strings"
	"time"

	"github.com/spf13/viper"
)

// Config holds all configuration for the scraper service
type Config struct {
	// AIDEV-NOTE: Server configuration
	Environment  string `mapstructure:"ENVIRONMENT"`
	Port         int    `mapstructure:"PORT"`
	ReadTimeout  int    `mapstructure:"READ_TIMEOUT"`
	WriteTimeout int    `mapstructure:"WRITE_TIMEOUT"`
	IdleTimeout  int    `mapstructure:"IDLE_TIMEOUT"`

	// AIDEV-NOTE: Database configuration
	DatabaseURL     string `mapstructure:"DATABASE_URL"`
	MaxConnections  int    `mapstructure:"MAX_DB_CONNECTIONS"`
	MaxIdleConns    int    `mapstructure:"MAX_IDLE_DB_CONNECTIONS"`
	ConnMaxLifetime int    `mapstructure:"DB_CONN_MAX_LIFETIME"`
	AutoMigrate     bool   `mapstructure:"AUTO_MIGRATE"`
	MigrationsPath  string `mapstructure:"MIGRATIONS_PATH"`

	// AIDEV-NOTE: Redis configuration for job queue
	RedisURL      string `mapstructure:"REDIS_URL"`
	RedisPassword string `mapstructure:"REDIS_PASSWORD"`
	RedisDB       int    `mapstructure:"REDIS_DB"`

	// AIDEV-NOTE: Scraping configuration
	UserAgent           string        `mapstructure:"USER_AGENT"`
	RequestTimeout      time.Duration `mapstructure:"REQUEST_TIMEOUT"`
	MaxConcurrentScrape int           `mapstructure:"MAX_CONCURRENT_SCRAPE"`
	RateLimitRPS        int           `mapstructure:"RATE_LIMIT_RPS"`
	RetryAttempts       int           `mapstructure:"RETRY_ATTEMPTS"`
	RetryDelay          time.Duration `mapstructure:"RETRY_DELAY"`

	// AIDEV-NOTE: Job queue configuration
	QueueWorkers    int           `mapstructure:"QUEUE_WORKERS"`
	JobTimeout      time.Duration `mapstructure:"JOB_TIMEOUT"`
	MaxQueueSize    int           `mapstructure:"MAX_QUEUE_SIZE"`
	CleanupInterval time.Duration `mapstructure:"CLEANUP_INTERVAL"`

	// AIDEV-NOTE: Logging configuration
	LogLevel string `mapstructure:"LOG_LEVEL"`
	LogJSON  bool   `mapstructure:"LOG_JSON"`

	// AIDEV-NOTE: Integration configuration
	FastAPIURL    string `mapstructure:"FASTAPI_URL"`
	WebhookSecret string `mapstructure:"WEBHOOK_SECRET"`
	APITimeout    int    `mapstructure:"API_TIMEOUT"`

	// AIDEV-NOTE: Circuit breaker configuration
	CircuitBreakerEnabled     bool          `mapstructure:"CIRCUIT_BREAKER_ENABLED"`
	CircuitBreakerMaxRequests int           `mapstructure:"CIRCUIT_BREAKER_MAX_REQUESTS"`
	CircuitBreakerInterval    time.Duration `mapstructure:"CIRCUIT_BREAKER_INTERVAL"`
	CircuitBreakerTimeout     time.Duration `mapstructure:"CIRCUIT_BREAKER_TIMEOUT"`

	// AIDEV-NOTE: Performance configuration
	EnableQueryCache          bool `mapstructure:"ENABLE_QUERY_CACHE"`
	CacheDefaultTTL          time.Duration `mapstructure:"CACHE_DEFAULT_TTL"`
	MaxRequestBodySize       int64 `mapstructure:"MAX_REQUEST_BODY_SIZE"`
	EnableRequestCompression bool `mapstructure:"ENABLE_REQUEST_COMPRESSION"`

	// AIDEV-NOTE: TikTok specific configuration
	TikTokConfig TikTokConfig `mapstructure:"TIKTOK"`

	// AIDEV-NOTE: 250903170004 Production monitoring configuration
	Monitoring MonitoringConfig `mapstructure:"MONITORING"`

	// AIDEV-NOTE: Security configuration
	Security SecurityConfig `mapstructure:"SECURITY"`

	// AIDEV-NOTE: Rate limiting configuration
	RateLimit RateLimitConfig `mapstructure:"RATE_LIMIT"`
}

// TikTokConfig holds TikTok-specific scraping configuration
type TikTokConfig struct {
	Enabled       bool          `mapstructure:"ENABLED"`
	RateLimitRPS  int           `mapstructure:"RATE_LIMIT_RPS"`
	ProxyURL      string        `mapstructure:"PROXY_URL"`
	SessionCookie string        `mapstructure:"SESSION_COOKIE"`
	MaxRetries    int           `mapstructure:"MAX_RETRIES"`
	Timeout       time.Duration `mapstructure:"TIMEOUT"`
}

// AIDEV-NOTE: 250903170005 Production monitoring configuration
type MonitoringConfig struct {
	Enabled               bool          `mapstructure:"ENABLED"`
	PrometheusPort        int           `mapstructure:"PROMETHEUS_PORT"`
	HealthCheckInterval   time.Duration `mapstructure:"HEALTH_CHECK_INTERVAL"`
	MetricsRetention      string        `mapstructure:"METRICS_RETENTION"`
	AlertWebhook          string        `mapstructure:"ALERT_WEBHOOK"`
	ErrorTracking         bool          `mapstructure:"ERROR_TRACKING"`
	PerformanceTracking   bool          `mapstructure:"PERFORMANCE_TRACKING"`
}

// SecurityConfig holds security-related configuration
type SecurityConfig struct {
	TrustedHosts           []string      `mapstructure:"TRUSTED_HOSTS"`
	MaxRequestSize         int64         `mapstructure:"MAX_REQUEST_SIZE"`
	RequestTimeout         time.Duration `mapstructure:"REQUEST_TIMEOUT"`
	EnableAuditLogging     bool          `mapstructure:"ENABLE_AUDIT_LOGGING"`
	SensitiveFields        []string      `mapstructure:"SENSITIVE_FIELDS"`
}

// RateLimitConfig holds rate limiting configuration
type RateLimitConfig struct {
	Enabled            bool `mapstructure:"ENABLED"`
	RequestsPerMinute  int  `mapstructure:"REQUESTS_PER_MINUTE"`
	BurstSize          int  `mapstructure:"BURST_SIZE"`
	PerIPLimit         int  `mapstructure:"PER_IP_LIMIT"`
}

// AIDEV-NOTE: 250903170003 Enhanced configuration loader with environment-specific support
// Load loads configuration from environment variables and config files
func Load() (*Config, error) {
	config := &Config{}

	// AIDEV-NOTE: Configure viper to read from environment and files
	viper.AutomaticEnv()
	viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	
	// AIDEV-NOTE: Set default values
	setDefaults()
	
	// AIDEV-NOTE: Get environment from ENV var or default to development
	environment := viper.GetString("ENVIRONMENT")
	if environment == "" {
		environment = "development"
	}
	
	// AIDEV-NOTE: Try to read environment-specific config file first
	viper.SetConfigName(environment)
	viper.SetConfigType("yaml")
	viper.AddConfigPath("./config/environments")
	viper.AddConfigPath("../../config/environments")
	viper.AddConfigPath("/app/config/environments")
	
	if err := viper.ReadInConfig(); err != nil {
		// AIDEV-NOTE: Fallback to generic config if environment-specific not found
		viper.SetConfigName("config")
		viper.AddConfigPath("./config")
		viper.AddConfigPath(".")
		
		if err := viper.ReadInConfig(); err != nil {
			if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
				return nil, fmt.Errorf("error reading config file: %w", err)
			}
			// Config file not found, continue with env vars and defaults
		}
	}

	// AIDEV-NOTE: Unmarshal configuration
	if err := viper.Unmarshal(config); err != nil {
		return nil, fmt.Errorf("unable to decode config: %w", err)
	}

	// AIDEV-NOTE: Validate configuration
	if err := validate(config); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	return config, nil
}

// setDefaults sets default configuration values
func setDefaults() {
	// AIDEV-NOTE: Server defaults
	viper.SetDefault("ENVIRONMENT", "development")
	viper.SetDefault("PORT", 8080)
	viper.SetDefault("READ_TIMEOUT", 30)
	viper.SetDefault("WRITE_TIMEOUT", 30)
	viper.SetDefault("IDLE_TIMEOUT", 120)

	// AIDEV-NOTE: Database defaults
	viper.SetDefault("MAX_DB_CONNECTIONS", 25)
	viper.SetDefault("MAX_IDLE_DB_CONNECTIONS", 5)
	viper.SetDefault("DB_CONN_MAX_LIFETIME", 300)
	viper.SetDefault("AUTO_MIGRATE", false)
	viper.SetDefault("MIGRATIONS_PATH", "./migrations")

	// AIDEV-NOTE: Redis defaults
	viper.SetDefault("REDIS_DB", 0)

	// AIDEV-NOTE: Scraping defaults
	viper.SetDefault("USER_AGENT", "KOL-Scraper/1.0 (+https://example.com/bot)")
	viper.SetDefault("REQUEST_TIMEOUT", "30s")
	viper.SetDefault("MAX_CONCURRENT_SCRAPE", 10)
	viper.SetDefault("RATE_LIMIT_RPS", 2)
	viper.SetDefault("RETRY_ATTEMPTS", 3)
	viper.SetDefault("RETRY_DELAY", "5s")

	// AIDEV-NOTE: Job queue defaults
	viper.SetDefault("QUEUE_WORKERS", 5)
	viper.SetDefault("JOB_TIMEOUT", "300s")
	viper.SetDefault("MAX_QUEUE_SIZE", 1000)
	viper.SetDefault("CLEANUP_INTERVAL", "3600s")

	// AIDEV-NOTE: Logging defaults
	viper.SetDefault("LOG_LEVEL", "info")
	viper.SetDefault("LOG_JSON", false)

	// AIDEV-NOTE: Integration defaults
	viper.SetDefault("API_TIMEOUT", 30)

	// AIDEV-NOTE: Circuit breaker defaults
	viper.SetDefault("CIRCUIT_BREAKER_ENABLED", true)
	viper.SetDefault("CIRCUIT_BREAKER_MAX_REQUESTS", 5)
	viper.SetDefault("CIRCUIT_BREAKER_INTERVAL", "60s")
	viper.SetDefault("CIRCUIT_BREAKER_TIMEOUT", "30s")

	// AIDEV-NOTE: Performance defaults
	viper.SetDefault("ENABLE_QUERY_CACHE", true)
	viper.SetDefault("CACHE_DEFAULT_TTL", "300s")
	viper.SetDefault("MAX_REQUEST_BODY_SIZE", 10485760) // 10MB
	viper.SetDefault("ENABLE_REQUEST_COMPRESSION", true)

	// AIDEV-NOTE: TikTok defaults
	viper.SetDefault("TIKTOK.ENABLED", true)
	viper.SetDefault("TIKTOK.RATE_LIMIT_RPS", 1)
	viper.SetDefault("TIKTOK.MAX_RETRIES", 3)
	viper.SetDefault("TIKTOK.TIMEOUT", "60s")

	// AIDEV-NOTE: 250903170006 Production monitoring defaults
	viper.SetDefault("MONITORING.ENABLED", true)
	viper.SetDefault("MONITORING.PROMETHEUS_PORT", 9090)
	viper.SetDefault("MONITORING.HEALTH_CHECK_INTERVAL", "30s")
	viper.SetDefault("MONITORING.METRICS_RETENTION", "7d")
	viper.SetDefault("MONITORING.ERROR_TRACKING", true)
	viper.SetDefault("MONITORING.PERFORMANCE_TRACKING", true)

	// AIDEV-NOTE: Security defaults
	viper.SetDefault("SECURITY.MAX_REQUEST_SIZE", 52428800) // 50MB
	viper.SetDefault("SECURITY.REQUEST_TIMEOUT", "30s")
	viper.SetDefault("SECURITY.ENABLE_AUDIT_LOGGING", true)
	viper.SetDefault("SECURITY.SENSITIVE_FIELDS", []string{"password", "token", "secret", "api_key"})

	// AIDEV-NOTE: Rate limiting defaults
	viper.SetDefault("RATE_LIMIT.ENABLED", false) // Disabled by default in development
	viper.SetDefault("RATE_LIMIT.REQUESTS_PER_MINUTE", 1000)
	viper.SetDefault("RATE_LIMIT.BURST_SIZE", 100)
	viper.SetDefault("RATE_LIMIT.PER_IP_LIMIT", 60)
}

// validate validates the configuration
func validate(config *Config) error {
	// AIDEV-NOTE: Validate required fields
	if config.DatabaseURL == "" {
		return fmt.Errorf("DATABASE_URL is required")
	}

	if config.RedisURL == "" {
		return fmt.Errorf("REDIS_URL is required")
	}

	if config.Port < 1 || config.Port > 65535 {
		return fmt.Errorf("PORT must be between 1 and 65535")
	}

	if config.MaxConcurrentScrape < 1 {
		return fmt.Errorf("MAX_CONCURRENT_SCRAPE must be at least 1")
	}

	if config.QueueWorkers < 1 {
		return fmt.Errorf("QUEUE_WORKERS must be at least 1")
	}

	// AIDEV-NOTE: Validate log level
	validLogLevels := []string{"trace", "debug", "info", "warn", "error", "fatal", "panic"}
	validLevel := false
	for _, level := range validLogLevels {
		if strings.EqualFold(config.LogLevel, level) {
			validLevel = true
			break
		}
	}
	if !validLevel {
		return fmt.Errorf("LOG_LEVEL must be one of: %s", strings.Join(validLogLevels, ", "))
	}

	return nil
}

// IsDevelopment returns true if running in development environment
func (c *Config) IsDevelopment() bool {
	return strings.EqualFold(c.Environment, "development")
}

// IsProduction returns true if running in production environment
func (c *Config) IsProduction() bool {
	return strings.EqualFold(c.Environment, "production")
}

// AIDEV-NOTE: 250903170007 Additional environment helpers
// IsStaging returns true if running in staging environment
func (c *Config) IsStaging() bool {
	return strings.EqualFold(c.Environment, "staging")
}

// GetLogFormat returns the appropriate log format based on environment
func (c *Config) GetLogFormat() string {
	if c.LogJSON || c.IsProduction() || c.IsStaging() {
		return "json"
	}
	return "console"
}

// GetMaxConnections returns environment-appropriate database connection limits
func (c *Config) GetMaxConnections() int {
	switch {
	case c.IsProduction():
		return 50
	case c.IsStaging():
		return 20
	default:
		return c.MaxConnections
	}
}