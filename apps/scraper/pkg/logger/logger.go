// AIDEV-NOTE: Structured logging package for KOL scraper service
// Uses logrus for structured logging with contextual information
package logger

import (
	"context"
	"strings"

	"github.com/sirupsen/logrus"
)

// Logger wraps logrus.Logger with additional methods
type Logger struct {
	*logrus.Logger
}

// Fields is an alias for logrus.Fields for convenience
type Fields = logrus.Fields

// New creates a new logger instance with specified level and environment
func New(level, environment string) *Logger {
	log := logrus.New()

	// AIDEV-NOTE: Set log level
	logLevel, err := logrus.ParseLevel(strings.ToLower(level))
	if err != nil {
		log.Warn("Invalid log level, defaulting to info", "level", level)
		logLevel = logrus.InfoLevel
	}
	log.SetLevel(logLevel)

	// AIDEV-NOTE: Set formatter based on environment
	if strings.EqualFold(environment, "production") {
		log.SetFormatter(&logrus.JSONFormatter{
			TimestampFormat: "2006-01-02T15:04:05.000Z",
		})
	} else {
		log.SetFormatter(&logrus.TextFormatter{
			FullTimestamp:   true,
			TimestampFormat: "2006-01-02 15:04:05",
			ForceColors:     true,
		})
	}

	return &Logger{Logger: log}
}

// WithField adds a single field to the logger
func (l *Logger) WithField(key string, value interface{}) *logrus.Entry {
	return l.Logger.WithField(key, value)
}

// WithFields adds multiple fields to the logger
func (l *Logger) WithFields(fields Fields) *logrus.Entry {
	return l.Logger.WithFields(logrus.Fields(fields))
}

// WithContext adds context information to the logger
func (l *Logger) WithContext(ctx context.Context) *logrus.Entry {
	return l.Logger.WithContext(ctx)
}

// WithError adds error field to the logger
func (l *Logger) WithError(err error) *logrus.Entry {
	return l.Logger.WithError(err)
}

// ScrapeLog logs scraping-related activities with structured data
func (l *Logger) ScrapeLog(platform, username, operation string, fields Fields) *logrus.Entry {
	logFields := Fields{
		"component": "scraper",
		"platform":  platform,
		"username":  username,
		"operation": operation,
	}
	
	// AIDEV-NOTE: Merge additional fields
	for k, v := range fields {
		logFields[k] = v
	}
	
	return l.WithFields(logFields)
}

// QueueLog logs job queue activities with structured data
func (l *Logger) QueueLog(jobID, jobType, status string, fields Fields) *logrus.Entry {
	logFields := Fields{
		"component": "queue",
		"job_id":    jobID,
		"job_type":  jobType,
		"status":    status,
	}
	
	// AIDEV-NOTE: Merge additional fields
	for k, v := range fields {
		logFields[k] = v
	}
	
	return l.WithFields(logFields)
}

// DatabaseLog logs database operations with structured data
func (l *Logger) DatabaseLog(operation, table string, fields Fields) *logrus.Entry {
	logFields := Fields{
		"component": "database",
		"operation": operation,
		"table":     table,
	}
	
	// AIDEV-NOTE: Merge additional fields
	for k, v := range fields {
		logFields[k] = v
	}
	
	return l.WithFields(logFields)
}

// HTTPLog logs HTTP requests and responses with structured data
func (l *Logger) HTTPLog(method, path string, statusCode int, fields Fields) *logrus.Entry {
	logFields := Fields{
		"component":   "http",
		"method":      method,
		"path":        path,
		"status_code": statusCode,
	}
	
	// AIDEV-NOTE: Merge additional fields
	for k, v := range fields {
		logFields[k] = v
	}
	
	return l.WithFields(logFields)
}

// IntegrationLog logs integration activities with external services
func (l *Logger) IntegrationLog(service, operation string, fields Fields) *logrus.Entry {
	logFields := Fields{
		"component": "integration",
		"service":   service,
		"operation": operation,
	}
	
	// AIDEV-NOTE: Merge additional fields
	for k, v := range fields {
		logFields[k] = v
	}
	
	return l.WithFields(logFields)
}

// MetricsLog logs performance and business metrics
func (l *Logger) MetricsLog(metric string, value interface{}, fields Fields) *logrus.Entry {
	logFields := Fields{
		"component": "metrics",
		"metric":    metric,
		"value":     value,
	}
	
	// AIDEV-NOTE: Merge additional fields
	for k, v := range fields {
		logFields[k] = v
	}
	
	return l.WithFields(logFields)
}

// ErrorLog logs errors with additional context for debugging
func (l *Logger) ErrorLog(err error, operation string, fields Fields) *logrus.Entry {
	logFields := Fields{
		"component": "error",
		"operation": operation,
		"error":     err.Error(),
	}
	
	// AIDEV-NOTE: Merge additional fields
	for k, v := range fields {
		logFields[k] = v
	}
	
	return l.WithError(err).WithFields(logFields)
}