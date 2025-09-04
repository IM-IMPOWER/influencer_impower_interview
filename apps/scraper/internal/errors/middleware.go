// AIDEV-NOTE: 250903170000 HTTP error handling middleware for Gin framework
// Provides consistent error response formatting and logging integration
package errors

import (
	"context"
	"fmt"
	"net/http"
	"runtime"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"kol-scraper/pkg/logger"
)

// AIDEV-NOTE: 250903170000 Error response structure for client consumption
type ErrorResponse struct {
	Success   bool                   `json:"success"`
	Error     ErrorDetails           `json:"error"`
	Timestamp time.Time              `json:"timestamp"`
	RequestID string                 `json:"request_id,omitempty"`
	TraceID   string                 `json:"trace_id,omitempty"`
}

type ErrorDetails struct {
	Code        string                 `json:"code"`
	Message     string                 `json:"message"`
	Category    string                 `json:"category,omitempty"`
	Details     string                 `json:"details,omitempty"`
	Context     map[string]interface{} `json:"context,omitempty"`
	RetryAfter  *int64                 `json:"retry_after,omitempty"` // seconds
	Suggestions []string               `json:"suggestions,omitempty"`
}

// AIDEV-NOTE: 250903170000 Middleware configuration for error handling
type MiddlewareConfig struct {
	Logger                *logger.Logger
	IncludeStackTrace     bool              // Include stack trace in development
	IncludeErrorDetails   bool              // Include detailed error information
	IncludeContext        bool              // Include error context in response
	Environment           string            // Environment (production, development, etc.)
	CustomErrorMessages   map[string]string // Custom error messages for specific codes
	SensitiveFields       []string          // Fields to redact from error responses
	EnableMetrics         bool              // Enable error metrics collection
	OnErrorCallback       func(*ServiceError, *gin.Context) // Custom error callback
}

// DefaultMiddlewareConfig returns a default middleware configuration
func DefaultMiddlewareConfig() MiddlewareConfig {
	return MiddlewareConfig{
		IncludeStackTrace:   false,
		IncludeErrorDetails: true,
		IncludeContext:      false,
		Environment:         "production",
		SensitiveFields: []string{
			"password", "token", "key", "secret", "authorization",
			"cookie", "session", "credentials", "auth",
		},
		EnableMetrics: true,
	}
}

// AIDEV-NOTE: 250903170000 Error handling middleware factory
func ErrorHandlerMiddleware(config MiddlewareConfig) gin.HandlerFunc {
	return gin.CustomRecoveryWithWriter(nil, func(c *gin.Context, recovered interface{}) {
		var err error
		
		switch x := recovered.(type) {
		case string:
			err = New(ErrCodeDatabaseQuery, x) // Generic error code for panics
		case error:
			err = x
		default:
			err = New(ErrCodeDatabaseQuery, fmt.Sprintf("Unknown panic: %v", x))
		}
		
		// Handle the error
		handleError(c, err, config)
		
		// Abort the request
		c.Abort()
	})
}

// AIDEV-NOTE: 250903170000 Main error handling function
func handleError(c *gin.Context, err error, config MiddlewareConfig) {
	// Extract or create ServiceError
	var serviceErr *ServiceError
	var ok bool
	
	if serviceErr, ok = err.(*ServiceError); !ok {
		// Convert generic error to ServiceError
		serviceErr = Wrap(ErrCodeDatabaseQuery, "Internal server error", err)
		serviceErr.enrichFromContext(c.Request.Context())
	}
	
	// Ensure required fields are set
	if serviceErr.Timestamp.IsZero() {
		serviceErr.Timestamp = time.Now().UTC()
	}
	
	// Extract request metadata
	requestID := getRequestID(c)
	traceID := getTraceID(c)
	
	if requestID != "" && serviceErr.RequestID == "" {
		serviceErr.RequestID = requestID
	}
	
	if traceID != "" && serviceErr.TraceID == "" {
		serviceErr.TraceID = traceID
	}
	
	// Add request context
	enrichErrorWithRequest(serviceErr, c)
	
	// Log the error
	if config.Logger != nil {
		logError(config.Logger, serviceErr, c)
	}
	
	// Collect metrics
	if config.EnableMetrics {
		collectErrorMetrics(serviceErr, c)
	}
	
	// Call custom error callback
	if config.OnErrorCallback != nil {
		config.OnErrorCallback(serviceErr, c)
	}
	
	// Create error response
	response := createErrorResponse(serviceErr, config)
	
	// Set appropriate headers
	setErrorHeaders(c, serviceErr)
	
	// Send response
	c.JSON(serviceErr.HTTPStatus, response)
}

// AIDEV-NOTE: 250903170000 Helper function to enrich error with request information
func enrichErrorWithRequest(serviceErr *ServiceError, c *gin.Context) {
	// Add request information to context
	serviceErr.WithContext("method", c.Request.Method)
	serviceErr.WithContext("path", c.Request.URL.Path)
	serviceErr.WithContext("user_agent", c.Request.UserAgent())
	serviceErr.WithContext("remote_addr", c.ClientIP())
	
	// Add platform and username if available in path parameters
	if platform := c.Param("platform"); platform != "" {
		serviceErr.WithContext("platform", platform)
		serviceErr.WithTag("platform", platform)
	}
	
	if username := c.Param("username"); username != "" {
		serviceErr.WithContext("username", username)
		serviceErr.WithTag("username", username)
	}
	
	if kolID := c.Param("kol_id"); kolID != "" {
		serviceErr.WithContext("kol_id", kolID)
	}
	
	if jobID := c.Param("job_id"); jobID != "" {
		serviceErr.WithContext("job_id", jobID)
		serviceErr.WithTag("job_id", jobID)
	}
	
	// Add query parameters if relevant
	if query := c.Request.URL.RawQuery; query != "" {
		serviceErr.WithContext("query_params", query)
	}
}

// AIDEV-NOTE: 250903170000 Create formatted error response for client
func createErrorResponse(serviceErr *ServiceError, config MiddlewareConfig) ErrorResponse {
	response := ErrorResponse{
		Success:   false,
		Timestamp: serviceErr.Timestamp,
		RequestID: serviceErr.RequestID,
		TraceID:   serviceErr.TraceID,
		Error: ErrorDetails{
			Code:     string(serviceErr.Code),
			Message:  serviceErr.GetUserSafeMessage(),
			Category: string(serviceErr.Category),
		},
	}
	
	// Include details in development or if explicitly enabled
	if config.IncludeErrorDetails || strings.EqualFold(config.Environment, "development") {
		if serviceErr.Details != "" {
			response.Error.Details = serviceErr.Details
		} else if serviceErr.Cause != nil {
			response.Error.Details = serviceErr.Cause.Error()
		}
	}
	
	// Include context if enabled and not in production
	if config.IncludeContext && !strings.EqualFold(config.Environment, "production") {
		response.Error.Context = redactSensitiveFields(serviceErr.Context, config.SensitiveFields)
	}
	
	// Add retry information
	if serviceErr.RetryAfter != nil {
		retryAfterSeconds := int64(serviceErr.RetryAfter.Seconds())
		response.Error.RetryAfter = &retryAfterSeconds
	}
	
	// Add suggestions based on error category
	response.Error.Suggestions = generateSuggestions(serviceErr)
	
	// Apply custom error messages
	if customMessage, exists := config.CustomErrorMessages[string(serviceErr.Code)]; exists {
		response.Error.Message = customMessage
	}
	
	return response
}

// AIDEV-NOTE: 250903170000 Generate helpful suggestions based on error type
func generateSuggestions(serviceErr *ServiceError) []string {
	var suggestions []string
	
	switch serviceErr.Category {
	case CategoryValidation:
		suggestions = append(suggestions, "Check the request format and required parameters")
		suggestions = append(suggestions, "Verify that all required fields are provided")
		
	case CategoryAuth:
		suggestions = append(suggestions, "Verify your authentication credentials")
		suggestions = append(suggestions, "Check if your token has expired")
		
	case CategoryRateLimit:
		suggestions = append(suggestions, "Reduce request frequency")
		if serviceErr.RetryAfter != nil {
			suggestions = append(suggestions, fmt.Sprintf("Wait %v before retrying", *serviceErr.RetryAfter))
		}
		
	case CategoryNetwork:
		suggestions = append(suggestions, "Check your internet connection")
		suggestions = append(suggestions, "Try again in a few moments")
		
	case CategoryIntegration:
		suggestions = append(suggestions, "The external service may be temporarily unavailable")
		suggestions = append(suggestions, "Try again later or contact support if the issue persists")
		
	case CategoryCircuitBreaker:
		suggestions = append(suggestions, "The service is temporarily unavailable")
		if serviceErr.RetryAfter != nil {
			suggestions = append(suggestions, fmt.Sprintf("Wait at least %v before retrying", *serviceErr.RetryAfter))
		}
	}
	
	return suggestions
}

// AIDEV-NOTE: 250903170000 Set appropriate HTTP headers based on error type
func setErrorHeaders(c *gin.Context, serviceErr *ServiceError) {
	// Set Retry-After header for rate limiting and service unavailable errors
	if serviceErr.RetryAfter != nil {
		c.Header("Retry-After", fmt.Sprintf("%.0f", serviceErr.RetryAfter.Seconds()))
	}
	
	// Set cache control headers for different error types
	switch serviceErr.Category {
	case CategoryAuth:
		c.Header("Cache-Control", "no-cache, no-store, must-revalidate")
		c.Header("Pragma", "no-cache")
		
	case CategoryValidation:
		c.Header("Cache-Control", "no-cache")
		
	case CategoryRateLimit:
		c.Header("Cache-Control", "no-cache, no-store")
		
	default:
		c.Header("Cache-Control", "no-cache")
	}
	
	// Add error tracking headers
	if serviceErr.TraceID != "" {
		c.Header("X-Trace-ID", serviceErr.TraceID)
	}
	
	if serviceErr.RequestID != "" {
		c.Header("X-Request-ID", serviceErr.RequestID)
	}
	
	// Add error code header for easier client-side handling
	c.Header("X-Error-Code", string(serviceErr.Code))
	c.Header("X-Error-Category", string(serviceErr.Category))
}

// AIDEV-NOTE: 250903170000 Log error with structured information
func logError(log *logger.Logger, serviceErr *ServiceError, c *gin.Context) {
	fields := logger.Fields{
		"error_id":      serviceErr.Context["error_id"],
		"error_code":    serviceErr.Code,
		"error_category": serviceErr.Category,
		"http_status":   serviceErr.HTTPStatus,
		"method":        c.Request.Method,
		"path":          c.Request.URL.Path,
		"user_agent":    c.Request.UserAgent(),
		"remote_addr":   c.ClientIP(),
		"trace_id":      serviceErr.TraceID,
		"request_id":    serviceErr.RequestID,
		"user_id":       serviceErr.UserID,
		"operation_id":  serviceErr.OperationID,
		"severity":      serviceErr.Severity,
		"is_retryable":  serviceErr.IsRetryable,
	}
	
	// Add tags
	for k, v := range serviceErr.Tags {
		fields[fmt.Sprintf("tag_%s", k)] = v
	}
	
	// Add sanitized context
	for k, v := range serviceErr.Context {
		if !isSensitiveField(k, []string{"password", "token", "key", "secret"}) {
			fields[fmt.Sprintf("ctx_%s", k)] = v
		}
	}
	
	// Log with appropriate level based on severity
	entry := log.ErrorLog(serviceErr, "request_error", fields)
	
	switch serviceErr.Severity {
	case SeverityCritical:
		entry.Fatal("Critical error occurred")
	case SeverityHigh:
		entry.Error("High severity error occurred")
	case SeverityMedium:
		entry.Warn("Medium severity error occurred")
	default:
		entry.Info("Low severity error occurred")
	}
}

// AIDEV-NOTE: 250903170000 Collect error metrics for monitoring
func collectErrorMetrics(serviceErr *ServiceError, c *gin.Context) {
	// TODO: Integrate with metrics collection system (Prometheus, etc.)
	// This is a placeholder for metrics collection
	
	// Example metrics that should be collected:
	// - Error count by code, category, severity
	// - Error rate by endpoint
	// - Response time distribution
	// - Circuit breaker state changes
	// - Retry attempts and success rates
}

// AIDEV-NOTE: 250903170000 Utility functions for request metadata extraction
func getRequestID(c *gin.Context) string {
	// Try various common request ID headers
	headers := []string{"X-Request-ID", "X-Request-Id", "Request-ID", "Request-Id"}
	
	for _, header := range headers {
		if id := c.GetHeader(header); id != "" {
			return id
		}
	}
	
	// Check if stored in context
	if id, exists := c.Get("request_id"); exists {
		if idStr, ok := id.(string); ok {
			return idStr
		}
	}
	
	return ""
}

func getTraceID(c *gin.Context) string {
	// Try various common trace ID headers
	headers := []string{"X-Trace-ID", "X-Trace-Id", "Trace-ID", "Trace-Id", "X-B3-TraceId"}
	
	for _, header := range headers {
		if id := c.GetHeader(header); id != "" {
			return id
		}
	}
	
	// Check if stored in context
	if id, exists := c.Get("trace_id"); exists {
		if idStr, ok := id.(string); ok {
			return idStr
		}
	}
	
	return ""
}

// AIDEV-NOTE: 250903170000 Security functions for data redaction
func redactSensitiveFields(data map[string]interface{}, sensitiveFields []string) map[string]interface{} {
	if data == nil {
		return nil
	}
	
	redacted := make(map[string]interface{})
	
	for k, v := range data {
		if isSensitiveField(k, sensitiveFields) {
			redacted[k] = "[REDACTED]"
		} else {
			redacted[k] = v
		}
	}
	
	return redacted
}

func isSensitiveField(field string, sensitiveFields []string) bool {
	fieldLower := strings.ToLower(field)
	
	for _, sensitive := range sensitiveFields {
		if strings.Contains(fieldLower, strings.ToLower(sensitive)) {
			return true
		}
	}
	
	return false
}

// AIDEV-NOTE: 250903170000 Convenience functions for common error scenarios

// HandleValidationErrors processes validation errors and returns appropriate response
func HandleValidationErrors(c *gin.Context, validationErrs []string) {
	err := ValidationError("Validation failed")
	for i, validationErr := range validationErrs {
		err.WithContext(fmt.Sprintf("validation_error_%d", i), validationErr)
	}
	
	config := DefaultMiddlewareConfig()
	handleError(c, err, config)
}

// HandleDatabaseError processes database errors with circuit breaker integration
func HandleDatabaseError(c *gin.Context, err error, operation string) {
	dbErr := DatabaseErrorFromContext(c.Request.Context(), 
		fmt.Sprintf("Database operation failed: %s", operation), err)
	
	config := DefaultMiddlewareConfig()
	handleError(c, dbErr, config)
}

// HandleIntegrationError processes integration errors with retry information
func HandleIntegrationError(c *gin.Context, err error, service string) {
	integrationErr := FastAPIErrorFromContext(c.Request.Context(),
		fmt.Sprintf("Integration with %s failed", service), err)
	integrationErr.WithContext("service", service)
	
	config := DefaultMiddlewareConfig()
	handleError(c, integrationErr, config)
}

// AbortWithError is a convenience method for Gin handlers
func AbortWithError(c *gin.Context, err error) {
	config := DefaultMiddlewareConfig()
	handleError(c, err, config)
	c.Abort()
}

// AbortWithServiceError is a convenience method for Gin handlers with ServiceError
func AbortWithServiceError(c *gin.Context, serviceErr *ServiceError) {
	config := DefaultMiddlewareConfig()
	handleError(c, serviceErr, config)
	c.Abort()
}