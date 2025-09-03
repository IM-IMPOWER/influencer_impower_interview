// AIDEV-NOTE: 250102170000 Comprehensive error handling system for KOL platform
// Provides categorized errors, context wrapping, HTTP integration, and monitoring
package errors

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"runtime"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// AIDEV-NOTE: 250102170000 Error categories for proper error classification
type ErrorCategory string

const (
	CategoryValidation    ErrorCategory = "validation"
	CategoryDatabase      ErrorCategory = "database"
	CategoryNetwork       ErrorCategory = "network"
	CategoryIntegration   ErrorCategory = "integration"
	CategoryBusiness      ErrorCategory = "business"
	CategoryConfiguration ErrorCategory = "configuration"
	CategorySecurity      ErrorCategory = "security"
	CategoryRateLimit     ErrorCategory = "rate_limit"
)

// AIDEV-NOTE: 250102170000 Error codes for client identification
type ErrorCode string

const (
	// Validation errors
	CodeInvalidInput      ErrorCode = "INVALID_INPUT"
	CodeMissingField      ErrorCode = "MISSING_FIELD"
	CodeInvalidFormat     ErrorCode = "INVALID_FORMAT"
	CodeValidationFailed  ErrorCode = "VALIDATION_FAILED"
	
	// Database errors
	CodeDatabaseConnection ErrorCode = "DATABASE_CONNECTION"
	CodeDatabaseQuery      ErrorCode = "DATABASE_QUERY"
	CodeRecordNotFound     ErrorCode = "RECORD_NOT_FOUND"
	CodeDuplicateRecord    ErrorCode = "DUPLICATE_RECORD"
	
	// Network errors
	CodeNetworkTimeout     ErrorCode = "NETWORK_TIMEOUT"
	CodeNetworkConnection  ErrorCode = "NETWORK_CONNECTION"
	CodeServiceUnavailable ErrorCode = "SERVICE_UNAVAILABLE"
	
	// Integration errors
	CodeIntegrationFailed  ErrorCode = "INTEGRATION_FAILED"
	CodeCircuitBreakerOpen ErrorCode = "CIRCUIT_BREAKER_OPEN"
	CodeUpstreamError      ErrorCode = "UPSTREAM_ERROR"
	
	// Business logic errors
	CodeBusinessRule       ErrorCode = "BUSINESS_RULE_VIOLATION"
	CodeInsufficientFunds  ErrorCode = "INSUFFICIENT_FUNDS"
	CodeResourceLimit      ErrorCode = "RESOURCE_LIMIT_EXCEEDED"
	
	// Configuration errors
	CodeMissingConfig      ErrorCode = "MISSING_CONFIGURATION"
	CodeInvalidConfig      ErrorCode = "INVALID_CONFIGURATION"
	
	// Security errors
	CodeUnauthorized       ErrorCode = "UNAUTHORIZED"
	CodeForbidden         ErrorCode = "FORBIDDEN"
	CodeInvalidToken      ErrorCode = "INVALID_TOKEN"
	
	// Rate limiting
	CodeRateLimitExceeded  ErrorCode = "RATE_LIMIT_EXCEEDED"
)

// AIDEV-NOTE: 250102170000 Core error interface with enhanced context
type KOLError interface {
	error
	Category() ErrorCategory
	Code() ErrorCode
	HTTPStatus() int
	Context() map[string]interface{}
	Stack() string
	Unwrap() error
	UserMessage() string
	IsRetryable() bool
}

// AIDEV-NOTE: 250102170000 Base error implementation
type baseError struct {
	category    ErrorCategory
	code        ErrorCode
	message     string
	userMsg     string
	context     map[string]interface{}
	stack       string
	cause       error
	timestamp   time.Time
	retryable   bool
}

// Error implements the error interface
func (e *baseError) Error() string {
	if e.cause != nil {
		return fmt.Sprintf("%s: %s (caused by: %v)", e.code, e.message, e.cause)
	}
	return fmt.Sprintf("%s: %s", e.code, e.message)
}

// Category returns the error category
func (e *baseError) Category() ErrorCategory {
	return e.category
}

// Code returns the error code
func (e *baseError) Code() ErrorCode {
	return e.code
}

// HTTPStatus returns the appropriate HTTP status code
func (e *baseError) HTTPStatus() int {
	switch e.category {
	case CategoryValidation:
		return http.StatusBadRequest
	case CategoryDatabase:
		if e.code == CodeRecordNotFound {
			return http.StatusNotFound
		}
		if e.code == CodeDuplicateRecord {
			return http.StatusConflict
		}
		return http.StatusInternalServerError
	case CategoryNetwork, CategoryIntegration:
		if e.code == CodeServiceUnavailable || e.code == CodeCircuitBreakerOpen {
			return http.StatusServiceUnavailable
		}
		if e.code == CodeNetworkTimeout {
			return http.StatusGatewayTimeout
		}
		return http.StatusBadGateway
	case CategoryBusiness:
		if e.code == CodeBusinessRule {
			return http.StatusUnprocessableEntity
		}
		return http.StatusBadRequest
	case CategoryConfiguration:
		return http.StatusInternalServerError
	case CategorySecurity:
		if e.code == CodeUnauthorized || e.code == CodeInvalidToken {
			return http.StatusUnauthorized
		}
		return http.StatusForbidden
	case CategoryRateLimit:
		return http.StatusTooManyRequests
	default:
		return http.StatusInternalServerError
	}
}

// Context returns error context information
func (e *baseError) Context() map[string]interface{} {
	if e.context == nil {
		return make(map[string]interface{})
	}
	return e.context
}

// Stack returns the stack trace
func (e *baseError) Stack() string {
	return e.stack
}

// Unwrap returns the underlying error
func (e *baseError) Unwrap() error {
	return e.cause
}

// UserMessage returns a user-friendly error message
func (e *baseError) UserMessage() string {
	if e.userMsg != "" {
		return e.userMsg
	}
	
	// Default user messages based on category
	switch e.category {
	case CategoryValidation:
		return "The provided information is invalid. Please check your input and try again."
	case CategoryDatabase:
		if e.code == CodeRecordNotFound {
			return "The requested resource was not found."
		}
		return "A data storage issue occurred. Please try again later."
	case CategoryNetwork, CategoryIntegration:
		return "A temporary service issue occurred. Please try again in a few moments."
	case CategoryBusiness:
		return "This operation cannot be completed due to business rules."
	case CategorySecurity:
		return "Access denied. Please check your credentials."
	case CategoryRateLimit:
		return "Too many requests. Please wait a moment before trying again."
	default:
		return "An unexpected error occurred. Please try again later."
	}
}

// IsRetryable indicates if the error condition might be temporary
func (e *baseError) IsRetryable() bool {
	return e.retryable
}

// AIDEV-NOTE: 250102170000 Error builder for fluent construction
type ErrorBuilder struct {
	err *baseError
}

// New creates a new error builder
func New(category ErrorCategory, code ErrorCode, message string) *ErrorBuilder {
	stack := captureStack(2)
	
	return &ErrorBuilder{
		err: &baseError{
			category:  category,
			code:      code,
			message:   message,
			context:   make(map[string]interface{}),
			stack:     stack,
			timestamp: time.Now(),
			retryable: isRetryableByDefault(category, code),
		},
	}
}

// WithCause adds the underlying cause
func (eb *ErrorBuilder) WithCause(cause error) *ErrorBuilder {
	eb.err.cause = cause
	return eb
}

// WithContext adds context information
func (eb *ErrorBuilder) WithContext(key string, value interface{}) *ErrorBuilder {
	eb.err.context[key] = value
	return eb
}

// WithContextMap adds multiple context values
func (eb *ErrorBuilder) WithContextMap(ctx map[string]interface{}) *ErrorBuilder {
	for k, v := range ctx {
		eb.err.context[k] = v
	}
	return eb
}

// WithUserMessage sets a user-friendly message
func (eb *ErrorBuilder) WithUserMessage(msg string) *ErrorBuilder {
	eb.err.userMsg = msg
	return eb
}

// WithRetryable sets the retryable flag
func (eb *ErrorBuilder) WithRetryable(retryable bool) *ErrorBuilder {
	eb.err.retryable = retryable
	return eb
}

// Build creates the final error
func (eb *ErrorBuilder) Build() KOLError {
	return eb.err
}

// AIDEV-NOTE: 250102170000 Convenience functions for common error types

// ValidationError creates a validation error
func ValidationError(message string) *ErrorBuilder {
	return New(CategoryValidation, CodeValidationFailed, message)
}

// DatabaseError creates a database error
func DatabaseError(message string) *ErrorBuilder {
	return New(CategoryDatabase, CodeDatabaseQuery, message)
}

// NetworkError creates a network error
func NetworkError(message string) *ErrorBuilder {
	return New(CategoryNetwork, CodeNetworkConnection, message)
}

// IntegrationError creates an integration error
func IntegrationError(message string) *ErrorBuilder {
	return New(CategoryIntegration, CodeIntegrationFailed, message)
}

// BusinessError creates a business logic error
func BusinessError(message string) *ErrorBuilder {
	return New(CategoryBusiness, CodeBusinessRule, message)
}

// NotFoundError creates a not found error
func NotFoundError(resource string) KOLError {
	return New(CategoryDatabase, CodeRecordNotFound, fmt.Sprintf("%s not found", resource)).
		WithUserMessage(fmt.Sprintf("The requested %s was not found.", resource)).
		Build()
}

// UnauthorizedError creates an unauthorized error
func UnauthorizedError(message string) KOLError {
	return New(CategorySecurity, CodeUnauthorized, message).
		WithUserMessage("Authentication required.").
		Build()
}

// RateLimitError creates a rate limit error
func RateLimitError(limit int) KOLError {
	return New(CategoryRateLimit, CodeRateLimitExceeded, fmt.Sprintf("Rate limit of %d exceeded", limit)).
		WithUserMessage("Too many requests. Please wait before trying again.").
		WithRetryable(true).
		Build()
}

// AIDEV-NOTE: 250102170000 Error wrapping utilities

// Wrap wraps an existing error with additional context
func Wrap(err error, category ErrorCategory, code ErrorCode, message string) KOLError {
	if err == nil {
		return nil
	}
	
	// If it's already a KOLError, preserve the original context
	if kolErr, ok := err.(KOLError); ok {
		return New(category, code, message).
			WithCause(kolErr).
			WithContextMap(kolErr.Context()).
			Build()
	}
	
	return New(category, code, message).
		WithCause(err).
		Build()
}

// WrapDatabase wraps a database error
func WrapDatabase(err error, operation string) KOLError {
	if err == nil {
		return nil
	}
	
	code := CodeDatabaseQuery
	if strings.Contains(err.Error(), "no rows") {
		code = CodeRecordNotFound
	} else if strings.Contains(err.Error(), "duplicate") {
		code = CodeDuplicateRecord
	}
	
	return Wrap(err, CategoryDatabase, code, fmt.Sprintf("Database %s failed", operation)).
		WithContext("operation", operation)
}

// WrapIntegration wraps an integration error
func WrapIntegration(err error, service string) KOLError {
	if err == nil {
		return nil
	}
	
	code := CodeIntegrationFailed
	if strings.Contains(err.Error(), "circuit breaker") {
		code = CodeCircuitBreakerOpen
	} else if strings.Contains(err.Error(), "timeout") {
		code = CodeNetworkTimeout
	}
	
	return Wrap(err, CategoryIntegration, code, fmt.Sprintf("Integration with %s failed", service)).
		WithContext("service", service).
		WithRetryable(true)
}

// AIDEV-NOTE: 250102170000 HTTP error response handling

// ErrorResponse represents the JSON error response
type ErrorResponse struct {
	Error       ErrorDetail `json:"error"`
	RequestID   string      `json:"request_id,omitempty"`
	Timestamp   time.Time   `json:"timestamp"`
	Path        string      `json:"path,omitempty"`
}

type ErrorDetail struct {
	Code        string                 `json:"code"`
	Message     string                 `json:"message"`
	Category    string                 `json:"category"`
	Context     map[string]interface{} `json:"context,omitempty"`
	Retryable   bool                   `json:"retryable"`
	TraceID     string                 `json:"trace_id,omitempty"`
}

// ToErrorResponse converts a KOLError to an ErrorResponse
func ToErrorResponse(err error, c *gin.Context) *ErrorResponse {
	var kolErr KOLError
	var ok bool
	
	if kolErr, ok = err.(KOLError); !ok {
		// Convert standard error to internal server error
		kolErr = New(CategoryConfiguration, CodeMissingConfig, err.Error()).Build()
	}
	
	requestID, _ := c.Get("request_id")
	traceID, _ := c.Get("trace_id")
	
	return &ErrorResponse{
		Error: ErrorDetail{
			Code:      string(kolErr.Code()),
			Message:   kolErr.UserMessage(),
			Category:  string(kolErr.Category()),
			Context:   kolErr.Context(),
			Retryable: kolErr.IsRetryable(),
			TraceID:   toString(traceID),
		},
		RequestID: toString(requestID),
		Timestamp: time.Now(),
		Path:      c.Request.URL.Path,
	}
}

// AIDEV-NOTE: 250102170000 Error middleware for Gin
func ErrorMiddleware() gin.HandlerFunc {
	return gin.CustomRecovery(func(c *gin.Context, recovered interface{}) {
		var err error
		
		switch x := recovered.(type) {
		case string:
			err = errors.New(x)
		case error:
			err = x
		default:
			err = fmt.Errorf("unknown panic: %v", x)
		}
		
		// Convert to KOL error
		kolErr := New(CategoryConfiguration, CodeMissingConfig, "Internal server error").
			WithCause(err).
			WithContext("panic", true).
			Build()
		
		HandleError(c, kolErr)
	})
}

// HandleError handles error responses in HTTP handlers
func HandleError(c *gin.Context, err error) {
	if err == nil {
		return
	}
	
	var kolErr KOLError
	var ok bool
	
	if kolErr, ok = err.(KOLError); !ok {
		kolErr = New(CategoryConfiguration, CodeMissingConfig, err.Error()).Build()
	}
	
	response := ToErrorResponse(kolErr, c)
	status := kolErr.HTTPStatus()
	
	// Log the error with context
	if logger, exists := c.Get("logger"); exists {
		if log, ok := logger.(interface {
			ErrorLog(error, string, map[string]interface{})
		}); ok {
			logContext := map[string]interface{}{
				"error_code":     kolErr.Code(),
				"error_category": kolErr.Category(),
				"http_status":    status,
				"path":           c.Request.URL.Path,
				"method":         c.Request.Method,
				"retryable":      kolErr.IsRetryable(),
			}
			
			// Add error context
			for k, v := range kolErr.Context() {
				logContext[k] = v
			}
			
			log.ErrorLog(kolErr, "http_error", logContext)
		}
	}
	
	c.JSON(status, response)
	c.Abort()
}

// AIDEV-NOTE: 250102170000 Utility functions

// captureStack captures the current stack trace
func captureStack(skip int) string {
	const maxStackSize = 32
	var buf [maxStackSize * 1024]byte
	n := runtime.Stack(buf[:], false)
	return string(buf[:n])
}

// isRetryableByDefault determines if an error is retryable by default
func isRetryableByDefault(category ErrorCategory, code ErrorCode) bool {
	switch category {
	case CategoryNetwork, CategoryIntegration:
		return code != CodeServiceUnavailable
	case CategoryDatabase:
		return code == CodeDatabaseConnection
	case CategoryRateLimit:
		return true
	default:
		return false
	}
}

// toString safely converts interface{} to string
func toString(v interface{}) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}

// AIDEV-NOTE: 250102170000 Context helpers for request tracking

// ContextKey type for context keys
type ContextKey string

const (
	ContextKeyRequestID ContextKey = "request_id"
	ContextKeyUserID    ContextKey = "user_id"
	ContextKeyTraceID   ContextKey = "trace_id"
	ContextKeyOperation ContextKey = "operation"
)

// WithRequestContext adds request context to an error
func WithRequestContext(ctx context.Context, err KOLError) KOLError {
	if err == nil {
		return nil
	}
	
	builder := New(err.Category(), err.Code(), err.Error()).
		WithCause(err.Unwrap()).
		WithContextMap(err.Context())
	
	if requestID := ctx.Value(ContextKeyRequestID); requestID != nil {
		builder = builder.WithContext("request_id", requestID)
	}
	
	if userID := ctx.Value(ContextKeyUserID); userID != nil {
		builder = builder.WithContext("user_id", userID)
	}
	
	if traceID := ctx.Value(ContextKeyTraceID); traceID != nil {
		builder = builder.WithContext("trace_id", traceID)
	}
	
	if operation := ctx.Value(ContextKeyOperation); operation != nil {
		builder = builder.WithContext("operation", operation)
	}
	
	return builder.Build()
}

// AIDEV-NOTE: 250102170000 Error metrics collection interface
type ErrorMetrics interface {
	RecordError(category ErrorCategory, code ErrorCode, duration time.Duration)
	GetErrorRate(category ErrorCategory) float64
	GetErrorCount(category ErrorCategory, code ErrorCode) int64
}

// DefaultErrorMetrics is a simple in-memory metrics collector
type DefaultErrorMetrics struct {
	errors map[string]int64
}

// NewDefaultErrorMetrics creates a new default metrics collector
func NewDefaultErrorMetrics() *DefaultErrorMetrics {
	return &DefaultErrorMetrics{
		errors: make(map[string]int64),
	}
}

// RecordError records an error occurrence
func (m *DefaultErrorMetrics) RecordError(category ErrorCategory, code ErrorCode, duration time.Duration) {
	key := fmt.Sprintf("%s:%s", category, code)
	m.errors[key]++
}

// GetErrorRate returns the error rate for a category (placeholder implementation)
func (m *DefaultErrorMetrics) GetErrorRate(category ErrorCategory) float64 {
	// In a real implementation, this would calculate rate over time
	return 0.0
}

// GetErrorCount returns the error count for a specific error type
func (m *DefaultErrorMetrics) GetErrorCount(category ErrorCategory, code ErrorCode) int64 {
	key := fmt.Sprintf("%s:%s", category, code)
	return m.errors[key]
}

// IsKOLError checks if an error is a KOLError
func IsKOLError(err error) bool {
	_, ok := err.(KOLError)
	return ok
}

// GetErrorCode safely extracts the error code from an error
func GetErrorCode(err error) ErrorCode {
	if kolErr, ok := err.(KOLError); ok {
		return kolErr.Code()
	}
	return CodeMissingConfig // Default for unknown errors
}