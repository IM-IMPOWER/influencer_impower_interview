// AIDEV-NOTE: 250102170000 Comprehensive tests for error handling system
// Tests error construction, wrapping, HTTP integration, and monitoring features
package errors

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestErrorBuilder(t *testing.T) {
	err := New(CategoryValidation, CodeInvalidInput, "Invalid user input").
		WithContext("field", "email").
		WithContext("value", "invalid-email").
		WithUserMessage("Please provide a valid email address.").
		WithRetryable(false).
		Build()

	if err.Category() != CategoryValidation {
		t.Errorf("Expected category %s, got %s", CategoryValidation, err.Category())
	}

	if err.Code() != CodeInvalidInput {
		t.Errorf("Expected code %s, got %s", CodeInvalidInput, err.Code())
	}

	if err.UserMessage() != "Please provide a valid email address." {
		t.Errorf("Expected custom user message, got %s", err.UserMessage())
	}

	if err.IsRetryable() {
		t.Error("Expected error to not be retryable")
	}

	context := err.Context()
	if context["field"] != "email" {
		t.Errorf("Expected field context 'email', got %v", context["field"])
	}

	if context["value"] != "invalid-email" {
		t.Errorf("Expected value context 'invalid-email', got %v", context["value"])
	}

	if err.HTTPStatus() != http.StatusBadRequest {
		t.Errorf("Expected HTTP status %d, got %d", http.StatusBadRequest, err.HTTPStatus())
	}
}

func TestErrorWrapping(t *testing.T) {
	originalErr := errors.New("original database error")
	
	wrappedErr := WrapDatabase(originalErr, "user_query")
	
	if wrappedErr.Category() != CategoryDatabase {
		t.Errorf("Expected category %s, got %s", CategoryDatabase, wrappedErr.Category())
	}
	
	if wrappedErr.Code() != CodeDatabaseQuery {
		t.Errorf("Expected code %s, got %s", CodeDatabaseQuery, wrappedErr.Code())
	}
	
	if wrappedErr.Unwrap() != originalErr {
		t.Error("Expected wrapped error to preserve original error")
	}
	
	context := wrappedErr.Context()
	if context["operation"] != "user_query" {
		t.Errorf("Expected operation context 'user_query', got %v", context["operation"])
	}
}

func TestConvenienceFunctions(t *testing.T) {
	tests := []struct {
		name         string
		createError  func() KOLError
		expectCode   ErrorCode
		expectStatus int
		expectCategory ErrorCategory
	}{
		{
			name:         "NotFoundError",
			createError:  func() KOLError { return NotFoundError("user") },
			expectCode:   CodeRecordNotFound,
			expectStatus: http.StatusNotFound,
			expectCategory: CategoryDatabase,
		},
		{
			name:         "UnauthorizedError",
			createError:  func() KOLError { return UnauthorizedError("invalid token") },
			expectCode:   CodeUnauthorized,
			expectStatus: http.StatusUnauthorized,
			expectCategory: CategorySecurity,
		},
		{
			name:         "RateLimitError",
			createError:  func() KOLError { return RateLimitError(100) },
			expectCode:   CodeRateLimitExceeded,
			expectStatus: http.StatusTooManyRequests,
			expectCategory: CategoryRateLimit,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.createError()
			
			if err.Code() != tt.expectCode {
				t.Errorf("Expected code %s, got %s", tt.expectCode, err.Code())
			}
			
			if err.HTTPStatus() != tt.expectStatus {
				t.Errorf("Expected HTTP status %d, got %d", tt.expectStatus, err.HTTPStatus())
			}
			
			if err.Category() != tt.expectCategory {
				t.Errorf("Expected category %s, got %s", tt.expectCategory, err.Category())
			}
		})
	}
}

func TestHTTPStatusMapping(t *testing.T) {
	tests := []struct {
		category ErrorCategory
		code     ErrorCode
		expected int
	}{
		{CategoryValidation, CodeInvalidInput, http.StatusBadRequest},
		{CategoryDatabase, CodeRecordNotFound, http.StatusNotFound},
		{CategoryDatabase, CodeDuplicateRecord, http.StatusConflict},
		{CategoryDatabase, CodeDatabaseQuery, http.StatusInternalServerError},
		{CategoryNetwork, CodeNetworkTimeout, http.StatusGatewayTimeout},
		{CategoryIntegration, CodeServiceUnavailable, http.StatusServiceUnavailable},
		{CategoryIntegration, CodeCircuitBreakerOpen, http.StatusServiceUnavailable},
		{CategoryBusiness, CodeBusinessRule, http.StatusUnprocessableEntity},
		{CategorySecurity, CodeUnauthorized, http.StatusUnauthorized},
		{CategorySecurity, CodeForbidden, http.StatusForbidden},
		{CategoryRateLimit, CodeRateLimitExceeded, http.StatusTooManyRequests},
	}

	for _, tt := range tests {
		t.Run(string(tt.category)+"_"+string(tt.code), func(t *testing.T) {
			err := New(tt.category, tt.code, "test message").Build()
			
			if err.HTTPStatus() != tt.expected {
				t.Errorf("Expected HTTP status %d for %s:%s, got %d", 
					tt.expected, tt.category, tt.code, err.HTTPStatus())
			}
		})
	}
}

func TestUserMessages(t *testing.T) {
	tests := []struct {
		category ErrorCategory
		expected string
	}{
		{CategoryValidation, "The provided information is invalid. Please check your input and try again."},
		{CategoryNetwork, "A temporary service issue occurred. Please try again in a few moments."},
		{CategoryBusiness, "This operation cannot be completed due to business rules."},
		{CategorySecurity, "Access denied. Please check your credentials."},
		{CategoryRateLimit, "Too many requests. Please wait a moment before trying again."},
	}

	for _, tt := range tests {
		t.Run(string(tt.category), func(t *testing.T) {
			err := New(tt.category, CodeValidationFailed, "test").Build()
			
			if err.UserMessage() != tt.expected {
				t.Errorf("Expected user message %q for %s, got %q", 
					tt.expected, tt.category, err.UserMessage())
			}
		})
	}
}

func TestRetryableLogic(t *testing.T) {
	tests := []struct {
		category  ErrorCategory
		code      ErrorCode
		retryable bool
	}{
		{CategoryNetwork, CodeNetworkConnection, true},
		{CategoryNetwork, CodeServiceUnavailable, false},
		{CategoryIntegration, CodeIntegrationFailed, true},
		{CategoryDatabase, CodeDatabaseConnection, true},
		{CategoryDatabase, CodeRecordNotFound, false},
		{CategoryRateLimit, CodeRateLimitExceeded, true},
		{CategoryValidation, CodeInvalidInput, false},
	}

	for _, tt := range tests {
		t.Run(string(tt.category)+"_"+string(tt.code), func(t *testing.T) {
			err := New(tt.category, tt.code, "test").Build()
			
			if err.IsRetryable() != tt.retryable {
				t.Errorf("Expected retryable %v for %s:%s, got %v", 
					tt.retryable, tt.category, tt.code, err.IsRetryable())
			}
		})
	}
}

func TestRequestContext(t *testing.T) {
	ctx := context.Background()
	ctx = context.WithValue(ctx, ContextKeyRequestID, "req-123")
	ctx = context.WithValue(ctx, ContextKeyUserID, "user-456")
	ctx = context.WithValue(ctx, ContextKeyTraceID, "trace-789")
	ctx = context.WithValue(ctx, ContextKeyOperation, "test_operation")

	originalErr := New(CategoryValidation, CodeInvalidInput, "test error").Build()
	contextErr := WithRequestContext(ctx, originalErr)

	context := contextErr.Context()
	
	if context["request_id"] != "req-123" {
		t.Errorf("Expected request_id 'req-123', got %v", context["request_id"])
	}
	
	if context["user_id"] != "user-456" {
		t.Errorf("Expected user_id 'user-456', got %v", context["user_id"])
	}
	
	if context["trace_id"] != "trace-789" {
		t.Errorf("Expected trace_id 'trace-789', got %v", context["trace_id"])
	}
	
	if context["operation"] != "test_operation" {
		t.Errorf("Expected operation 'test_operation', got %v", context["operation"])
	}
}

func TestErrorResponse(t *testing.T) {
	gin.SetMode(gin.TestMode)
	
	router := gin.New()
	router.Use(func(c *gin.Context) {
		c.Set("request_id", "test-request-123")
		c.Set("trace_id", "test-trace-456")
		c.Next()
	})
	
	router.GET("/test", func(c *gin.Context) {
		err := New(CategoryValidation, CodeInvalidInput, "Invalid input data").
			WithContext("field", "email").
			WithUserMessage("Please provide a valid email address.").
			Build()
		
		HandleError(c, err)
	})

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	router.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status %d, got %d", http.StatusBadRequest, w.Code)
	}

	if w.Header().Get("Content-Type") != "application/json; charset=utf-8" {
		t.Errorf("Expected JSON content type, got %s", w.Header().Get("Content-Type"))
	}

	// Check that response body contains expected fields
	body := w.Body.String()
	expectedFields := []string{
		`"code":"INVALID_INPUT"`,
		`"message":"Please provide a valid email address."`,
		`"category":"validation"`,
		`"request_id":"test-request-123"`,
		`"trace_id":"test-trace-456"`,
	}

	for _, field := range expectedFields {
		if !contains(body, field) {
			t.Errorf("Expected response body to contain %s, got: %s", field, body)
		}
	}
}

func TestErrorMetrics(t *testing.T) {
	metrics := NewDefaultErrorMetrics()

	// Record some errors
	metrics.RecordError(CategoryDatabase, CodeRecordNotFound, 0)
	metrics.RecordError(CategoryDatabase, CodeRecordNotFound, 0)
	metrics.RecordError(CategoryValidation, CodeInvalidInput, 0)

	// Check counts
	dbCount := metrics.GetErrorCount(CategoryDatabase, CodeRecordNotFound)
	if dbCount != 2 {
		t.Errorf("Expected database error count 2, got %d", dbCount)
	}

	validationCount := metrics.GetErrorCount(CategoryValidation, CodeInvalidInput)
	if validationCount != 1 {
		t.Errorf("Expected validation error count 1, got %d", validationCount)
	}

	// Check non-existent error
	nonExistentCount := metrics.GetErrorCount(CategoryNetwork, CodeNetworkTimeout)
	if nonExistentCount != 0 {
		t.Errorf("Expected non-existent error count 0, got %d", nonExistentCount)
	}
}

func TestErrorChaining(t *testing.T) {
	rootErr := errors.New("root cause")
	
	level1Err := New(CategoryDatabase, CodeDatabaseQuery, "Level 1 error").
		WithCause(rootErr).
		Build()
	
	level2Err := New(CategoryIntegration, CodeIntegrationFailed, "Level 2 error").
		WithCause(level1Err).
		Build()

	// Test unwrapping
	if level2Err.Unwrap() != level1Err {
		t.Error("Level 2 error should unwrap to level 1 error")
	}

	if level1Err.Unwrap() != rootErr {
		t.Error("Level 1 error should unwrap to root error")
	}

	// Test error message formatting
	expectedPattern := "INTEGRATION_FAILED: Level 2 error (caused by:"
	if !contains(level2Err.Error(), expectedPattern) {
		t.Errorf("Expected error message to contain %q, got: %s", expectedPattern, level2Err.Error())
	}
}

func TestUtilityFunctions(t *testing.T) {
	// Test IsKOLError
	kolErr := New(CategoryValidation, CodeInvalidInput, "test").Build()
	stdErr := errors.New("standard error")

	if !IsKOLError(kolErr) {
		t.Error("IsKOLError should return true for KOLError")
	}

	if IsKOLError(stdErr) {
		t.Error("IsKOLError should return false for standard error")
	}

	// Test GetErrorCode
	code := GetErrorCode(kolErr)
	if code != CodeInvalidInput {
		t.Errorf("Expected error code %s, got %s", CodeInvalidInput, code)
	}

	stdCode := GetErrorCode(stdErr)
	if stdCode != CodeMissingConfig {
		t.Errorf("Expected default error code %s, got %s", CodeMissingConfig, stdCode)
	}
}

func TestIntegrationWrapping(t *testing.T) {
	// Test circuit breaker error detection
	circuitErr := errors.New("circuit breaker is OPEN")
	wrappedCircuitErr := WrapIntegration(circuitErr, "fastapi")

	if wrappedCircuitErr.Code() != CodeCircuitBreakerOpen {
		t.Errorf("Expected circuit breaker code, got %s", wrappedCircuitErr.Code())
	}

	// Test timeout error detection
	timeoutErr := errors.New("request timeout after 30s")
	wrappedTimeoutErr := WrapIntegration(timeoutErr, "fastapi")

	if wrappedTimeoutErr.Code() != CodeNetworkTimeout {
		t.Errorf("Expected timeout code, got %s", wrappedTimeoutErr.Code())
	}

	// Test generic integration error
	genericErr := errors.New("service unavailable")
	wrappedGenericErr := WrapIntegration(genericErr, "fastapi")

	if wrappedGenericErr.Code() != CodeIntegrationFailed {
		t.Errorf("Expected integration failed code, got %s", wrappedGenericErr.Code())
	}

	// All integration errors should be retryable
	if !wrappedCircuitErr.IsRetryable() {
		t.Error("Circuit breaker errors should be retryable")
	}

	if !wrappedTimeoutErr.IsRetryable() {
		t.Error("Timeout errors should be retryable")
	}

	if !wrappedGenericErr.IsRetryable() {
		t.Error("Generic integration errors should be retryable")
	}
}

// Helper function to check if string contains substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		(s == substr || (len(s) > len(substr) && 
			(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr ||
				func() bool {
					for i := 1; i < len(s)-len(substr)+1; i++ {
						if s[i:i+len(substr)] == substr {
							return true
						}
					}
					return false
				}())))
}