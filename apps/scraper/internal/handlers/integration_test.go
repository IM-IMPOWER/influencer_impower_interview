// AIDEV-NOTE: 250102120330 Comprehensive tests for integration handlers
// Table-driven tests for POC2/POC4 integration endpoints
package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"kol-scraper/internal/fastapi"
	"kol-scraper/internal/models"
	"kol-scraper/pkg/logger"
)

// AIDEV-NOTE: MockFastAPIClient for testing
type MockFastAPIClient struct {
	mock.Mock
}

func (m *MockFastAPIClient) MatchKOLs(ctx context.Context, req *fastapi.MatchKOLsRequest) (*fastapi.MatchKOLsResponse, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*fastapi.MatchKOLsResponse), args.Error(1)
}

func (m *MockFastAPIClient) OptimizeBudget(ctx context.Context, req *fastapi.OptimizeBudgetRequest) (*fastapi.OptimizeBudgetResponse, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*fastapi.OptimizeBudgetResponse), args.Error(1)
}

func (m *MockFastAPIClient) HealthCheck(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

func (m *MockFastAPIClient) Close() error {
	args := m.Called()
	return args.Error(0)
}

// AIDEV-NOTE: Test fixtures for KOL matching
var validKOLMatchingRequest = models.KOLMatchingRequest{
	CampaignBrief: "Looking for tech influencers to promote our new app",
	Budget:        5000.0,
	TargetTier:    []models.KOLTier{models.TierMicro, models.TierMid},
	Platforms:     []models.Platform{models.PlatformTikTok, models.PlatformInstagram},
	Categories:    []models.Category{models.CategoryTech, models.CategoryEntertainment},
	MinFollowers:  10000,
	MaxFollowers:  500000,
	MaxResults:    20,
}

var sampleMatchKOLsResponse = &fastapi.MatchKOLsResponse{
	Matches: []fastapi.KOLMatch{
		{
			KOLID:          "kol-123",
			Username:       "techinfluencer1",
			Platform:       "tiktok",
			DisplayName:    "Tech Influencer",
			FollowerCount:  150000,
			EngagementRate: 4.5,
			Score:          8.7,
			Tier:           "mid",
			Category:       "tech",
			EstimatedCost:  2500.0,
			Reasoning:      "High engagement in tech content",
		},
		{
			KOLID:          "kol-456",
			Username:       "techreviewer",
			Platform:       "instagram",
			DisplayName:    "Tech Reviewer",
			FollowerCount:  75000,
			EngagementRate: 6.2,
			Score:          8.2,
			Tier:           "micro",
			Category:       "tech",
			EstimatedCost:  1800.0,
			Reasoning:      "Excellent tech review content",
		},
	},
	Meta: fastapi.MatchMeta{
		TotalFound:    50,
		QueryTime:     125.5,
		AlgorithmUsed: "ml_matching_v2",
		Confidence:    0.87,
	},
}

// AIDEV-NOTE: Test fixtures for budget optimization
var validBudgetOptimizationRequest = models.BudgetOptimizationRequest{
	TotalBudget:   10000.0,
	CampaignGoals: []string{"brand_awareness", "engagement"},
	KOLCandidates: []string{"kol-123", "kol-456", "kol-789"},
	TargetReach:   500000,
	TargetEngagement: 5.0,
	Constraints: models.BudgetConstraints{
		MaxKOLs: 5,
		MinKOLs: 2,
	},
}

var sampleOptimizeBudgetResponse = &fastapi.OptimizeBudgetResponse{
	Allocation: []fastapi.BudgetAllocation{
		{
			KOLID:           "kol-123",
			Username:        "techinfluencer1",
			Platform:        "tiktok",
			AllocatedBudget: 4000.0,
			EstimatedReach:  200000,
			EstimatedEngagement: 5.2,
			Priority:        1,
			Reasoning:       "High ROI potential",
		},
		{
			KOLID:           "kol-456",
			Username:        "techreviewer",
			Platform:        "instagram",
			AllocatedBudget: 3000.0,
			EstimatedReach:  150000,
			EstimatedEngagement: 6.0,
			Priority:        2,
			Reasoning:       "Strong engagement rate",
		},
	},
	Summary: fastapi.BudgetSummary{
		TotalAllocated:         7000.0,
		RemainingBudget:        3000.0,
		EstimatedTotalReach:    350000,
		EstimatedAvgEngagement: 5.6,
		OptimalKOLsSelected:    2,
		EfficiencyScore:        0.89,
	},
	Meta: fastapi.OptimizationMeta{
		OptimizationTime: 230.8,
		AlgorithmUsed:    "genetic_optimization_v3",
		Iterations:       15,
		Confidence:       0.91,
	},
}

// AIDEV-NOTE: Test suite for MatchKOLs endpoint
func TestMatchKOLs(t *testing.T) {
	tests := []struct {
		name           string
		setupMock      func(*MockFastAPIClient)
		request        interface{}
		expectedStatus int
		expectedError  string
		validateResponse func(*testing.T, map[string]interface{})
	}{
		{
			name: "successful_kol_matching",
			setupMock: func(mockClient *MockFastAPIClient) {
				mockClient.On("MatchKOLs", mock.Anything, mock.AnythingOfType("*fastapi.MatchKOLsRequest")).
					Return(sampleMatchKOLsResponse, nil)
			},
			request:        validKOLMatchingRequest,
			expectedStatus: http.StatusOK,
			validateResponse: func(t *testing.T, response map[string]interface{}) {
				assert.True(t, response["success"].(bool))
				assert.Contains(t, response, "data")
				
				data := response["data"].(map[string]interface{})
				matches := data["matches"].([]interface{})
				assert.Len(t, matches, 2)
				
				firstMatch := matches[0].(map[string]interface{})
				kol := firstMatch["kol"].(map[string]interface{})
				assert.Equal(t, "kol-123", kol["id"])
				assert.Equal(t, "techinfluencer1", kol["username"])
				assert.Equal(t, 8.7, firstMatch["score"])
			},
		},
		{
			name: "invalid_request_body",
			setupMock: func(mockClient *MockFastAPIClient) {
				// No mock setup needed for invalid request
			},
			request:        "invalid-json",
			expectedStatus: http.StatusBadRequest,
			expectedError:  "Invalid request body",
		},
		{
			name: "validation_failed_empty_brief",
			setupMock: func(mockClient *MockFastAPIClient) {
				// No mock setup needed for validation failure
			},
			request: models.KOLMatchingRequest{
				CampaignBrief: "",
				Budget:        5000.0,
				MaxResults:    20,
			},
			expectedStatus: http.StatusBadRequest,
			expectedError:  "campaign brief is required",
		},
		{
			name: "validation_failed_low_budget",
			setupMock: func(mockClient *MockFastAPIClient) {
				// No mock setup needed for validation failure
			},
			request: models.KOLMatchingRequest{
				CampaignBrief: "Test campaign",
				Budget:        50.0, // Below minimum
				MaxResults:    20,
			},
			expectedStatus: http.StatusBadRequest,
			expectedError:  "budget must be at least $100",
		},
		{
			name: "fastapi_client_error",
			setupMock: func(mockClient *MockFastAPIClient) {
				mockClient.On("MatchKOLs", mock.Anything, mock.AnythingOfType("*fastapi.MatchKOLsRequest")).
					Return(nil, assert.AnError)
			},
			request:        validKOLMatchingRequest,
			expectedStatus: http.StatusInternalServerError,
			expectedError:  "KOL matching failed",
		},
		{
			name: "no_fastapi_client",
			setupMock: func(mockClient *MockFastAPIClient) {
				// Handler will have no FastAPI client
			},
			request:        validKOLMatchingRequest,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  "FastAPI client not available",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// AIDEV-NOTE: Setup test environment
			gin.SetMode(gin.TestMode)
			log := logger.New("debug", "test")
			
			handler := &Handler{
				logger: log,
			}

			// AIDEV-NOTE: Setup mock client if test requires it
			var mockClient *MockFastAPIClient
			if tt.name != "no_fastapi_client" {
				mockClient = &MockFastAPIClient{}
				tt.setupMock(mockClient)
				handler.fastapiClient = mockClient
			}

			// AIDEV-NOTE: Create test request
			var reqBody []byte
			var err error
			if str, ok := tt.request.(string); ok {
				reqBody = []byte(str)
			} else {
				reqBody, err = json.Marshal(tt.request)
				require.NoError(t, err)
			}

			// AIDEV-NOTE: Create test HTTP request
			req := httptest.NewRequest("POST", "/api/v1/integration/match-kols", bytes.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()

			// AIDEV-NOTE: Create Gin context and call handler
			c, _ := gin.CreateTestContext(rec)
			c.Request = req

			handler.MatchKOLs(c)

			// AIDEV-NOTE: Assert response status
			assert.Equal(t, tt.expectedStatus, rec.Code)

			// AIDEV-NOTE: Parse and validate response
			var response map[string]interface{}
			err = json.Unmarshal(rec.Body.Bytes(), &response)
			require.NoError(t, err)

			if tt.expectedError != "" {
				assert.False(t, response["success"].(bool))
				assert.Contains(t, response["error"].(string), tt.expectedError)
			} else if tt.validateResponse != nil {
				tt.validateResponse(t, response)
			}

			// AIDEV-NOTE: Assert mock expectations
			if mockClient != nil {
				mockClient.AssertExpectations(t)
			}
		})
	}
}

// AIDEV-NOTE: Test suite for OptimizeBudget endpoint
func TestOptimizeBudget(t *testing.T) {
	tests := []struct {
		name           string
		setupMock      func(*MockFastAPIClient)
		request        interface{}
		expectedStatus int
		expectedError  string
		validateResponse func(*testing.T, map[string]interface{})
	}{
		{
			name: "successful_budget_optimization",
			setupMock: func(mockClient *MockFastAPIClient) {
				mockClient.On("OptimizeBudget", mock.Anything, mock.AnythingOfType("*fastapi.OptimizeBudgetRequest")).
					Return(sampleOptimizeBudgetResponse, nil)
			},
			request:        validBudgetOptimizationRequest,
			expectedStatus: http.StatusOK,
			validateResponse: func(t *testing.T, response map[string]interface{}) {
				assert.True(t, response["success"].(bool))
				assert.Contains(t, response, "data")
				
				data := response["data"].(map[string]interface{})
				allocations := data["allocation"].([]interface{})
				assert.Len(t, allocations, 2)
				
				summary := data["summary"].(map[string]interface{})
				assert.Equal(t, 7000.0, summary["total_allocated"])
				assert.Equal(t, 3000.0, summary["remaining_budget"])
			},
		},
		{
			name: "validation_failed_no_goals",
			setupMock: func(mockClient *MockFastAPIClient) {
				// No mock setup needed for validation failure
			},
			request: models.BudgetOptimizationRequest{
				TotalBudget:   5000.0,
				CampaignGoals: []string{}, // Empty goals
				KOLCandidates: []string{"kol-123"},
			},
			expectedStatus: http.StatusBadRequest,
			expectedError:  "at least one campaign goal is required",
		},
		{
			name: "validation_failed_no_candidates",
			setupMock: func(mockClient *MockFastAPIClient) {
				// No mock setup needed for validation failure
			},
			request: models.BudgetOptimizationRequest{
				TotalBudget:   5000.0,
				CampaignGoals: []string{"brand_awareness"},
				KOLCandidates: []string{}, // Empty candidates
			},
			expectedStatus: http.StatusBadRequest,
			expectedError:  "at least one KOL candidate is required",
		},
		{
			name: "validation_failed_low_budget",
			setupMock: func(mockClient *MockFastAPIClient) {
				// No mock setup needed for validation failure
			},
			request: models.BudgetOptimizationRequest{
				TotalBudget:   50.0, // Below minimum
				CampaignGoals: []string{"brand_awareness"},
				KOLCandidates: []string{"kol-123"},
			},
			expectedStatus: http.StatusBadRequest,
			expectedError:  "total budget must be at least $100",
		},
		{
			name: "fastapi_client_error",
			setupMock: func(mockClient *MockFastAPIClient) {
				mockClient.On("OptimizeBudget", mock.Anything, mock.AnythingOfType("*fastapi.OptimizeBudgetRequest")).
					Return(nil, assert.AnError)
			},
			request:        validBudgetOptimizationRequest,
			expectedStatus: http.StatusInternalServerError,
			expectedError:  "Budget optimization failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// AIDEV-NOTE: Setup test environment
			gin.SetMode(gin.TestMode)
			log := logger.New("debug", "test")
			
			handler := &Handler{
				logger: log,
			}

			// AIDEV-NOTE: Setup mock client
			mockClient := &MockFastAPIClient{}
			tt.setupMock(mockClient)
			handler.fastapiClient = mockClient

			// AIDEV-NOTE: Create test request
			reqBody, err := json.Marshal(tt.request)
			require.NoError(t, err)

			// AIDEV-NOTE: Create test HTTP request
			req := httptest.NewRequest("POST", "/api/v1/integration/optimize-budget", bytes.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()

			// AIDEV-NOTE: Create Gin context and call handler
			c, _ := gin.CreateTestContext(rec)
			c.Request = req

			handler.OptimizeBudget(c)

			// AIDEV-NOTE: Assert response status
			assert.Equal(t, tt.expectedStatus, rec.Code)

			// AIDEV-NOTE: Parse and validate response
			var response map[string]interface{}
			err = json.Unmarshal(rec.Body.Bytes(), &response)
			require.NoError(t, err)

			if tt.expectedError != "" {
				assert.False(t, response["success"].(bool))
				assert.Contains(t, response["error"].(string), tt.expectedError)
			} else if tt.validateResponse != nil {
				tt.validateResponse(t, response)
			}

			// AIDEV-NOTE: Assert mock expectations
			mockClient.AssertExpectations(t)
		})
	}
}

// AIDEV-NOTE: Benchmark tests for performance validation
func BenchmarkMatchKOLs(b *testing.B) {
	gin.SetMode(gin.TestMode)
	log := logger.New("error", "test") // Reduce logging for benchmark
	
	handler := &Handler{
		logger: log,
	}

	mockClient := &MockFastAPIClient{}
	mockClient.On("MatchKOLs", mock.Anything, mock.AnythingOfType("*fastapi.MatchKOLsRequest")).
		Return(sampleMatchKOLsResponse, nil)
	handler.fastapiClient = mockClient

	reqBody, _ := json.Marshal(validKOLMatchingRequest)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("POST", "/api/v1/integration/match-kols", bytes.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		c, _ := gin.CreateTestContext(rec)
		c.Request = req

		handler.MatchKOLs(c)

		if rec.Code != http.StatusOK {
			b.Fatalf("Expected status 200, got %d", rec.Code)
		}
	}
}

func BenchmarkOptimizeBudget(b *testing.B) {
	gin.SetMode(gin.TestMode)
	log := logger.New("error", "test") // Reduce logging for benchmark
	
	handler := &Handler{
		logger: log,
	}

	mockClient := &MockFastAPIClient{}
	mockClient.On("OptimizeBudget", mock.Anything, mock.AnythingOfType("*fastapi.OptimizeBudgetRequest")).
		Return(sampleOptimizeBudgetResponse, nil)
	handler.fastapiClient = mockClient

	reqBody, _ := json.Marshal(validBudgetOptimizationRequest)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("POST", "/api/v1/integration/optimize-budget", bytes.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		c, _ := gin.CreateTestContext(rec)
		c.Request = req

		handler.OptimizeBudget(c)

		if rec.Code != http.StatusOK {
			b.Fatalf("Expected status 200, got %d", rec.Code)
		}
	}
}

// AIDEV-NOTE: Helper tests for utility functions
func TestValidation(t *testing.T) {
	t.Run("kol_matching_validation", func(t *testing.T) {
		tests := []struct {
			name    string
			request models.KOLMatchingRequest
			wantErr bool
			errMsg  string
		}{
			{
				name:    "valid_request",
				request: validKOLMatchingRequest,
				wantErr: false,
			},
			{
				name: "brief_too_short",
				request: models.KOLMatchingRequest{
					CampaignBrief: "short",
					Budget:        1000.0,
				},
				wantErr: true,
				errMsg:  "campaign brief must be at least 10 characters",
			},
			{
				name: "invalid_follower_range",
				request: models.KOLMatchingRequest{
					CampaignBrief: "Valid campaign brief for testing",
					Budget:        1000.0,
					MinFollowers:  100000,
					MaxFollowers:  50000, // Less than min
				},
				wantErr: true,
				errMsg:  "min_followers must be less than max_followers",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				err := tt.request.Validate()
				if tt.wantErr {
					assert.Error(t, err)
					assert.Contains(t, err.Error(), tt.errMsg)
				} else {
					assert.NoError(t, err)
				}
			})
		}
	})

	t.Run("budget_optimization_validation", func(t *testing.T) {
		tests := []struct {
			name    string
			request models.BudgetOptimizationRequest
			wantErr bool
			errMsg  string
		}{
			{
				name:    "valid_request",
				request: validBudgetOptimizationRequest,
				wantErr: false,
			},
			{
				name: "invalid_constraint_range",
				request: models.BudgetOptimizationRequest{
					TotalBudget:   5000.0,
					CampaignGoals: []string{"brand_awareness"},
					KOLCandidates: []string{"kol-123"},
					Constraints: models.BudgetConstraints{
						MinKOLs: 5,
						MaxKOLs: 3, // Less than min
					},
				},
				wantErr: true,
				errMsg:  "min_kols must be less than max_kols",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				err := tt.request.Validate()
				if tt.wantErr {
					assert.Error(t, err)
					assert.Contains(t, err.Error(), tt.errMsg)
				} else {
					assert.NoError(t, err)
				}
			})
		}
	})
}

// AIDEV-NOTE: Integration test with realistic data flow
func TestEndToEndIntegration(t *testing.T) {
	// AIDEV-NOTE: This test would require a real database and FastAPI instance
	// For now, it serves as a template for future integration testing
	t.Skip("Requires live database and FastAPI instance")
	
	// AIDEV-NOTE: Would test full flow:
	// 1. Store KOLs in database
	// 2. Call MatchKOLs endpoint
	// 3. Use matched KOLs for budget optimization
	// 4. Verify audit trail in integration_requests table
}

// AIDEV-NOTE: Performance and load testing
func TestConcurrentRequests(t *testing.T) {
	gin.SetMode(gin.TestMode)
	log := logger.New("error", "test")
	
	handler := &Handler{
		logger: log,
	}

	mockClient := &MockFastAPIClient{}
	// AIDEV-NOTE: Allow multiple concurrent calls
	mockClient.On("MatchKOLs", mock.Anything, mock.AnythingOfType("*fastapi.MatchKOLsRequest")).
		Return(sampleMatchKOLsResponse, nil).Times(10)
	handler.fastapiClient = mockClient

	reqBody, _ := json.Marshal(validKOLMatchingRequest)

	// AIDEV-NOTE: Test concurrent requests
	concurrency := 10
	done := make(chan bool, concurrency)

	for i := 0; i < concurrency; i++ {
		go func() {
			defer func() { done <- true }()

			req := httptest.NewRequest("POST", "/api/v1/integration/match-kols", bytes.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()

			c, _ := gin.CreateTestContext(rec)
			c.Request = req

			handler.MatchKOLs(c)
			assert.Equal(t, http.StatusOK, rec.Code)
		}()
	}

	// AIDEV-NOTE: Wait for all goroutines to complete
	for i := 0; i < concurrency; i++ {
		select {
		case <-done:
			// Success
		case <-time.After(5 * time.Second):
			t.Fatal("Timeout waiting for concurrent requests")
		}
	}

	mockClient.AssertExpectations(t)
}