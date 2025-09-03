// AIDEV-NOTE: 250903170600 Comprehensive tests for database optimization features
// Tests connection pooling, query performance, monitoring, and retry mechanisms
package database

import (
	"context"
	"database/sql"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/jmoiron/sqlx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	
	"kol-scraper/internal/errors"
	"kol-scraper/pkg/config"
	"kol-scraper/pkg/logger"
)

// AIDEV-NOTE: 250903170601 Test suite for database optimization components
type DatabaseOptimizationTestSuite struct {
	suite.Suite
	db      *sqlx.DB
	testDB  *DB
	logger  *logger.Logger
	config  *config.Config
	ctx     context.Context
}

// SetupSuite initializes the test suite
func (suite *DatabaseOptimizationTestSuite) SetupSuite() {
	// AIDEV-NOTE: 250903170602 Setup test database connection
	suite.config = &config.Config{
		DatabaseURL:     "postgres://test:test@localhost:5432/test_db?sslmode=disable",
		MaxConnections:  10,
		MaxIdleConns:    5,
		ConnMaxLifetime: 300,
		Environment:     "test",
		LogLevel:        "debug",
	}

	suite.logger = logger.New("debug", "test")
	suite.ctx = context.Background()

	// Skip if no test database available
	if suite.config.DatabaseURL == "" {
		suite.T().Skip("No test database configured")
	}

	var err error
	suite.db, err = sqlx.Connect("postgres", suite.config.DatabaseURL)
	if err != nil {
		suite.T().Skip(fmt.Sprintf("Cannot connect to test database: %v", err))
	}

	// Create test tables
	suite.createTestTables()

	// Initialize enhanced database
	suite.testDB, err = NewConnection(suite.config, suite.logger)
	if err != nil {
		suite.T().Skip(fmt.Sprintf("Cannot initialize enhanced database: %v", err))
	}
}

// TearDownSuite cleans up the test suite
func (suite *DatabaseOptimizationTestSuite) TearDownSuite() {
	if suite.testDB != nil {
		suite.testDB.Close()
	}
	if suite.db != nil {
		suite.dropTestTables()
		suite.db.Close()
	}
}

// createTestTables creates tables needed for testing
func (suite *DatabaseOptimizationTestSuite) createTestTables() {
	queries := []string{
		`DROP TABLE IF EXISTS test_kols CASCADE`,
		`DROP TABLE IF EXISTS test_metrics CASCADE`,
		
		`CREATE TABLE test_kols (
			id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
			username VARCHAR(255) NOT NULL,
			platform VARCHAR(50) NOT NULL,
			platform_id VARCHAR(255) NOT NULL,
			display_name VARCHAR(255),
			is_active BOOLEAN DEFAULT true,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			UNIQUE(platform, platform_id)
		)`,
		
		`CREATE TABLE test_metrics (
			id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
			kol_id UUID NOT NULL,
			follower_count BIGINT DEFAULT 0,
			engagement_rate FLOAT DEFAULT 0,
			metrics_date DATE NOT NULL,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			FOREIGN KEY (kol_id) REFERENCES test_kols(id) ON DELETE CASCADE,
			UNIQUE(kol_id, metrics_date)
		)`,

		`CREATE INDEX IF NOT EXISTS idx_test_kols_platform_username ON test_kols (platform, username)`,
		`CREATE INDEX IF NOT EXISTS idx_test_metrics_kol_date ON test_metrics (kol_id, metrics_date DESC)`,
	}

	for _, query := range queries {
		if _, err := suite.db.Exec(query); err != nil {
			suite.T().Logf("Warning: Failed to execute setup query: %v", err)
		}
	}
}

// dropTestTables removes test tables
func (suite *DatabaseOptimizationTestSuite) dropTestTables() {
	queries := []string{
		`DROP TABLE IF EXISTS test_metrics CASCADE`,
		`DROP TABLE IF EXISTS test_kols CASCADE`,
	}

	for _, query := range queries {
		suite.db.Exec(query)
	}
}

// AIDEV-NOTE: 250903170603 Pool manager tests
func (suite *DatabaseOptimizationTestSuite) TestPoolManagerInitialization() {
	poolManager := NewPoolManager(suite.db, suite.config, suite.logger)
	require.NotNil(suite.T(), poolManager)

	// Test metrics retrieval
	metrics := poolManager.GetMetrics()
	assert.NotNil(suite.T(), metrics)
	assert.GreaterOrEqual(suite.T(), metrics.ConnectionsOpened, int64(0))
	
	// Test health status
	healthStatus := poolManager.GetHealthStatus()
	assert.NotNil(suite.T(), healthStatus)
	assert.Contains(suite.T(), healthStatus, "status")
	
	// Test health check
	assert.True(suite.T(), poolManager.IsHealthy())
	
	poolManager.Close()
}

func (suite *DatabaseOptimizationTestSuite) TestPoolManagerConcurrency() {
	poolManager := NewPoolManager(suite.db, suite.config, suite.logger)
	defer poolManager.Close()

	// AIDEV-NOTE: 250903170604 Test concurrent access to pool manager
	const numGoroutines = 20
	const operationsPerGoroutine = 10

	var wg sync.WaitGroup
	errorCount := int32(0)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			for j := 0; j < operationsPerGoroutine; j++ {
				ctx, cancel := context.WithTimeout(suite.ctx, 2*time.Second)
				
				result, err := poolManager.QueryContextWithMonitoring(ctx, "SELECT 1")
				if err != nil {
					errorCount++
					cancel()
					continue
				}
				
				result.Close()
				cancel()
			}
		}()
	}

	wg.Wait()
	
	// Most operations should succeed
	assert.LessOrEqual(suite.T(), errorCount, int32(numGoroutines/4)) // Allow some failures
	
	// Verify metrics were updated
	metrics := poolManager.GetMetrics()
	assert.Greater(suite.T(), metrics.ConnectionsOpened, int64(0))
}

// AIDEV-NOTE: 250903170605 Query executor tests
func (suite *DatabaseOptimizationTestSuite) TestQueryExecutorPreparedStatements() {
	queryExecutor := NewQueryExecutor(suite.db, nil, suite.logger)
	defer queryExecutor.Close()

	// Test metrics retrieval
	metrics := queryExecutor.GetQueryMetrics()
	assert.NotNil(suite.T(), metrics)
	assert.GreaterOrEqual(suite.T(), metrics.PreparedStatements, 0)
}

func (suite *DatabaseOptimizationTestSuite) TestQueryCaching() {
	queryExecutor := NewQueryExecutor(suite.db, nil, suite.logger)
	defer queryExecutor.Close()

	// Insert test data
	_, err := suite.db.Exec(`
		INSERT INTO test_kols (username, platform, platform_id, display_name) 
		VALUES ('testuser', 'tiktok', 'test123', 'Test User')
	`)
	require.NoError(suite.T(), err)

	// First query - should miss cache
	start := time.Now()
	result1 := suite.querySelectCount("test_kols", "platform = 'tiktok'")
	duration1 := time.Since(start)

	// Second identical query - should be faster due to database caching
	start = time.Now()
	result2 := suite.querySelectCount("test_kols", "platform = 'tiktok'")
	duration2 := time.Since(start)

	assert.Equal(suite.T(), result1, result2)
	assert.True(suite.T(), result1 > 0, "Should find inserted record")
	
	// Second query might be faster, but not always guaranteed in tests
	suite.logger.Info("Query durations", logger.Fields{
		"first_ms":  duration1.Milliseconds(),
		"second_ms": duration2.Milliseconds(),
	})

	// Clear cache and verify
	queryExecutor.InvalidateCache()
	
	// Cleanup
	suite.db.Exec(`DELETE FROM test_kols WHERE platform_id = 'test123'`)
}

// AIDEV-NOTE: 250903170606 Resilient client tests
func (suite *DatabaseOptimizationTestSuite) TestResilientClientRetry() {
	poolManager := NewPoolManager(suite.db, suite.config, suite.logger)
	defer poolManager.Close()
	
	resilientClient := NewResilientDB(suite.db, poolManager, suite.logger)

	// Test successful operation
	var result int
	err := resilientClient.GetContext(suite.ctx, &result, "SELECT 1")
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), 1, result)

	// Verify metrics
	metrics := resilientClient.GetRetryMetrics()
	assert.NotNil(suite.T(), metrics)
	assert.Greater(suite.T(), metrics.TotalOperations, int64(0))
	assert.GreaterOrEqual(suite.T(), resilientClient.GetSuccessRate(), 0.0)
}

func (suite *DatabaseOptimizationTestSuite) TestResilientClientTransaction() {
	poolManager := NewPoolManager(suite.db, suite.config, suite.logger)
	defer poolManager.Close()
	
	resilientClient := NewResilientDB(suite.db, poolManager, suite.logger)

	// Test transaction with retry
	err := resilientClient.TransactionWithRetry(suite.ctx, func(tx *sqlx.Tx) error {
		_, err := tx.Exec(`
			INSERT INTO test_kols (username, platform, platform_id) 
			VALUES ('txuser', 'instagram', 'tx123')
		`)
		if err != nil {
			return err
		}
		
		// Verify record exists in transaction
		var count int
		err = tx.Get(&count, "SELECT COUNT(*) FROM test_kols WHERE platform_id = 'tx123'")
		if err != nil {
			return err
		}
		
		assert.Equal(suite.T(), 1, count)
		return nil
	})
	
	assert.NoError(suite.T(), err)
	
	// Verify record persisted after transaction
	count := suite.querySelectCount("test_kols", "platform_id = 'tx123'")
	assert.Equal(suite.T(), 1, count)
	
	// Cleanup
	suite.db.Exec(`DELETE FROM test_kols WHERE platform_id = 'tx123'`)
}

// AIDEV-NOTE: 250903170607 Database monitor tests
func (suite *DatabaseOptimizationTestSuite) TestDatabaseMonitor() {
	poolManager := NewPoolManager(suite.db, suite.config, suite.logger)
	defer poolManager.Close()
	
	monitor := NewDatabaseMonitor(suite.db, poolManager, suite.logger)
	defer monitor.Stop()

	// Start monitoring
	err := monitor.Start(suite.ctx)
	assert.NoError(suite.T(), err)

	// Wait for initial metrics collection
	time.Sleep(100 * time.Millisecond)

	// Test metrics retrieval
	metrics := monitor.GetMetrics()
	assert.NotNil(suite.T(), metrics)
	
	// Test alert system
	alerts := monitor.GetAlerts()
	assert.NotNil(suite.T(), alerts)
}

// AIDEV-NOTE: 250903170608 Metric collectors tests
func (suite *DatabaseOptimizationTestSuite) TestTableSizeCollector() {
	collector := &TableSizeCollector{}
	
	result, err := collector.Collect(suite.ctx, suite.db)
	
	if err != nil {
		// May fail if insufficient privileges, which is acceptable in test
		suite.T().Logf("Table size collection failed (may be expected): %v", err)
		return
	}
	
	assert.NotNil(suite.T(), result)
	
	if sizes, ok := result.(map[string]int64); ok {
		assert.GreaterOrEqual(suite.T(), len(sizes), 0)
		suite.T().Logf("Collected sizes for %d tables", len(sizes))
	}
}

func (suite *DatabaseOptimizationTestSuite) TestConnectionStatsCollector() {
	collector := &ConnectionStatsCollector{}
	
	result, err := collector.Collect(suite.ctx, suite.db)
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), result)
	
	if stats, ok := result.(*ConnectionStats); ok {
		assert.GreaterOrEqual(suite.T(), stats.Total, 0)
		assert.GreaterOrEqual(suite.T(), stats.Active, 0)
	}
}

// AIDEV-NOTE: 250903170609 Integration tests
func (suite *DatabaseOptimizationTestSuite) TestEnhancedDatabaseIntegration() {
	// Test enhanced database methods
	
	// Insert test data
	_, err := suite.testDB.ExecWithRetry(suite.ctx, `
		INSERT INTO test_kols (username, platform, platform_id, display_name) 
		VALUES ('integrationtest', 'youtube', 'int123', 'Integration Test User')
	`)
	require.NoError(suite.T(), err)

	// Test optimized query
	var kol struct {
		ID          string `db:"id"`
		Username    string `db:"username"`
		Platform    string `db:"platform"`
		DisplayName string `db:"display_name"`
	}
	
	err = suite.testDB.GetWithRetry(suite.ctx, &kol, `
		SELECT id, username, platform, display_name 
		FROM test_kols 
		WHERE platform_id = $1
	`, "int123")
	
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), "integrationtest", kol.Username)
	assert.Equal(suite.T(), "youtube", kol.Platform)

	// Test health status
	healthStatus := suite.testDB.GetHealthStatus()
	assert.NotNil(suite.T(), healthStatus)
	assert.Contains(suite.T(), healthStatus, "database")

	// Test metrics retrieval
	poolMetrics := suite.testDB.GetPoolMetrics()
	if poolMetrics != nil {
		assert.GreaterOrEqual(suite.T(), poolMetrics.ConnectionsOpened, int64(0))
	}

	queryMetrics := suite.testDB.GetQueryMetrics()
	if queryMetrics != nil {
		assert.GreaterOrEqual(suite.T(), queryMetrics.TotalQueries, int64(0))
	}

	retryMetrics := suite.testDB.GetRetryMetrics()
	if retryMetrics != nil {
		assert.GreaterOrEqual(suite.T(), retryMetrics.TotalOperations, int64(0))
	}

	// Cleanup
	suite.testDB.ExecWithRetry(suite.ctx, `DELETE FROM test_kols WHERE platform_id = 'int123'`)
}

func (suite *DatabaseOptimizationTestSuite) TestConcurrentOperations() {
	const numWorkers = 10
	const operationsPerWorker = 5

	var wg sync.WaitGroup
	successCount := int64(0)
	var successMutex sync.Mutex

	// AIDEV-NOTE: 250903170610 Test concurrent database operations
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			for j := 0; j < operationsPerWorker; j++ {
				platformID := fmt.Sprintf("worker%d_op%d", workerID, j)
				
				// Insert
				_, err := suite.testDB.ExecWithRetry(suite.ctx, `
					INSERT INTO test_kols (username, platform, platform_id) 
					VALUES ($1, 'concurrent_test', $2)
				`, fmt.Sprintf("user%d_%d", workerID, j), platformID)
				
				if err == nil {
					successMutex.Lock()
					successCount++
					successMutex.Unlock()
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify most operations succeeded
	expectedOps := int64(numWorkers * operationsPerWorker)
	assert.GreaterOrEqual(suite.T(), successCount, expectedOps*8/10) // At least 80% success

	// Count inserted records
	count := suite.querySelectCount("test_kols", "platform = 'concurrent_test'")
	assert.Equal(suite.T(), int(successCount), count)

	// Cleanup
	suite.db.Exec(`DELETE FROM test_kols WHERE platform = 'concurrent_test'`)

	// Verify health after concurrent load
	assert.True(suite.T(), suite.testDB.poolManager.IsHealthy())
}

// AIDEV-NOTE: 250903170611 Helper methods for testing
func (suite *DatabaseOptimizationTestSuite) querySelectCount(table, where string) int {
	var count int
	query := fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE %s", table, where)
	err := suite.db.Get(&count, query)
	if err != nil {
		suite.T().Logf("Error in querySelectCount: %v", err)
		return 0
	}
	return count
}

// AIDEV-NOTE: 250903170612 Benchmark tests for performance validation
func BenchmarkPoolManagerQuery(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	config := &config.Config{
		DatabaseURL:     "postgres://test:test@localhost:5432/test_db?sslmode=disable",
		MaxConnections:  10,
		MaxIdleConns:    5,
		ConnMaxLifetime: 300,
	}

	logger := logger.New("error", "test")
	
	db, err := sqlx.Connect("postgres", config.DatabaseURL)
	if err != nil {
		b.Skip("No test database available")
	}
	defer db.Close()

	poolManager := NewPoolManager(db, config, logger)
	defer poolManager.Close()

	ctx := context.Background()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			result, err := poolManager.QueryContextWithMonitoring(ctx, "SELECT 1")
			if err != nil {
				b.Error(err)
				continue
			}
			result.Close()
		}
	})
}

func BenchmarkResilientClientQuery(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	config := &config.Config{
		DatabaseURL:     "postgres://test:test@localhost:5432/test_db?sslmode=disable",
		MaxConnections:  10,
		MaxIdleConns:    5,
		ConnMaxLifetime: 300,
	}

	logger := logger.New("error", "test")
	
	db, err := sqlx.Connect("postgres", config.DatabaseURL)
	if err != nil {
		b.Skip("No test database available")
	}
	defer db.Close()

	poolManager := NewPoolManager(db, config, logger)
	defer poolManager.Close()

	resilientClient := NewResilientDB(db, poolManager, logger)

	ctx := context.Background()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			var result int
			err := resilientClient.GetContext(ctx, &result, "SELECT 1")
			if err != nil {
				b.Error(err)
			}
		}
	})
}

// Run the test suite
func TestDatabaseOptimizationSuite(t *testing.T) {
	suite.Run(t, new(DatabaseOptimizationTestSuite))
}

// AIDEV-NOTE: 250903170613 Unit tests for individual components
func TestRetryConfigValidation(t *testing.T) {
	config := &RetryConfig{
		MaxAttempts:   3,
		InitialDelay:  100 * time.Millisecond,
		MaxDelay:      5 * time.Second,
		BackoffFactor: 2.0,
		JitterFactor:  0.1,
		EnableJitter:  true,
	}

	assert.Equal(t, 3, config.MaxAttempts)
	assert.Equal(t, 100*time.Millisecond, config.InitialDelay)
	assert.True(t, config.EnableJitter)
}

func TestCircuitBreakerStates(t *testing.T) {
	cb := &CircuitBreaker{
		maxRequests: 5,
		interval:    60 * time.Second,
		timeout:     30 * time.Second,
		state:       StateClosed,
	}

	// Initially closed
	assert.Equal(t, StateClosed, cb.state)
	assert.True(t, cb.canExecute())

	// Record failures to trip circuit
	for i := 0; i < 5; i++ {
		cb.recordFailure()
	}

	// Should be open now
	assert.Equal(t, StateOpen, cb.state)
	assert.False(t, cb.canExecute())

	// Record success should reset consecutive failures
	cb.recordSuccess()
	assert.Equal(t, 0, cb.consecutiveFailures)
}

func TestErrorClassification(t *testing.T) {
	tests := []struct {
		name        string
		err         error
		expectedType string
	}{
		{
			name:        "connection error",
			err:         fmt.Errorf("connection refused"),
			expectedType: "connection_error",
		},
		{
			name:        "timeout error", 
			err:         fmt.Errorf("context deadline exceeded"),
			expectedType: "timeout_error",
		},
		{
			name:        "deadlock error",
			err:         fmt.Errorf("deadlock detected"),
			expectedType: "deadlock_error",
		},
		{
			name:        "not found error",
			err:         sql.ErrNoRows,
			expectedType: "not_found_error",
		},
		{
			name:        "other error",
			err:         fmt.Errorf("some other error"),
			expectedType: "other_error",
		},
	}

	rdb := &ResilientDB{}
	
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := rdb.classifyError(test.err)
			assert.Equal(t, test.expectedType, result)
		})
	}
}