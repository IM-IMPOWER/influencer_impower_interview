# KOL Platform Algorithm Testing Guide

## Overview

This document provides comprehensive guidance for testing the KOL platform's AI-powered matching and scoring algorithms, budget optimization systems, and constraint filtering mechanisms.

## Test Architecture

### Test Pyramid Structure

```
                    E2E Tests (Integration)
                   /                      \
              GraphQL Tests            Workflow Tests  
             /                                        \
        Unit Tests                               Performance Tests
       /         \                               /                 \
  Scoring    Optimization              Scalability           Benchmarks
  Tests      Tests                     Tests                 Tests
```

### Test Categories

1. **Unit Tests** - Individual algorithm components
2. **Integration Tests** - Service interactions and workflows  
3. **Performance Tests** - Scalability and benchmarks
4. **Mathematical Validation** - Algorithm correctness
5. **Edge Case Tests** - Boundary conditions and error handling

## Test Files Structure

```
apps/api/tests/
├── conftest.py                                    # Test configuration and fixtures
├── fixtures/
│   ├── test_data_factory.py                     # Test data generation
│   └── sample_data/                             # Static test data
├── test_scoring_algorithms_validation.py        # Mathematical validation tests
├── test_constraint_filtering_system.py          # Constraint filtering tests
├── test_budget_optimization_algorithms.py       # Optimization algorithm tests
├── test_integration_scoring_workflows.py        # End-to-end workflow tests
├── test_performance_benchmarks.py               # Performance and scalability tests
├── test_kol_scorer.py                           # KOL scorer unit tests
├── test_enhanced_budget_optimizer.py            # Budget optimizer unit tests
├── test_graphql_integration.py                  # GraphQL resolver tests
└── test_missing_data.py                        # Missing data handling tests
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "scoring"              # Scoring algorithm tests
pytest -m "optimization"         # Optimization algorithm tests  
pytest -m "performance"          # Performance tests
pytest -m "integration"          # Integration tests
pytest -m "mathematical"         # Mathematical validation tests

# Run tests with coverage
pytest --cov=src/kol_api --cov-report=html

# Run performance benchmarks
pytest -m "performance and not slow"

# Run slow integration tests
pytest -m "slow"
```

### Test Filtering

```bash
# Run tests by pattern
pytest -k "kol_scorer"          # All KOL scorer tests
pytest -k "constraint"          # All constraint-related tests
pytest -k "genetic"            # Genetic algorithm tests

# Exclude certain tests  
pytest -k "not slow"           # Exclude slow tests
pytest -k "not performance"    # Exclude performance tests

# Run tests by file
pytest tests/test_scoring_algorithms_validation.py
pytest tests/test_budget_optimization_algorithms.py
```

### CI/CD Test Execution

```bash
# GitHub Actions workflow triggers
git push origin master          # Runs full test suite
git commit -m "feat: optimize [perf-test]"  # Triggers performance tests

# Manual workflow dispatch
gh workflow run test-algorithms.yml
```

## Test Data Generation

### Using Test Factories

```python
from tests.fixtures.test_data_factory import KOLDataFactory, CampaignDataFactory

# Generate realistic KOL profiles
kol, metrics = KOLDataFactory.create_kol_profile(
    tier="micro",
    quality_level="high", 
    data_completeness="complete"
)

# Generate campaign requirements
campaign = CampaignDataFactory.create_campaign_requirements(
    budget_size="medium",
    complexity="complex"
)

# Create optimization scenarios
scenarios = TestScenarioFactory.create_budget_optimization_scenarios()
```

### Data Quality Levels

- **High Quality**: Verified accounts, consistent engagement, complete data
- **Medium Quality**: Standard accounts, moderate engagement, partial data  
- **Low Quality**: Unverified accounts, inconsistent engagement, minimal data

### Data Completeness Levels

- **Complete**: All required fields populated with realistic data
- **Partial**: Some fields missing, simulates real-world data gaps
- **Minimal**: Only essential fields, tests robustness to missing data

## Algorithm Testing Guidelines

### Mathematical Validation Tests

These tests ensure algorithms maintain mathematical correctness:

```python
class TestScoringAlgorithmMathematicalProperties:
    def test_score_normalization_bounds(self):
        """Ensure all scores are bounded [0,1]."""
        
    def test_score_monotonicity_properties(self):
        """Better metrics should yield better scores."""
        
    def test_weighted_score_calculation_accuracy(self):
        """Weighted scores should be mathematically accurate."""
```

### Performance Requirements

| Algorithm | Small Dataset | Medium Dataset | Large Dataset |
|-----------|--------------|----------------|---------------|
| KOL Scoring | <100ms/KOL | <150ms/KOL | <200ms/KOL |
| Genetic Algorithm | <5s | <30s | <120s |
| Constraint Satisfaction | <2s | <10s | <45s |
| Linear Programming | <1s | <5s | <20s |

### Memory Usage Limits

- **Single KOL Scoring**: <1MB per KOL
- **Batch Processing**: <500MB total
- **Large Dataset**: <1GB peak usage
- **Memory Leaks**: <50MB increase over time

## Constraint Testing

### Hard Constraints (Must Never Violate)

```python
def test_hard_constraints():
    """Test constraints that must never be violated."""
    
    # Budget constraint
    assert total_cost <= max_budget
    
    # Count constraints  
    assert min_kols <= selected_count <= max_kols
    
    # Risk constraints
    assert all(kol.risk <= max_risk for kol in selected)
```

### Soft Constraints (Can Violate with Penalties)

```python  
def test_soft_constraints():
    """Test constraints that apply penalties when violated."""
    
    # Performance targets
    if total_reach < min_reach:
        assert "min_reach" in reported_violations
        
    # Diversity requirements
    if portfolio_diversity < target_diversity:
        assert "diversity" in reported_violations
```

## Performance Testing

### Scalability Testing

```python
@pytest.mark.performance
def test_scoring_scalability():
    """Test scoring performance with increasing dataset sizes."""
    
    datasets = {
        "small": 50,
        "medium": 200, 
        "large": 1000
    }
    
    for size_name, kol_count in datasets.items():
        # Generate dataset
        kols = generate_test_kols(kol_count)
        
        # Measure performance
        start_time = time.time()
        results = score_kols(kols)
        elapsed = time.time() - start_time
        
        # Validate performance
        avg_time = elapsed / kol_count
        assert avg_time < PERFORMANCE_THRESHOLD
```

### Memory Profiling

```python
@memory_profiler.profile
def test_memory_usage():
    """Profile memory usage during algorithm execution."""
    
    # Generate large dataset
    kols = generate_test_kols(1000)
    
    # Monitor memory during processing
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Process data
    results = process_kols(kols)
    
    # Check for memory leaks
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < MEMORY_THRESHOLD
```

## Integration Testing

### GraphQL Integration

```python
@pytest.mark.integration
async def test_graphql_scoring_integration():
    """Test KOL scoring through GraphQL resolvers."""
    
    query = """
        mutation ScoreKOLForCampaign($kolId: ID!, $campaignId: ID!) {
            scoreKol(kolId: $kolId, campaignId: $campaignId) {
                compositeScore
                scoreComponents {
                    roiScore
                    audienceQualityScore
                    brandSafetyScore
                }
                confidence
            }
        }
    """
    
    result = await graphql_client.execute(query, variables={
        "kolId": "test_kol_123",
        "campaignId": "test_campaign_456"
    })
    
    assert result["data"]["scoreKol"]["compositeScore"] > 0
    assert 0 <= result["data"]["scoreKol"]["confidence"] <= 1
```

### Database Integration

```python
@pytest.mark.requires_db
async def test_database_integration():
    """Test algorithm integration with database."""
    
    # Setup test data in database
    await create_test_kols(db_session, count=10)
    await create_test_campaign(db_session)
    
    # Run algorithm with database data
    optimizer = EnhancedBudgetOptimizerService(db_session)
    result = await optimizer.optimize_campaign_budget_advanced(
        campaign_id="test_campaign",
        constraints=test_constraints
    )
    
    # Validate database interactions
    assert result.selected_kols
    assert all(kol.kol_id in db_kol_ids for kol in result.selected_kols)
```

## Error Handling Testing

### Missing Data Scenarios

```python
@pytest.mark.missing_data
def test_missing_engagement_rate():
    """Test handling of missing engagement rate data."""
    
    kol = create_kol_with_missing_data(["engagement_rate"])
    
    # Should handle gracefully
    score = scorer.score_kol(kol, campaign)
    
    # Should return valid score with reduced confidence
    assert 0 <= score.composite_score <= 1
    assert score.overall_confidence < 0.8  # Reduced confidence
    assert "missing_engagement_rate" in score.warnings
```

### Error Recovery

```python  
@pytest.mark.error_handling
def test_external_service_failure_recovery():
    """Test recovery from external service failures."""
    
    # Mock external service failure
    with patch('external_service.call') as mock_service:
        mock_service.side_effect = TimeoutError("Service unavailable")
        
        # Should continue with fallback
        score = scorer.score_kol(kol, campaign)
        
        # Should use fallback scoring with warning
        assert score is not None
        assert "external_service_timeout" in score.warnings
        assert score.overall_confidence < 0.9
```

## Test Environment Setup

### Local Development

```bash
# Install test dependencies
pip install -e .
pip install pytest pytest-asyncio pytest-cov pytest-mock

# Set environment variables
export TESTING=true
export LOG_LEVEL=ERROR
export DATABASE_URL=sqlite:///test.db

# Run tests
pytest
```

### Docker Environment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run tests
CMD ["pytest", "--tb=short", "--durations=10"]
```

### GitHub Actions

The CI/CD pipeline automatically runs tests on:
- Push to master/develop branches
- Pull request creation
- Scheduled daily performance tests
- Manual workflow dispatch

## Coverage Requirements

### Coverage Targets by Module

| Module | Minimum Coverage | Target Coverage |
|--------|------------------|----------------|
| Scoring Algorithms | 85% | 90% |
| Budget Optimization | 80% | 85% |
| Constraint Filtering | 85% | 90% |
| GraphQL Resolvers | 75% | 80% |
| Integration Services | 70% | 75% |

### Coverage Exclusions

- Debug code (`def _debug_*`)
- Performance profiling (`@profile`)
- Error handling for external dependencies
- Platform-specific code
- Development utilities

## Debugging Test Failures

### Common Issues

1. **Timeout Failures**: Increase timeout in pytest.ini or mark as slow
2. **Memory Issues**: Check for data leaks, reduce dataset sizes
3. **Flaky Tests**: Add retry logic, improve test isolation
4. **Performance Regressions**: Check for algorithm changes, system load

### Debugging Commands

```bash
# Run single test with debug output
pytest tests/test_scoring.py::test_kol_scoring -v -s

# Run with pdb debugger
pytest --pdb tests/test_scoring.py::test_kol_scoring

# Profile memory usage
pytest --profile-svg tests/test_performance.py

# Run with coverage debug
pytest --cov-report=term-missing --cov-report=html
```

## Test Data Management

### Test Data Lifecycle

1. **Generation**: Create realistic test data using factories
2. **Isolation**: Each test gets independent data  
3. **Cleanup**: Automatic cleanup after test completion
4. **Archival**: Performance test results archived for trend analysis

### Data Privacy

- All test data is synthetic/generated
- No real KOL or campaign data in tests
- Sensitive fields are masked or randomized
- Test databases are separate from production

## Contributing to Tests

### Adding New Tests

1. Follow naming convention: `test_<functionality>_<scenario>`
2. Use appropriate test markers (`@pytest.mark.performance`)
3. Include docstrings describing test purpose
4. Add test data factories for new data types
5. Update this documentation for new test categories

### Test Review Checklist

- [ ] Tests are isolated and independent
- [ ] Performance requirements are met  
- [ ] Edge cases are covered
- [ ] Error handling is tested
- [ ] Documentation is updated
- [ ] CI/CD pipeline runs successfully

## Monitoring and Metrics

### Test Metrics Dashboard

- Test execution time trends
- Coverage percentage over time
- Flaky test detection
- Performance regression alerts
- Test failure patterns

### Performance Baselines

Performance test results are tracked over time to detect regressions:

- **Scoring Performance**: <100ms per KOL (99th percentile)
- **Optimization Speed**: <30s for 200 candidates
- **Memory Usage**: <1GB peak for large datasets
- **Success Rate**: >95% for all test categories

## Troubleshooting

### Common Test Environment Issues

**Database Connection Errors**:
```bash
export DATABASE_URL=sqlite:///test.db
pytest tests/test_database.py
```

**Redis Connection Errors**:  
```bash
redis-server --daemonize yes
export REDIS_URL=redis://localhost:6379
```

**Memory Issues**:
```bash
# Increase available memory for tests
export PYTEST_XDIST_WORKER_COUNT=2
pytest -n 2 tests/
```

**Performance Test Failures**:
```bash
# Run performance tests in isolation
pytest -m "performance" --tb=short --durations=30
```

For additional support, refer to the test logs and error messages, or contact the development team.