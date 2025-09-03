# Enhanced GraphQL Schema Guide

## Overview

This document describes the sophisticated GraphQL schema that supports the advanced POC2 (KOL Matching) and POC4 (Budget Optimizer) implementations. The schema provides comprehensive access to multi-factor scoring algorithms, vector similarity search, constraint-based optimization, and sophisticated data management.

## 🚀 Key Features Supported

### POC2: Sophisticated KOL Matching
- **Multi-factor scoring** with 6 core components (ROI, audience quality, brand safety, content relevance, demographic fit, reliability)
- **Confidence-based filtering** for missing data handling
- **Semantic similarity search** using vector embeddings
- **Performance predictions** with risk assessment
- **Advanced filtering** with comprehensive criteria

### POC4: Budget Optimization
- **Constraint-based optimization** with multiple objectives
- **Tier-based allocation strategies** 
- **Alternative scenario generation**
- **Risk analysis and sensitivity testing**
- **Performance forecasting** with confidence intervals

### Data Management Enhancements
- **Markdown brief parsing** with NLP extraction
- **Enhanced data refresh** with priority queuing
- **Comprehensive export capabilities**
- **Data quality reporting**
- **Brand safety management** with audit trails

## 📊 Core Type Enhancements

### Enhanced KOL Type

```graphql
type KOL {
  # Basic information
  id: String!
  username: String!
  displayName: String!
  platform: PlatformTypeEnum!
  tier: KOLTierEnum!
  primaryCategory: ContentCategoryEnum!
  
  # Status and quality indicators
  isVerified: Boolean!
  isBrandSafe: Boolean!
  dataCompleteness: DecimalType!
  hasSufficientData: Boolean!
  missingDataFields: [String!]!
  
  # Enhanced metrics with comprehensive data
  metrics: KOLMetrics
  
  # Sophisticated scoring (POC2)
  score: KOLScore
  
  # Brand safety details
  brandSafetyNotes: String
  brandSafetyLastUpdated: DateTimeType
}
```

### Advanced Scoring Components

```graphql
type ScoreComponents {
  # Core scoring factors (POC2 weights)
  roiScore: DecimalType!                    # 25%
  audienceQualityScore: DecimalType!        # 25%
  brandSafetyScore: DecimalType!            # 20%
  contentRelevanceScore: DecimalType!       # 15%
  demographicFitScore: DecimalType!         # 10%
  reliabilityScore: DecimalType!            # 5%
  
  # Confidence scores for missing data handling
  roiConfidence: DecimalType!
  audienceConfidence: DecimalType!
  brandSafetyConfidence: DecimalType!
  contentRelevanceConfidence: DecimalType!
  demographicConfidence: DecimalType!
  reliabilityConfidence: DecimalType!
  
  # Overall metrics
  overallConfidence: DecimalType!
  scoringTimestamp: DateTimeType!
  dataFreshnessDays: Int!
  sampleSize: Int
}
```

### Semantic Matching Data

```graphql
type SemanticMatchingData {
  similarityScore: DecimalType!
  contentMatchScore: DecimalType!
  brandAffinityScore: DecimalType!
  embeddingConfidence: DecimalType!
  matchedContentCategories: [String!]!
  semanticKeywords: [String!]!
}
```

### Performance Predictions

```graphql
type PerformancePrediction {
  predictedReach: Int!
  predictedEngagement: Int!
  predictedConversions: Int!
  predictedRoi: DecimalType!
  predictionConfidence: DecimalType!
  riskFactors: [String!]!
  historicalCampaignCount: Int!
}
```

## 🔍 Sophisticated Query Operations

### Enhanced KOL Matching

```graphql
query EnhancedKOLMatching(
  $campaignId: String!
  $confidenceThreshold: Float = 0.7
  $enableSemanticMatching: Boolean = true
) {
  matchKolsForCampaign(
    campaignId: $campaignId
    confidenceThreshold: $confidenceThreshold
    enableSemanticMatching: $enableSemanticMatching
  ) {
    matchedKols {
      id
      username
      score {
        scoreComponents {
          roiScore
          audienceQualityScore
          overallConfidence
        }
        semanticMatching {
          similarityScore
          contentMatchScore
        }
        performancePrediction {
          predictedReach
          predictedRoi
          riskFactors
        }
      }
    }
    # Rich metadata
    scoringMethod
    algorithmVersion
    processingTimeSeconds
    dataQualitySummary
    semanticMatchingEnabled
  }
}
```

### Direct Requirements Matching

```graphql
query MatchWithRequirements($requirements: CampaignRequirementsInput!) {
  matchKolsWithRequirements(requirements: $requirements) {
    matchedKols {
      id
      username
      score {
        overallScore
        scoreComponents {
          overallConfidence
        }
      }
    }
    confidenceScores
    processingTimeSeconds
  }
}
```

### Budget Optimization

```graphql
query OptimizeBudget(
  $campaignId: String!
  $optimizationObjective: String!
  $totalBudget: Float!
  $constraints: BudgetOptimizationConstraintsInput
) {
  optimizeBudget(
    campaignId: $campaignId
    optimizationObjective: $optimizationObjective
    totalBudget: $totalBudget
    constraints: $constraints
    generateAlternatives: true
    includeRiskAnalysis: true
  ) {
    optimizedPlan {
      id
      optimizationScore
      predictedRoi
      riskAssessment
      allocations {
        targetKolIds
        expectedRoi
        riskScore
        optimizationConfidence
      }
    }
    alternativePlans {
      id
      optimizationScore
      predictedRoi
    }
    riskAssessment
    sensitivityAnalysis
    processingTimeSeconds
  }
}
```

## 🔧 Advanced Input Types

### Campaign Requirements Input

```graphql
input CampaignRequirementsInput {
  # Target criteria
  targetKolTiers: [String!]!
  targetCategories: [String!]!
  totalBudget: Float!
  
  # Audience constraints
  minFollowerCount: Int
  maxFollowerCount: Int
  minEngagementRate: Float
  minAvgViews: Int
  
  # Demographic targeting
  targetDemographics: String  # JSON
  targetLocations: [String!]
  targetLanguages: [String!]
  targetAgeRanges: [String!]
  
  # Brand safety requirements
  requireBrandSafe: Boolean = true
  requireVerified: Boolean = false
  excludeControversial: Boolean = true
  
  # Content requirements
  requiredHashtags: [String!]
  excludedHashtags: [String!]
  contentSentimentRequirements: String
  
  # Campaign specific
  campaignObjective: String = "balanced"
  expectedConversionRate: Float
}
```

### Budget Optimization Constraints

```graphql
input BudgetOptimizationConstraintsInput {
  # Tier distribution requirements
  minNanoPercentage: Float
  minMicroPercentage: Float
  minMidPercentage: Float
  minMacroPercentage: Float
  minMegaPercentage: Float
  
  # KOL quantity constraints
  minKolsRequired: Int = 1
  maxKolsAllowed: Int
  
  # Budget allocation constraints
  maxSingleKolPercentage: Float
  minReservedBuffer: Float = 0.0
  
  # Performance requirements
  minTotalReach: Int
  minTotalEngagement: Int
  targetRoi: Float
  
  # Risk management
  maxRiskScore: Float
  diversificationRequirement: Boolean = false
  
  # Objective weights
  reachWeight: Float = 0.3
  engagementWeight: Float = 0.3
  costEfficiencyWeight: Float = 0.2
  riskMitigationWeight: Float = 0.2
}
```

## 🔄 Enhanced Mutations

### Parse Campaign Brief

```graphql
mutation ParseCampaignBrief($brief: BriefUploadInput!) {
  parseCampaignBrief(brief: $brief) {
    success
    campaignRequirements  # JSON serialized
    parsingConfidence
    ambiguousRequirements
    missingRequirements
    extractedEntities
    isActionable
    requiredManualInputs
  }
}
```

### Enhanced Data Refresh

```graphql
mutation EnhancedDataRefresh(
  $kolIds: [String!]
  $priority: String = "normal"
  $forceRefresh: Boolean = false
) {
  triggerKolDataRefresh(
    kolIds: $kolIds
    priority: $priority
    forceRefresh: $forceRefresh
  ) {
    success
    kolsRefreshed
    platformsUpdated
    dataCompletenessImprovements
    refreshErrors
    processingTimeSeconds
    nextRefreshRecommended
  }
}
```

### Export with Comprehensive Data

```graphql
mutation ExportKolData(
  $filters: KOLFilterInput
  $format: String = "csv"
  $includeScores: Boolean = true
  $includePredictions: Boolean = false
) {
  exportKolData(
    filters: $filters
    format: $format
    includeScores: $includeScores
    includePredictions: $includePredictions
  ) {
    success
    exportUrl
    recordCount
    fileSizeBytes
    expiresAt
    processingTimeSeconds
  }
}
```

## 📈 Analytics and Reporting

### Market Analysis

```graphql
query KolMarketAnalysis(
  $categories: [String!]!
  $budgetRange: [Float!]
  $geographicFocus: [String!]
) {
  kolMarketAnalysis(
    categories: $categories
    budgetRange: $budgetRange
    geographicFocus: $geographicFocus
  )
  # Returns comprehensive market insights
}
```

### Data Quality Reporting

```graphql
query DataQualityReport($platform: String, $category: String) {
  dataQualityReport(platform: $platform, category: $category)
  # Returns data completeness and freshness metrics
}
```

## 🔒 Security and Performance Features

### Query Complexity Analysis
- **Increased complexity limit** (2000) for sophisticated AI operations
- **Field-level authorization** for sensitive operations
- **Query depth limiting** (15 levels) for nested operations

### Rate Limiting and Caching
- **DataLoader pattern** implementation to prevent N+1 queries
- **Resolver-level caching** for expensive operations
- **Confidence-based result caching** for AI operations

### Error Handling
- **Partial response support** for missing data scenarios
- **Graceful degradation** when confidence thresholds aren't met
- **Detailed error context** with operation metadata

## 🔄 Migration Guide

### From Legacy Schema

1. **Update KOL queries** to include new scoring components:
   ```diff
   kols {
     score {
       overallScore
   +   scoreComponents {
   +     roiScore
   +     audienceQualityScore
   +     overallConfidence
   +   }
     }
   }
   ```

2. **Enhance budget optimization calls**:
   ```diff
   optimizeBudget(
     campaignId: $id
     objective: $objective
   + totalBudget: $budget
   + constraints: $constraints
   + generateAlternatives: true
   ) {
     optimizedPlan {
       id
   +   optimizationScore
   +   riskAssessment
     }
   + alternativePlans { ... }
   }
   ```

3. **Update filtering for comprehensive criteria**:
   ```diff
   kols(filters: {
     platform: INSTAGRAM
     tier: MICRO
   + minAudienceQuality: 0.8
   + requireCompleteData: true
   + maxFakeFollowers: 0.1
   }) { ... }
   ```

## 🎯 Best Practices

### Performance Optimization
1. **Use fragments** for reusable KOL data structures
2. **Specify confidence thresholds** appropriate for your use case
3. **Enable semantic matching** only when needed (computationally expensive)
4. **Batch KOL operations** using bulk mutations

### Data Quality Management
1. **Check data completeness** before making critical decisions
2. **Monitor confidence scores** for AI-generated results
3. **Use data quality reports** to identify refresh needs
4. **Set appropriate confidence thresholds** (0.7+ recommended for production)

### Budget Optimization
1. **Generate alternatives** for comparison and risk assessment
2. **Include risk analysis** for high-value campaigns
3. **Use tier constraints** to ensure diversification
4. **Monitor optimization scores** and constraint satisfaction

## 📚 Example Integration

### React Component with Apollo Client

```typescript
import { useQuery } from '@apollo/client';
import { ENHANCED_KOL_MATCHING_QUERY } from './sophisticated-queries';

const EnhancedKOLMatching: React.FC<{ campaignId: string }> = ({ campaignId }) => {
  const { data, loading, error } = useQuery(ENHANCED_KOL_MATCHING_QUERY, {
    variables: {
      campaignId,
      confidenceThreshold: 0.8,
      enableSemanticMatching: true
    }
  });

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorDisplay error={error} />;

  const { matchedKols, processingTimeSeconds, dataQualitySummary } = data.matchKolsForCampaign;

  return (
    <div>
      <KOLMatchingResults kols={matchedKols} />
      <ProcessingMetadata 
        processingTime={processingTimeSeconds}
        dataQuality={dataQualitySummary} 
      />
    </div>
  );
};
```

This enhanced GraphQL schema provides a comprehensive API that fully leverages the sophisticated algorithms implemented in your POC2 and POC4 services, enabling powerful KOL matching and budget optimization capabilities with proper data quality management and risk assessment.