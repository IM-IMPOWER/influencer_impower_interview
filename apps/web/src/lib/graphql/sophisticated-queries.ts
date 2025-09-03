/**
 * Sophisticated GraphQL Queries for Enhanced KOL Platform
 * 
 * This file contains example queries and mutations that demonstrate
 * the advanced capabilities of the POC2 (KOL Matching) and POC4 (Budget Optimizer)
 * implementations with comprehensive type safety.
 */

import { gql } from '@apollo/client';

// AIDEV-NOTE: 250903170000 - Basic health check query for system status monitoring
export const HEALTH_CHECK = gql`
  query HealthCheck {
    healthCheck {
      status
      timestamp
      version
      services
    }
  }
`;

// AIDEV-NOTE: Enhanced KOL matching with sophisticated multi-factor scoring
export const ENHANCED_KOL_MATCHING_QUERY = gql`
  query EnhancedKOLMatching(
    $campaignId: String!
    $limit: Int = 50
    $confidenceThreshold: Float = 0.7
    $enableSemanticMatching: Boolean = true
  ) {
    matchKolsForCampaign(
      campaignId: $campaignId
      limit: $limit
      confidenceThreshold: $confidenceThreshold
      enableSemanticMatching: $enableSemanticMatching
    ) {
      matchedKols {
        id
        username
        displayName
        platform
        tier
        primaryCategory
        isVerified
        isBrandSafe
        
        # Enhanced metrics with comprehensive data points
        metrics {
          followerCount
          engagementRate
          avgLikes
          avgComments
          avgViews
          audienceQualityScore
          fakeFollowerPercentage
          postsLast30Days
          campaignSuccessRate
          responseRate
          ratePerPost
          dataCompletenessScore
        }
        
        # Sophisticated scoring components (POC2)
        score {
          overallScore
          scoreComponents {
            roiScore
            audienceQualityScore
            brandSafetyScore
            contentRelevanceScore
            demographicFitScore
            reliabilityScore
            overallConfidence
            scoringTimestamp
            dataFreshnessDays
          }
          semanticMatching {
            similarityScore
            contentMatchScore
            brandAffinityScore
            embeddingConfidence
            matchedContentCategories
            semanticKeywords
          }
          performancePrediction {
            predictedReach
            predictedEngagement
            predictedConversions
            predictedRoi
            predictionConfidence
            riskFactors
            historicalCampaignCount
          }
          scoringAlgorithm
          algorithmVersion
        }
        
        # Data quality indicators
        dataCompleteness
        hasSufficientData
        missingDataFields
      }
      
      # Enhanced metadata
      totalCount
      scoringMethod
      algorithmVersion
      weightsUsed
      totalCandidatesEvaluated
      candidatesPassedScoring
      processingTimeSeconds
      dataQualitySummary
      missingDataWarnings
      semanticMatchingEnabled
      embeddingQualityScore
    }
  }
`;

// AIDEV-NOTE: Direct KOL matching with comprehensive requirements
export const MATCH_KOLS_WITH_REQUIREMENTS = gql`
  query MatchKolsWithRequirements(
    $requirements: CampaignRequirementsInput!
    $limit: Int = 50
    $confidenceThreshold: Float = 0.7
  ) {
    matchKolsWithRequirements(
      requirements: $requirements
      limit: $limit
      confidenceThreshold: $confidenceThreshold
    ) {
      matchedKols {
        id
        username
        displayName
        platform
        tier
        score {
          overallScore
          scoreComponents {
            roiScore
            audienceQualityScore
            brandSafetyScore
            overallConfidence
          }
        }
      }
      matchCriteria
      confidenceScores
      processingTimeSeconds
    }
  }
`;

// AIDEV-NOTE: Sophisticated budget optimization (POC4)
export const BUDGET_OPTIMIZATION_QUERY = gql`
  query OptimizeBudget(
    $campaignId: String!
    $optimizationObjective: String!
    $totalBudget: Float!
    $constraints: BudgetOptimizationConstraintsInput
    $generateAlternatives: Boolean = true
    $includeRiskAnalysis: Boolean = true
  ) {
    optimizeBudget(
      campaignId: $campaignId
      optimizationObjective: $optimizationObjective
      totalBudget: $totalBudget
      constraints: $constraints
      generateAlternatives: $generateAlternatives
      includeRiskAnalysis: $includeRiskAnalysis
    ) {
      # Primary optimized plan
      optimizedPlan {
        id
        name
        optimizationObjective
        allocationStrategy
        optimizationAlgorithm
        algorithmVersion
        totalBudget
        availableBudget
        reservedBudget
        
        # Performance predictions
        predictedReach
        predictedEngagement
        predictedConversions
        predictedRoi
        
        # Optimization metrics
        optimizationScore
        constraintsSatisfactionScore
        riskAssessment
        objectiveWeights
        tierRequirements
        
        # Allocations with detailed breakdown
        allocations {
          id
          allocationName
          allocationType
          allocatedAmount
          targetTier
          targetCategory
          targetKolIds
          expectedReach
          expectedEngagement
          expectedConversions
          expectedRoi
          efficiencyScore
          riskScore
          optimizationConfidence
          allocationConstraints
        }
        
        hasAlternatives
        alternativeCount
      }
      
      # Alternative scenarios
      alternativePlans {
        id
        name
        totalBudget
        optimizationScore
        predictedRoi
        riskAssessment
      }
      
      # Comprehensive analysis
      optimizationMetadata
      constraintSatisfaction
      efficiencyMetrics
      riskAssessment
      sensitivityAnalysis
      performanceForecasts
      confidenceIntervals
      
      # Processing metadata
      optimizationAlgorithm
      algorithmVersion
      processingTimeSeconds
      iterationsPerformed
    }
  }
`;

// AIDEV-NOTE: Budget optimization scenarios for comparison
export const BUDGET_SCENARIOS_QUERY = gql`
  query BudgetOptimizationScenarios(
    $campaignId: String!
    $budgetRanges: [Float!]!
    $objectives: [String!]!
  ) {
    budgetOptimizationScenarios(
      campaignId: $campaignId
      budgetRanges: $budgetRanges
      objectives: $objectives
    ) {
      optimizedPlan {
        id
        name
        totalBudget
        optimizationObjective
        optimizationScore
        predictedRoi
        predictedReach
        predictedEngagement
      }
      optimizationMetadata
      efficiencyMetrics
      processingTimeSeconds
    }
  }
`;

// AIDEV-NOTE: Enhanced analytics with predictions and risk factors
export const KOL_PERFORMANCE_ANALYTICS = gql`
  query KolPerformanceAnalytics(
    $kolIds: [String!]!
    $dateFrom: String
    $dateTo: String
    $includePredictions: Boolean = true
    $includeRiskFactors: Boolean = true
  ) {
    kolPerformanceAnalytics(
      kolIds: $kolIds
      dateFrom: $dateFrom
      dateTo: $dateTo
      includePredictions: $includePredictions
      includeRiskFactors: $includeRiskFactors
    )
  }
`;

// AIDEV-NOTE: Market analysis for strategic planning
export const KOL_MARKET_ANALYSIS = gql`
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
  }
`;

// AIDEV-NOTE: Data quality reporting
export const DATA_QUALITY_REPORT = gql`
  query DataQualityReport(
    $platform: String
    $category: String
  ) {
    dataQualityReport(
      platform: $platform
      category: $category
    )
  }
`;

// AIDEV-NOTE: Enhanced KOL filtering with comprehensive criteria
export const ENHANCED_KOLS_QUERY = gql`
  query EnhancedKols(
    $filters: KOLFilterInput
    $limit: Int = 20
    $offset: Int = 0
    $sortBy: String
    $search: String
  ) {
    kols(
      filters: $filters
      limit: $limit
      offset: $offset
      sortBy: $sortBy
      search: $search
    ) {
      id
      username
      displayName
      platform
      tier
      primaryCategory
      isVerified
      isBrandSafe
      brandSafetyNotes
      brandSafetyLastUpdated
      
      metrics {
        followerCount
        engagementRate
        audienceQualityScore
        fakeFollowerPercentage
        campaignSuccessRate
        ratePerPost
        dataCompletenessScore
      }
      
      score {
        overallScore
        scoreComponents {
          overallConfidence
          roiScore
          audienceQualityScore
          brandSafetyScore
        }
      }
      
      dataCompleteness
      hasSufficientData
      missingDataFields
    }
  }
`;

// AIDEV-NOTE: Mutations for sophisticated operations

export const PARSE_CAMPAIGN_BRIEF = gql`
  mutation ParseCampaignBrief($brief: BriefUploadInput!) {
    parseCampaignBrief(brief: $brief) {
      success
      message
      campaignRequirements
      parsingConfidence
      ambiguousRequirements
      missingRequirements
      extractedEntities
      sentimentAnalysis
      isActionable
      requiredManualInputs
    }
  }
`;

// AIDEV-NOTE: 250903170000 - POC2 KOL-to-brief matching mutation with file upload
export const MATCH_KOLS_TO_BRIEF = gql`
  mutation MatchKolsToBrief(
    $briefFile: Upload!
    $confidenceThreshold: Float = 0.7
    $limit: Int = 50
    $enableSemanticMatching: Boolean = true
  ) {
    matchKolsToBrief(
      briefFile: $briefFile
      confidenceThreshold: $confidenceThreshold
      limit: $limit
      enableSemanticMatching: $enableSemanticMatching
    ) {
      success
      message
      processingId
      extractedRequirements {
        targetKolTiers
        targetCategories
        totalBudget
        minFollowerCount
        maxFollowerCount
        minEngagementRate
        targetDemographics
        targetLocations
        targetLanguages
        requireBrandSafe
        requireVerified
        campaignObjective
        contentTheme
        keywords
        excludedKeywords
        deliverables
        timeline
        specialRequirements
      }
      matchedKols {
        id
        username
        displayName
        platform
        tier
        primaryCategory
        isVerified
        isBrandSafe
        profileImageUrl
        followerCount
        engagementRate
        averageViews
        ratePerPost
        
        score {
          overallScore
          scoreComponents {
            roiScore
            audienceQualityScore
            brandSafetyScore
            contentRelevanceScore
            demographicFitScore
            overallConfidence
          }
          semanticMatching {
            similarityScore
            contentMatchScore
            brandAffinityScore
            matchedContentCategories
            semanticKeywords
          }
          performancePrediction {
            predictedReach
            predictedEngagement
            predictedConversions
            predictedRoi
            predictionConfidence
          }
        }
        
        matchReasons
        fitExplanation
        potentialConcerns
        recommendedApproach
      }
      
      briefAnalysis {
        extractedThemes
        brandTone
        targetAudience
        contentRequirements
        budgetInsights
        timelineAnalysis
        feasibilityScore
        complexityLevel
      }
      
      totalMatches
      processingTimeSeconds
      briefParsingConfidence
      matchingAlgorithm
      dataQualityWarnings
    }
  }
`;

// AIDEV-NOTE: 250903170000 - Query to get status of brief processing
export const GET_BRIEF_PROCESSING_STATUS = gql`
  query GetBriefProcessingStatus($processingId: String!) {
    briefProcessingStatus(processingId: $processingId) {
      id
      status
      progress
      message
      startedAt
      completedAt
      errorDetails
      resultUrl
    }
  }
`;

export const CREATE_OPTIMIZED_BUDGET_PLAN = gql`
  mutation CreateOptimizedBudgetPlan(
    $input: BudgetPlanCreateInput!
    $autoOptimize: Boolean = true
    $generateAlternatives: Boolean = true
  ) {
    createBudgetPlan(
      input: $input
      autoOptimize: $autoOptimize
      generateAlternatives: $generateAlternatives
    ) {
      success
      message
      data
      warnings
      processingTimeSeconds
      affectedRecords
    }
  }
`;

export const ENHANCED_DATA_REFRESH = gql`
  mutation EnhancedDataRefresh(
    $kolIds: [String!]
    $platform: String
    $priority: String = "normal"
    $forceRefresh: Boolean = false
  ) {
    triggerKolDataRefresh(
      kolIds: $kolIds
      platform: $platform
      priority: $priority
      forceRefresh: $forceRefresh
    ) {
      success
      message
      kolsRefreshed
      platformsUpdated
      dataCompletenessImprovements
      newMetricsCollected
      refreshErrors
      rateLimitWarnings
      processingTimeSeconds
      nextRefreshRecommended
    }
  }
`;

export const EXPORT_KOL_DATA = gql`
  mutation ExportKolData(
    $kolIds: [String!]
    $filters: KOLFilterInput
    $format: String = "csv"
    $includeScores: Boolean = true
    $includePredictions: Boolean = false
  ) {
    exportKolData(
      kolIds: $kolIds
      filters: $filters
      format: $format
      includeScores: $includeScores
      includePredictions: $includePredictions
    ) {
      success
      message
      exportUrl
      fileFormat
      recordCount
      fileSizeBytes
      expiresAt
      exportId
      processingTimeSeconds
    }
  }
`;

export const BULK_UPDATE_BRAND_SAFETY = gql`
  mutation BulkUpdateBrandSafety(
    $updates: [String!]!
    $reviewerId: String!
  ) {
    bulkUpdateBrandSafety(
      updates: $updates
      reviewerId: $reviewerId
    ) {
      success
      message
      data
      warnings
      processingTimeSeconds
      affectedRecords
      operationId
      timestamp
    }
  }
`;

// AIDEV-NOTE: Fragment for reusable KOL data structure
export const ENHANCED_KOL_FRAGMENT = gql`
  fragment EnhancedKolData on KOL {
    id
    username
    displayName
    platform
    tier
    primaryCategory
    secondaryCategories
    isVerified
    isBrandSafe
    languages
    targetDemographics
    
    metrics {
      followerCount
      followingCount
      engagementRate
      avgLikes
      avgComments
      avgViews
      audienceQualityScore
      fakeFollowerPercentage
      postsLast30Days
      campaignSuccessRate
      responseRate
      ratePerPost
      dataCompletenessScore
      metricsDate
    }
    
    score {
      overallScore
      scoreComponents {
        roiScore
        audienceQualityScore
        brandSafetyScore
        contentRelevanceScore
        demographicFitScore
        reliabilityScore
        overallConfidence
        scoringTimestamp
      }
      scoringAlgorithm
      algorithmVersion
    }
    
    dataCompleteness
    hasSufficientData
    missingDataFields
    createdAt
    lastScraped
    lastScored
  }
`;

// AIDEV-NOTE: Fragment for comprehensive budget plan data
export const COMPREHENSIVE_BUDGET_PLAN_FRAGMENT = gql`
  fragment ComprehensiveBudgetPlan on BudgetPlan {
    id
    name
    description
    status
    optimizationObjective
    allocationStrategy
    optimizationAlgorithm
    algorithmVersion
    
    totalBudget
    availableBudget
    reservedBudget
    
    predictedReach
    predictedEngagement
    predictedConversions
    predictedRoi
    
    optimizationScore
    constraintsSatisfactionScore
    riskAssessment
    objectiveWeights
    tierRequirements
    
    createdAt
    optimizationCompletedAt
    
    allocations {
      id
      allocationName
      allocationType
      allocatedAmount
      spentAmount
      targetTier
      targetCategory
      targetKolIds
      expectedReach
      expectedEngagement
      expectedConversions
      expectedRoi
      efficiencyScore
      riskScore
      optimizationConfidence
      isCommitted
      allocationConstraints
      alternativeAllocations
    }
    
    hasAlternatives
    alternativeCount
  }
`;

// AIDEV-NOTE: Type definitions for TypeScript integration
export interface CampaignRequirements {
  targetKolTiers: string[];
  targetCategories: string[];
  totalBudget: number;
  minFollowerCount?: number;
  maxFollowerCount?: number;
  minEngagementRate?: number;
  minAvgViews?: number;
  targetDemographics?: string; // JSON string
  targetLocations: string[];
  targetLanguages: string[];
  targetAgeRanges: string[];
  requireBrandSafe: boolean;
  requireVerified: boolean;
  excludeControversial: boolean;
  requiredHashtags: string[];
  excludedHashtags: string[];
  contentSentimentRequirements?: string;
  campaignObjective: string;
  expectedConversionRate?: number;
}

export interface BudgetOptimizationConstraints {
  minNanoPercentage?: number;
  minMicroPercentage?: number;
  minMidPercentage?: number;
  minMacroPercentage?: number;
  minMegaPercentage?: number;
  minKolsRequired: number;
  maxKolsAllowed?: number;
  maxSingleKolPercentage?: number;
  minReservedBuffer: number;
  minTotalReach?: number;
  minTotalEngagement?: number;
  targetRoi?: number;
  maxRiskScore?: number;
  diversificationRequirement: boolean;
  categoryDistribution?: string; // JSON string
  reachWeight: number;
  engagementWeight: number;
  costEfficiencyWeight: number;
  riskMitigationWeight: number;
}

export interface BriefUpload {
  briefContent: string;
  filename: string;
  autoExtractRequirements: boolean;
  confidenceThreshold: number;
  manualOverrides?: string; // JSON string
}