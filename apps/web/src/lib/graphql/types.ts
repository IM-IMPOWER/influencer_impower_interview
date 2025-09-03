// AIDEV-NOTE: 250903170015 - GraphQL TypeScript types for the KOL Platform

export interface KOL {
  id: string
  handle: string
  platform: string
  followers: number
  engagement: number
  categories: string[]
  location: string
  avatar: string
  rates?: {
    post: number
    story: number
    reel: number
  }
  semanticScore?: number
  createdAt: string
  updatedAt: string
  bio?: string
  contentSamples?: ContentSample[]
  campaignHistory?: CampaignHistory[]
  audienceDemographics?: AudienceDemographics
  relationshipStatus?: string
  lastContact?: string
  activeCampaigns?: Campaign[]
}

export interface ContentSample {
  url: string
  type: 'post' | 'story' | 'reel' | 'video'
  metrics: {
    likes: number
    shares: number
    comments: number
    views: number
  }
}

export interface CampaignHistory {
  id: string
  name: string
  completedAt: string
  performance: {
    reach: number
    engagement: number
    conversions: number
  }
}

export interface AudienceDemographics {
  ageGroups: {
    range: string
    percentage: number
  }[]
  genderSplit: {
    male: number
    female: number
    other: number
  }
  topLocations: {
    country: string
    percentage: number
  }[]
}

export interface Campaign {
  id: string
  name: string
  status: 'Active' | 'Planning' | 'Completed' | 'Pending'
  brief: string
  budget: number
  progress: number
  totalReach: string
  avgEngagement: string
  kols?: number | KOL[]
  createdAt: string
  updatedAt: string
  timeline?: TimelineItem[]
  analytics?: CampaignAnalytics
  content?: CampaignContent[]
}

export interface TimelineItem {
  id: string
  title: string
  description: string
  date: string
  status: string
  type: string
}

export interface CampaignAnalytics {
  totalReach: number
  totalEngagement: number
  totalImpressions: number
  costPerEngagement: number
  costPerReach: number
  roi: number
  conversionRate: number
}

export interface CampaignContent {
  id: string
  type: 'post' | 'story' | 'reel' | 'video'
  status: 'pending' | 'approved' | 'changes_requested' | 'published'
  kol: KOL
  submittedAt?: string
  approvedAt?: string
  scheduledDate?: string
  publishedDate?: string
  metrics?: {
    likes: number
    shares: number
    comments: number
    views: number
    saves: number
  }
}

export interface Activity {
  id: string
  text: string
  time: string
  type: 'content_approval' | 'offer_accepted' | 'post_published' | 'campaign_created' | 'kol_added'
  userId?: string
  campaignId?: string
  kolId?: string
}

export interface KOLMatch {
  kol: KOL
  matchScore: number
  reasoning: string
  confidence: number
  reasoning_breakdown?: {
    category_match: number
    audience_alignment: number
    engagement_quality: number
    content_style: number
  }
}

export interface KOLFiltersInput {
  categories?: string[]
  minFollowers?: number
  maxFollowers?: number
  minEngagement?: number
  maxEngagement?: number
  location?: string
  platforms?: string[]
  verified?: boolean
  brandSafe?: boolean
}

export interface AdvancedKOLFiltersInput extends KOLFiltersInput {
  minAudienceQuality?: number
  maxFakeFollowers?: number
  requiredLanguages?: string[]
  excludeCategories?: string[]
  priceRange?: {
    min: number
    max: number
  }
  lastActiveWithin?: string
}

export interface KOLSortingInput {
  field: 'followers' | 'engagement' | 'match_score' | 'price' | 'audience_quality'
  direction: 'asc' | 'desc'
}

export interface PaginationInput {
  limit: number
  offset: number
}

export interface BudgetOptimizationInput {
  campaignId?: string
  totalBudget: number
  minTotalReach?: number
  categoryMix?: Record<string, number>
  maxKOLs?: number
  targetDemographics?: string[]
  constraints?: BudgetOptimizationConstraints
  objective: 'reach' | 'engagement' | 'conversions' | 'roi' | 'cost_efficiency'
}

export interface BudgetOptimizationConstraints {
  minNanoPercentage?: number
  minMicroPercentage?: number
  minMidPercentage?: number
  minMacroPercentage?: number
  minMegaPercentage?: number
  minKolsRequired?: number
  maxKolsAllowed?: number
  maxSingleKolPercentage?: number
  minReservedBuffer?: number
  minTotalReach?: number
  minTotalEngagement?: number
  targetRoi?: number
  maxRiskScore?: number
  diversificationRequirement?: boolean
  categoryDistribution?: Record<string, number>
  reachWeight?: number
  engagementWeight?: number
  costEfficiencyWeight?: number
  riskMitigationWeight?: number
}

export interface BudgetOptimization {
  optimizedPlan: BudgetPlan
  alternativePlans?: BudgetPlan[]
  insights?: {
    budgetDistribution: {
      category: string
      amount: number
      percentage: number
    }[]
    riskAssessment: {
      level: 'low' | 'medium' | 'high'
      factors: string[]
      mitigation: string[]
    }
    marketTrends: {
      insight: string
      impact: string
      confidence: number
    }[]
  }
}

export interface BudgetPlan {
  id: string
  name: string
  optimizationObjective: string
  allocationStrategy: string
  totalBudget: number
  availableBudget: number
  reservedBudget: number
  predictedReach: number
  predictedEngagement: number
  predictedConversions: number
  predictedRoi: number
  optimizationScore: number
  constraintsSatisfactionScore: number
  riskAssessment: string
  recommendations: BudgetRecommendation[]
  alternatives?: {
    scenario: string
    totalBudget: number
    projectedReach: number
    projectedEngagement: number
    expectedROI: number
    tradeoffs: string[]
  }[]
}

export interface BudgetRecommendation {
  kol: KOL
  suggestedBudget: number
  expectedReach: number
  expectedEngagement: number
  rationale: string
  priority: number
}

export interface CreateCampaignInput {
  name: string
  brief: string
  budget: number
  status?: string
  objectives?: string[]
  targetAudience?: string
  timeline?: {
    startDate: string
    endDate: string
  }
}

export interface UpdateCampaignInput {
  name?: string
  brief?: string
  budget?: number
  status?: string
  objectives?: string[]
  targetAudience?: string
  timeline?: {
    startDate: string
    endDate: string
  }
}

export interface CampaignFiltersInput {
  status?: string
  budget?: {
    min: number
    max: number
  }
  dateRange?: {
    start: string
    end: string
  }
  categories?: string[]
}

export interface OutreachMessageInput {
  campaignId: string
  kolId: string
  content: string
  type: 'initial' | 'follow_up' | 'negotiation' | 'final_offer'
  scheduledFor?: string
}

export interface CampaignKOLTermsInput {
  paymentAmount: number
  deliverables: string[]
  timeline: string
  specialInstructions?: string
}

export interface PerformancePredictionInput {
  campaignId: string
  kolIds: string[]
  budget: number
  duration: number
  targetMetrics: string[]
}

export interface RecommendationConstraintsInput {
  maxBudget: number
  minKOLs?: number
  maxKOLs?: number
  requiredCategories?: string[]
  excludeKOLs?: string[]
  prioritizeVerified?: boolean
  brandSafetyRequired?: boolean
}

export interface CompetitorAnalysisInput {
  competitors: string[]
  timeRange: string
  categories: string[]
  includeEstimates: boolean
}

export interface HealthCheck {
  status: 'healthy' | 'degraded' | 'unhealthy'
  timestamp: string
  version: string
  services: {
    database: 'healthy' | 'unhealthy'
    redis: 'healthy' | 'unhealthy'
    ml_service: 'healthy' | 'unhealthy'
  }
}

export interface DashboardData {
  kpis: {
    activeCampaigns: number
    totalKOLs: number
    totalReach: number
    avgEngagement: number
    totalSpend: number
    roi: number
  }
  recentActivities: Activity[]
  campaignPerformance: {
    campaignId: string
    name: string
    reach: number
    engagement: number
    spend: number
    roi: number
    status: string
  }[]
  topKOLs: (KOL & {
    totalReach: number
    avgEngagement: number
    campaignCount: number
  })[]
}

export interface DiscoverKOLsResult {
  kols: KOL[]
  total: number
  hasMore: boolean
  facets: {
    categories: { name: string; count: number }[]
    locations: { name: string; count: number }[]
    platforms: { name: string; count: number }[]
  }
}

export interface MyKOLsResult {
  kols: (KOL & {
    relationshipStatus: string
    lastContact?: string
    activeCampaigns: Campaign[]
  })[]
  total: number
  hasMore: boolean
}

// GraphQL operation result types
export interface ApiResponse<T> {
  data?: T
  loading: boolean
  error?: any
}

export interface MutationResponse {
  success: boolean
  message: string
  data?: any
  warnings?: string[]
  processingTimeSeconds?: number
  affectedRecords?: number
}

// Real-time subscription types
export interface CampaignActivityUpdate {
  conversationId: string
  kolId: string
  status: string
  lastMessage: string
  timestamp: string
  messageType: string
}

export interface CampaignMetricsUpdate {
  campaignId: string
  totalReach: number
  totalEngagement: number
  totalImpressions: number
  activePosts: number
  pendingApprovals: number
  completedDeliverables: number
  timestamp: string
}

// Error types
export interface GraphQLError {
  message: string
  locations?: { line: number; column: number }[]
  path?: string[]
  extensions?: Record<string, any>
}

// Utility types
export type OptionalExcept<T, K extends keyof T> = Partial<T> & Pick<T, K>
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P]
}