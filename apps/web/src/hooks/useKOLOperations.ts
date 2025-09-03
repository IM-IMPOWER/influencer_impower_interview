"use client"

import { useState, useCallback, useMemo } from 'react'
import { useQuery, useMutation, useSubscription } from '@apollo/client'
import { 
  DISCOVER_KOLS, 
  MATCH_KOLS_TO_BRIEF, 
  GET_KOL_BY_ID, 
  GET_MY_KOLS,
  ADD_KOL_TO_PLAN,
  OPTIMIZE_BUDGET
} from '@/lib/graphql/kol.graphql'
import { 
  GET_ALL_CAMPAIGNS, 
  CREATE_CAMPAIGN, 
  UPDATE_CAMPAIGN,
  DELETE_CAMPAIGN,
  CAMPAIGN_ACTIVITY_SUBSCRIPTION,
  CAMPAIGN_METRICS_SUBSCRIPTION
} from '@/lib/graphql/campaign.graphql'
import { 
  ENHANCED_KOL_MATCHING_QUERY,
  BUDGET_OPTIMIZATION_QUERY,
  GET_DASHBOARD_DATA
} from '@/lib/graphql/sophisticated-queries'
import { toast } from 'sonner'

// AIDEV-NOTE: 250903170014 - Comprehensive KOL operations hooks using Apollo GraphQL

export interface KOLFilters {
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

export interface PaginationInput {
  limit: number
  offset: number
}

/**
 * Hook for KOL discovery with advanced filtering and search
 */
export function useKOLDiscovery(filters: KOLFilters = {}, pagination: PaginationInput = { limit: 20, offset: 0 }) {
  const { data, loading, error, refetch, fetchMore } = useQuery(DISCOVER_KOLS, {
    variables: { filters, pagination },
    errorPolicy: 'all',
    notifyOnNetworkStatusChange: true
  })

  const loadMore = useCallback(() => {
    return fetchMore({
      variables: {
        pagination: {
          limit: pagination.limit,
          offset: data?.discoverKOLs?.kols?.length || 0
        }
      },
      updateQuery: (prev, { fetchMoreResult }) => {
        if (!fetchMoreResult) return prev
        return {
          discoverKOLs: {
            ...fetchMoreResult.discoverKOLs,
            kols: [...(prev.discoverKOLs?.kols || []), ...(fetchMoreResult.discoverKOLs?.kols || [])],
            hasMore: fetchMoreResult.discoverKOLs.hasMore
          }
        }
      }
    })
  }, [fetchMore, data?.discoverKOLs?.kols?.length, pagination.limit])

  return {
    kols: data?.discoverKOLs?.kols || [],
    total: data?.discoverKOLs?.total || 0,
    hasMore: data?.discoverKOLs?.hasMore || false,
    facets: data?.discoverKOLs?.facets || {},
    loading,
    error,
    refetch,
    loadMore
  }
}

/**
 * Hook for semantic KOL matching based on campaign brief
 */
export function useKOLMatching(brief: string, options: { limit?: number } = {}) {
  const { data, loading, error, refetch } = useQuery(MATCH_KOLS_TO_BRIEF, {
    variables: { brief, limit: options.limit || 10 },
    skip: !brief || brief.length < 3,
    errorPolicy: 'all',
    pollInterval: 2000 // Poll for ML processing updates
  })

  return {
    matches: data?.matchKOLsToBrief || [],
    loading,
    error,
    refetch,
    hasResults: (data?.matchKOLsToBrief?.length || 0) > 0
  }
}

/**
 * Hook for enhanced KOL matching with sophisticated scoring
 */
export function useEnhancedKOLMatching(campaignId: string, options: {
  limit?: number
  confidenceThreshold?: number
  enableSemanticMatching?: boolean
} = {}) {
  const { data, loading, error, refetch } = useQuery(ENHANCED_KOL_MATCHING_QUERY, {
    variables: {
      campaignId,
      limit: options.limit || 50,
      confidenceThreshold: options.confidenceThreshold || 0.7,
      enableSemanticMatching: options.enableSemanticMatching ?? true
    },
    skip: !campaignId,
    errorPolicy: 'all'
  })

  return {
    matchedKOLs: data?.matchKolsForCampaign?.matchedKols || [],
    metadata: {
      totalCount: data?.matchKolsForCampaign?.totalCount,
      scoringMethod: data?.matchKolsForCampaign?.scoringMethod,
      processingTime: data?.matchKolsForCampaign?.processingTimeSeconds
    },
    loading,
    error,
    refetch
  }
}

/**
 * Hook for individual KOL details
 */
export function useKOLDetails(kolId: string) {
  const { data, loading, error, refetch } = useQuery(GET_KOL_BY_ID, {
    variables: { id: kolId },
    skip: !kolId,
    errorPolicy: 'all'
  })

  return {
    kol: data?.kol,
    loading,
    error,
    refetch
  }
}

/**
 * Hook for user's KOL portfolio
 */
export function useMyKOLs(status?: string) {
  const { data, loading, error, refetch, fetchMore } = useQuery(GET_MY_KOLS, {
    variables: { 
      status,
      pagination: { limit: 20, offset: 0 }
    },
    errorPolicy: 'all'
  })

  const loadMore = useCallback(() => {
    return fetchMore({
      variables: {
        pagination: {
          limit: 20,
          offset: data?.myKOLs?.kols?.length || 0
        }
      },
      updateQuery: (prev, { fetchMoreResult }) => {
        if (!fetchMoreResult) return prev
        return {
          myKOLs: {
            ...fetchMoreResult.myKOLs,
            kols: [...(prev.myKOLs?.kols || []), ...(fetchMoreResult.myKOLs?.kols || [])],
            hasMore: fetchMoreResult.myKOLs.hasMore
          }
        }
      }
    })
  }, [fetchMore, data?.myKOLs?.kols?.length])

  return {
    kols: data?.myKOLs?.kols || [],
    total: data?.myKOLs?.total || 0,
    hasMore: data?.myKOLs?.hasMore || false,
    loading,
    error,
    refetch,
    loadMore
  }
}

/**
 * Hook for KOL plan management
 */
export function useKOLPlan() {
  const [planKOLs, setPlanKOLs] = useState<any[]>([])

  const [addKOLToPlan] = useMutation(ADD_KOL_TO_PLAN, {
    onCompleted: () => {
      toast.success('KOL added to plan successfully')
    },
    onError: (error) => {
      toast.error(`Failed to add KOL to plan: ${error.message}`)
    }
  })

  const addToPlan = useCallback((kol: any) => {
    if (!planKOLs.find(k => k.id === kol.id)) {
      setPlanKOLs(prev => [...prev, kol])
    }
  }, [planKOLs])

  const removeFromPlan = useCallback((kolId: string) => {
    setPlanKOLs(prev => prev.filter(k => k.id !== kolId))
  }, [])

  const clearPlan = useCallback(() => {
    setPlanKOLs([])
  }, [])

  const totalBudget = useMemo(() => {
    return planKOLs.reduce((sum, kol) => sum + (kol.rates?.post || 0), 0)
  }, [planKOLs])

  const projectedReach = useMemo(() => {
    return planKOLs.reduce((sum, kol) => sum + (kol.followers || 0), 0)
  }, [planKOLs])

  return {
    planKOLs,
    addToPlan,
    removeFromPlan,
    clearPlan,
    planCount: planKOLs.length,
    totalBudget,
    projectedReach,
    addKOLToPlan
  }
}

/**
 * Hook for budget optimization
 */
export function useBudgetOptimization() {
  const [optimizeBudget, { data, loading, error }] = useMutation(BUDGET_OPTIMIZATION_QUERY, {
    onCompleted: () => {
      toast.success('Budget optimization completed')
    },
    onError: (error) => {
      toast.error(`Budget optimization failed: ${error.message}`)
    }
  })

  const optimize = useCallback((campaignId: string, totalBudget: number, objective: string, constraints?: any) => {
    return optimizeBudget({
      variables: {
        campaignId,
        optimizationObjective: objective,
        totalBudget,
        constraints,
        generateAlternatives: true,
        includeRiskAnalysis: true
      }
    })
  }, [optimizeBudget])

  return {
    optimize,
    optimizedPlan: data?.optimizeBudget?.optimizedPlan,
    alternatives: data?.optimizeBudget?.alternativePlans || [],
    metadata: data?.optimizeBudget,
    loading,
    error
  }
}

/**
 * Hook for campaign management
 */
export function useCampaigns(filters?: any) {
  const { data, loading, error, refetch } = useQuery(GET_ALL_CAMPAIGNS, {
    variables: { filters, pagination: { limit: 20, offset: 0 } },
    errorPolicy: 'all'
  })

  const [createCampaign] = useMutation(CREATE_CAMPAIGN, {
    onCompleted: () => {
      toast.success('Campaign created successfully')
      refetch()
    },
    onError: (error) => {
      toast.error(`Failed to create campaign: ${error.message}`)
    },
    refetchQueries: [{ query: GET_ALL_CAMPAIGNS }]
  })

  const [updateCampaign] = useMutation(UPDATE_CAMPAIGN, {
    onCompleted: () => {
      toast.success('Campaign updated successfully')
    },
    onError: (error) => {
      toast.error(`Failed to update campaign: ${error.message}`)
    },
    refetchQueries: [{ query: GET_ALL_CAMPAIGNS }]
  })

  const [deleteCampaign] = useMutation(DELETE_CAMPAIGN, {
    onCompleted: () => {
      toast.success('Campaign deleted successfully')
      refetch()
    },
    onError: (error) => {
      toast.error(`Failed to delete campaign: ${error.message}`)
    },
    refetchQueries: [{ query: GET_ALL_CAMPAIGNS }]
  })

  return {
    campaigns: data?.campaigns?.campaigns || [],
    total: data?.campaigns?.total || 0,
    loading,
    error,
    refetch,
    createCampaign,
    updateCampaign,
    deleteCampaign
  }
}

/**
 * Hook for real-time campaign monitoring
 */
export function useCampaignMonitoring(campaignId: string) {
  const { data: activityData } = useSubscription(CAMPAIGN_ACTIVITY_SUBSCRIPTION, {
    variables: { campaignId },
    skip: !campaignId
  })

  const { data: metricsData } = useSubscription(CAMPAIGN_METRICS_SUBSCRIPTION, {
    variables: { campaignId },
    skip: !campaignId
  })

  return {
    latestActivity: activityData?.campaignActivity,
    liveMetrics: metricsData?.campaignMetrics
  }
}

/**
 * Hook for dashboard data
 */
export function useDashboard(timeRange: string = '30d') {
  const { data, loading, error, refetch } = useQuery(GET_DASHBOARD_DATA, {
    variables: { timeRange },
    errorPolicy: 'all',
    pollInterval: 30000 // Refresh every 30 seconds
  })

  return {
    kpis: data?.dashboard?.kpis,
    recentActivities: data?.dashboard?.recentActivities || [],
    campaignPerformance: data?.dashboard?.campaignPerformance || [],
    topKOLs: data?.dashboard?.topKOLs || [],
    loading,
    error,
    refetch
  }
}

/**
 * Combined hook for complete KOL platform operations
 */
export function useKOLPlatform() {
  const [activeFilters, setActiveFilters] = useState<KOLFilters>({})
  const [searchBrief, setSearchBrief] = useState('')
  
  const discovery = useKOLDiscovery(activeFilters)
  const matching = useKOLMatching(searchBrief)
  const plan = useKOLPlan()
  const budget = useBudgetOptimization()
  const campaigns = useCampaigns()
  const dashboard = useDashboard()

  return {
    // State
    activeFilters,
    setActiveFilters,
    searchBrief,
    setSearchBrief,
    
    // Operations
    discovery,
    matching,
    plan,
    budget,
    campaigns,
    dashboard,
    
    // Utilities
    isLoading: discovery.loading || matching.loading || budget.loading || campaigns.loading,
    hasError: !!(discovery.error || matching.error || budget.error || campaigns.error)
  }
}