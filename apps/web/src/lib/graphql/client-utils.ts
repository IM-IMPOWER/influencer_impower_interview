"use client"

import { ApolloCache, TypedDocumentNode } from '@apollo/client'
import { apolloClient } from '@/lib/apollo'
import { toast } from 'sonner'

// AIDEV-NOTE: 250903170016 - GraphQL client utilities for optimistic updates and cache management

/**
 * Utility function for optimistic updates with rollback on error
 */
export async function withOptimisticUpdate<TData, TVariables>(
  mutation: TypedDocumentNode<TData, TVariables>,
  variables: TVariables,
  optimisticResponse: TData,
  update?: (cache: ApolloCache<any>, result: any) => void
) {
  try {
    const result = await apolloClient.mutate({
      mutation,
      variables,
      optimisticResponse,
      update,
      errorPolicy: 'all'
    })
    
    if (result.errors) {
      throw new Error(result.errors[0].message)
    }
    
    return result
  } catch (error: any) {
    toast.error(`Operation failed: ${error.message}`)
    throw error
  }
}

/**
 * Cache update helpers
 */
export const cacheUtils = {
  /**
   * Add item to cached list
   */
  addToList: <T>(cache: ApolloCache<any>, query: any, newItem: T, listPath: string) => {
    try {
      const existingData = cache.readQuery({ query })
      if (existingData) {
        cache.writeQuery({
          query,
          data: {
            ...existingData,
            [listPath]: [...(existingData as any)[listPath], newItem]
          }
        })
      }
    } catch (error) {
      // Query not in cache yet, ignore
    }
  },

  /**
   * Remove item from cached list
   */
  removeFromList: <T extends { id: string }>(
    cache: ApolloCache<any>, 
    query: any, 
    itemId: string, 
    listPath: string
  ) => {
    try {
      const existingData = cache.readQuery({ query })
      if (existingData) {
        cache.writeQuery({
          query,
          data: {
            ...existingData,
            [listPath]: (existingData as any)[listPath].filter((item: T) => item.id !== itemId)
          }
        })
      }
    } catch (error) {
      // Query not in cache yet, ignore
    }
  },

  /**
   * Update item in cached list
   */
  updateInList: <T extends { id: string }>(
    cache: ApolloCache<any>, 
    query: any, 
    updatedItem: Partial<T> & { id: string }, 
    listPath: string
  ) => {
    try {
      const existingData = cache.readQuery({ query })
      if (existingData) {
        cache.writeQuery({
          query,
          data: {
            ...existingData,
            [listPath]: (existingData as any)[listPath].map((item: T) => 
              item.id === updatedItem.id ? { ...item, ...updatedItem } : item
            )
          }
        })
      }
    } catch (error) {
      // Query not in cache yet, ignore
    }
  },

  /**
   * Increment counter in cache
   */
  incrementCounter: (
    cache: ApolloCache<any>, 
    query: any, 
    counterPath: string, 
    increment: number = 1
  ) => {
    try {
      const existingData = cache.readQuery({ query })
      if (existingData) {
        const pathParts = counterPath.split('.')
        const newData = { ...existingData }
        let current = newData as any
        
        for (let i = 0; i < pathParts.length - 1; i++) {
          current = current[pathParts[i]]
        }
        
        current[pathParts[pathParts.length - 1]] += increment
        
        cache.writeQuery({ query, data: newData })
      }
    } catch (error) {
      // Query not in cache yet, ignore
    }
  }
}

/**
 * Error handling utilities
 */
export const errorUtils = {
  /**
   * Extract user-friendly error message from GraphQL error
   */
  getErrorMessage: (error: any): string => {
    if (error.graphQLErrors?.length > 0) {
      return error.graphQLErrors[0].message
    }
    if (error.networkError) {
      return 'Network connection error. Please check your internet connection.'
    }
    return error.message || 'An unexpected error occurred'
  },

  /**
   * Handle common GraphQL errors with appropriate user feedback
   */
  handleError: (error: any, context?: string) => {
    const message = errorUtils.getErrorMessage(error)
    const contextMessage = context ? `${context}: ${message}` : message
    
    if (error.networkError?.statusCode === 401) {
      toast.error('Please log in to continue')
      // Redirect to login or refresh auth token
    } else if (error.networkError?.statusCode === 403) {
      toast.error('You do not have permission to perform this action')
    } else if (error.networkError?.statusCode >= 500) {
      toast.error('Server error. Please try again later.')
    } else {
      toast.error(contextMessage)
    }
  }
}

/**
 * Loading state management
 */
export const loadingUtils = {
  /**
   * Create loading state manager for multiple operations
   */
  createLoadingManager: () => {
    const loadingStates = new Map<string, boolean>()
    
    return {
      setLoading: (key: string, loading: boolean) => {
        loadingStates.set(key, loading)
      },
      isLoading: (key: string) => loadingStates.get(key) || false,
      isAnyLoading: () => Array.from(loadingStates.values()).some(loading => loading),
      clear: () => loadingStates.clear()
    }
  }
}

/**
 * Pagination utilities
 */
export const paginationUtils = {
  /**
   * Create pagination state manager
   */
  createPaginationManager: (initialLimit = 20) => {
    let limit = initialLimit
    let offset = 0
    let hasMore = true
    let total = 0
    
    return {
      getVariables: () => ({ limit, offset }),
      nextPage: () => {
        if (hasMore) {
          offset += limit
        }
      },
      reset: () => {
        offset = 0
        hasMore = true
      },
      updateFromResult: (result: { total?: number; hasMore?: boolean; items?: any[] }) => {
        if (result.total !== undefined) total = result.total
        if (result.hasMore !== undefined) hasMore = result.hasMore
        if (result.items) {
          hasMore = result.items.length === limit && offset + limit < total
        }
      },
      getState: () => ({ limit, offset, hasMore, total }),
      setLimit: (newLimit: number) => {
        limit = newLimit
        offset = 0
      }
    }
  }
}

/**
 * Subscription management utilities
 */
export const subscriptionUtils = {
  /**
   * Create subscription manager with auto-cleanup
   */
  createSubscriptionManager: () => {
    const subscriptions = new Set<() => void>()
    
    return {
      add: (unsubscribe: () => void) => {
        subscriptions.add(unsubscribe)
      },
      remove: (unsubscribe: () => void) => {
        subscriptions.delete(unsubscribe)
      },
      cleanup: () => {
        subscriptions.forEach(unsubscribe => unsubscribe())
        subscriptions.clear()
      }
    }
  }
}

/**
 * Real-time data utilities
 */
export const realtimeUtils = {
  /**
   * Create real-time data manager with throttling
   */
  createRealtimeManager: <T>(updateCallback: (data: T) => void, throttleMs = 1000) => {
    let lastUpdate = 0
    let pendingData: T | null = null
    let timeoutId: NodeJS.Timeout | null = null
    
    const flushUpdate = () => {
      if (pendingData) {
        updateCallback(pendingData)
        pendingData = null
        lastUpdate = Date.now()
      }
      timeoutId = null
    }
    
    return {
      update: (data: T) => {
        pendingData = data
        const now = Date.now()
        const timeSinceLastUpdate = now - lastUpdate
        
        if (timeSinceLastUpdate >= throttleMs) {
          flushUpdate()
        } else if (!timeoutId) {
          timeoutId = setTimeout(flushUpdate, throttleMs - timeSinceLastUpdate)
        }
      },
      flush: () => {
        if (timeoutId) {
          clearTimeout(timeoutId)
          flushUpdate()
        }
      },
      cleanup: () => {
        if (timeoutId) {
          clearTimeout(timeoutId)
          timeoutId = null
        }
        pendingData = null
      }
    }
  }
}

/**
 * Data transformation utilities
 */
export const transformUtils = {
  /**
   * Transform GraphQL data to component-friendly format
   */
  transformKOL: (kol: any) => ({
    ...kol,
    followersFormatted: new Intl.NumberFormat().format(kol.followers || 0),
    engagementFormatted: `${((kol.engagement || 0) * 100).toFixed(2)}%`,
    ratesFormatted: kol.rates ? {
      post: new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(kol.rates.post),
      story: new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(kol.rates.story),
      reel: new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(kol.rates.reel)
    } : null,
    createdAtFormatted: new Date(kol.createdAt).toLocaleDateString(),
    updatedAtFormatted: new Date(kol.updatedAt).toLocaleDateString()
  }),

  /**
   * Transform campaign data
   */
  transformCampaign: (campaign: any) => ({
    ...campaign,
    budgetFormatted: new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(campaign.budget || 0),
    progressFormatted: `${campaign.progress || 0}%`,
    createdAtFormatted: new Date(campaign.createdAt).toLocaleDateString(),
    statusColor: {
      'Active': 'green',
      'Planning': 'yellow',
      'Completed': 'blue',
      'Pending': 'gray'
    }[campaign.status] || 'gray'
  })
}

/**
 * Performance monitoring utilities
 */
export const performanceUtils = {
  /**
   * Monitor GraphQL operation performance
   */
  createPerformanceMonitor: () => {
    const metrics = new Map<string, { count: number; totalTime: number; avgTime: number }>()
    
    return {
      start: (operationName: string) => {
        const startTime = performance.now()
        return {
          end: () => {
            const endTime = performance.now()
            const duration = endTime - startTime
            
            const existing = metrics.get(operationName) || { count: 0, totalTime: 0, avgTime: 0 }
            existing.count++
            existing.totalTime += duration
            existing.avgTime = existing.totalTime / existing.count
            
            metrics.set(operationName, existing)
            
            if (duration > 1000) {
              console.warn(`Slow GraphQL operation: ${operationName} took ${duration.toFixed(2)}ms`)
            }
          }
        }
      },
      getMetrics: () => Object.fromEntries(metrics),
      reset: () => metrics.clear()
    }
  }
}