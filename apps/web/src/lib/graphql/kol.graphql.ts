import { gql } from '@apollo/client'
import { KOL_FRAGMENT, MATCH_RESULT_FRAGMENT } from './fragments'

// AIDEV-NOTE: 250903170002 - Updated GraphQL operations with fragments and better type safety

export const DISCOVER_KOLS = gql`
  query DiscoverKOLs($filters: KOLFiltersInput!, $pagination: PaginationInput!) {
    discoverKOLs(filters: $filters, pagination: $pagination) {
      kols {
        ...KOLFragment
      }
      total
      hasMore
      facets {
        categories {
          name
          count
        }
        locations {
          name
          count
        }
        platforms {
          name
          count
        }
      }
    }
  }
  ${KOL_FRAGMENT}
`

export const MATCH_KOLS_TO_BRIEF = gql`
  query MatchKOLsToBrief($brief: String!, $limit: Int = 10) {
    matchKOLsToBrief(brief: $brief, limit: $limit) {
      ...MatchResultFragment
    }
  }
  ${MATCH_RESULT_FRAGMENT}
`

export const OPTIMIZE_BUDGET = gql`
  mutation OptimizeBudget($request: BudgetOptimizationInput!) {
    optimizeBudget(request: $request) {
      recommendedKOLs {
        kol {
          id
          handle
          followers
          engagement
        }
        suggestedSpend
        expectedReach
        rationale
      }
      totalCost
      projectedReach
      categoryBreakdown
    }
  }
`

export const OUTREACH_CONVERSATIONS = gql`
  subscription OutreachUpdates($userId: ID!) {
    outreachUpdates(userId: $userId) {
      conversationId
      kolId
      status
      lastMessage
      timestamp
      messageType
    }
  }
`

export const SEND_OUTREACH_MESSAGE = gql`
  mutation SendOutreachMessage($input: OutreachMessageInput!) {
    sendOutreachMessage(input: $input) {
      id
      content
      timestamp
      status
    }
  }
`

export const GET_KOL_BY_ID = gql`
  query GetKOLById($id: ID!) {
    kol(id: $id) {
      ...KOLFragment
      bio
      contentSamples {
        url
        type
        metrics {
          likes
          shares
          comments
          views
        }
      }
      campaignHistory {
        id
        name
        completedAt
        performance {
          reach
          engagement
          conversions
        }
      }
      audienceDemographics {
        ageGroups {
          range
          percentage
        }
        genderSplit {
          male
          female
          other
        }
        topLocations {
          country
          percentage
        }
      }
    }
  }
  ${KOL_FRAGMENT}
`

export const GET_MY_KOLS = gql`
  query GetMyKOLs($status: String, $pagination: PaginationInput) {
    myKOLs(status: $status, pagination: $pagination) {
      kols {
        ...KOLFragment
        relationshipStatus
        lastContact
        activeCampaigns {
          id
          name
          status
        }
      }
      total
      hasMore
    }
  }
  ${KOL_FRAGMENT}
`

export const ADD_KOL_TO_PLAN = gql`
  mutation AddKOLToPlan($kolId: ID!, $planId: ID!) {
    addKOLToPlan(kolId: $kolId, planId: $planId) {
      success
      plan {
        id
        kols {
          ...KOLFragment
        }
        totalBudget
        projectedReach
      }
    }
  }
  ${KOL_FRAGMENT}
`