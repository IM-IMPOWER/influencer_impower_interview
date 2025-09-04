import { gql } from '@apollo/client'
import { CAMPAIGN_FRAGMENT, KOL_FRAGMENT, ACTIVITY_FRAGMENT } from './fragments'

// AIDEV-NOTE: 250903170008 - Campaign GraphQL operations with comprehensive functionality

export const GET_ALL_CAMPAIGNS = gql`
  query GetAllCampaigns($filters: CampaignFiltersInput, $pagination: PaginationInput) {
    campaigns(filters: $filters, pagination: $pagination) {
      campaigns {
        ...CampaignFragment
        kols {
          ...KOLFragment
        }
      }
      total
      hasMore
    }
  }
  ${CAMPAIGN_FRAGMENT}
  ${KOL_FRAGMENT}
`

export const GET_CAMPAIGN_BY_ID = gql`
  query GetCampaignById($id: ID!) {
    campaign(id: $id) {
      ...CampaignFragment
      kols {
        ...KOLFragment
        relationshipStatus
        lastContact
        contractStatus
        deliverables {
          id
          type
          status
          dueDate
          submittedAt
          content
          feedback
        }
      }
      timeline {
        id
        title
        description
        date
        status
        type
      }
      analytics {
        totalReach
        totalEngagement
        totalImpressions
        costPerEngagement
        costPerReach
        roi
        conversionRate
      }
      content {
        id
        type
        status
        kol {
          ...KOLFragment
        }
        submittedAt
        approvedAt
        scheduledDate
        publishedDate
        metrics {
          likes
          shares
          comments
          views
          saves
        }
      }
    }
  }
  ${CAMPAIGN_FRAGMENT}
  ${KOL_FRAGMENT}
`

export const CREATE_CAMPAIGN = gql`
  mutation CreateCampaign($input: CreateCampaignInput!) {
    createCampaign(input: $input) {
      ...CampaignFragment
    }
  }
  ${CAMPAIGN_FRAGMENT}
`

export const UPDATE_CAMPAIGN = gql`
  mutation UpdateCampaign($id: ID!, $input: UpdateCampaignInput!) {
    updateCampaign(id: $id, input: $input) {
      ...CampaignFragment
    }
  }
  ${CAMPAIGN_FRAGMENT}
`

export const DELETE_CAMPAIGN = gql`
  mutation DeleteCampaign($id: ID!) {
    deleteCampaign(id: $id) {
      success
      message
    }
  }
`

export const ADD_KOL_TO_CAMPAIGN = gql`
  mutation AddKOLToCampaign($campaignId: ID!, $kolId: ID!, $terms: CampaignKOLTermsInput) {
    addKOLToCampaign(campaignId: $campaignId, kolId: $kolId, terms: $terms) {
      success
      campaign {
        ...CampaignFragment
        kols {
          ...KOLFragment
          relationshipStatus
        }
      }
    }
  }
  ${CAMPAIGN_FRAGMENT}
  ${KOL_FRAGMENT}
`

export const REMOVE_KOL_FROM_CAMPAIGN = gql`
  mutation RemoveKOLFromCampaign($campaignId: ID!, $kolId: ID!) {
    removeKOLFromCampaign(campaignId: $campaignId, kolId: $kolId) {
      success
      campaign {
        ...CampaignFragment
      }
    }
  }
  ${CAMPAIGN_FRAGMENT}
`

export const APPROVE_CONTENT = gql`
  mutation ApproveContent($contentId: ID!, $feedback: String) {
    approveContent(contentId: $contentId, feedback: $feedback) {
      id
      status
      approvedAt
      feedback
    }
  }
`

export const REQUEST_CONTENT_CHANGES = gql`
  mutation RequestContentChanges($contentId: ID!, $feedback: String!) {
    requestContentChanges(contentId: $contentId, feedback: $feedback) {
      id
      status
      feedback
      changesRequestedAt
    }
  }
`

export const CAMPAIGN_ACTIVITY_SUBSCRIPTION = gql`
  subscription CampaignActivity($campaignId: ID!) {
    campaignActivity(campaignId: $campaignId) {
      ...ActivityFragment
    }
  }
  ${ACTIVITY_FRAGMENT}
`

export const CAMPAIGN_METRICS_SUBSCRIPTION = gql`
  subscription CampaignMetrics($campaignId: ID!) {
    campaignMetrics(campaignId: $campaignId) {
      campaignId
      totalReach
      totalEngagement
      totalImpressions
      activePosts
      pendingApprovals
      completedDeliverables
      timestamp
    }
  }
`