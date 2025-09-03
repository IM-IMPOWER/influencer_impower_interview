import { gql } from '@apollo/client'

// AIDEV-NOTE: 250903170001 - GraphQL fragments for reusable data structures across queries

export const KOL_FRAGMENT = gql`
  fragment KOLFragment on KOL {
    id
    handle
    platform
    followers
    engagement
    categories
    location
    avatar
    rates {
      post
      story
      reel
    }
    semanticScore
    createdAt
    updatedAt
  }
`

export const CAMPAIGN_FRAGMENT = gql`
  fragment CampaignFragment on Campaign {
    id
    name
    status
    brief
    budget
    progress
    totalReach
    avgEngagement
    kols
    createdAt
    updatedAt
  }
`

export const ACTIVITY_FRAGMENT = gql`
  fragment ActivityFragment on Activity {
    id
    text
    time
    type
    userId
    campaignId
    kolId
  }
`

export const MATCH_RESULT_FRAGMENT = gql`
  fragment MatchResultFragment on KOLMatch {
    kol {
      ...KOLFragment
    }
    matchScore
    reasoning
    confidence
    reasoning_breakdown {
      category_match
      audience_alignment
      engagement_quality
      content_style
    }
  }
  ${KOL_FRAGMENT}
`