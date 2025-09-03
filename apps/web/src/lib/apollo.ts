import { ApolloClient, InMemoryCache, createHttpLink, split, from } from '@apollo/client'
import { setContext } from '@apollo/client/link/context'
import { onError } from '@apollo/client/link/error'
import { GraphQLWsLink } from '@apollo/client/link/subscriptions'
import { createClient } from 'graphql-ws'
import { getMainDefinition } from '@apollo/client/utilities'
import { toast } from 'sonner'

// AIDEV-NOTE: 250903170003 - Enhanced Apollo Client setup with error handling and better auth

// Error handling link
const errorLink = onError(({ graphQLErrors, networkError, operation, forward }) => {
  if (graphQLErrors) {
    graphQLErrors.forEach(({ message, locations, path }) => {
      console.error(`GraphQL error: Message: ${message}, Location: ${locations}, Path: ${path}`)
      toast.error(`GraphQL error: ${message}`)
    })
  }

  if (networkError) {
    console.error(`Network error: ${networkError}`)
    toast.error('Network connection error')
  }
})

// HTTP Link for queries/mutations
const httpLink = createHttpLink({
  uri: process.env.NEXT_PUBLIC_GRAPHQL_ENDPOINT || 'http://localhost:8000/graphql',
  credentials: 'include', // Include cookies for Better-Auth
})

// Auth Link
const authLink = setContext((_, { headers }) => {
  // Better-Auth uses HTTP-only cookies, but we can add additional headers if needed
  return {
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    }
  }
})

// WebSocket Link for subscriptions (real-time)
let wsLink: GraphQLWsLink | null = null

if (typeof window !== 'undefined') {
  wsLink = new GraphQLWsLink(
    createClient({
      url: process.env.NEXT_PUBLIC_WS_ENDPOINT || 'ws://localhost:8000/graphql',
      connectionParams: async () => {
        // Add auth token from cookies if available
        return {
          credentials: 'include'
        }
      },
      on: {
        error: (error) => {
          console.error('WebSocket error:', error)
        }
      }
    })
  )
}

// Split link - WebSocket for subscriptions, HTTP for queries/mutations
const splitLink = wsLink ? split(
  ({ query }) => {
    const definition = getMainDefinition(query)
    return (
      definition.kind === 'OperationDefinition' &&
      definition.operation === 'subscription'
    )
  },
  wsLink,
  from([errorLink, authLink, httpLink])
) : from([errorLink, authLink, httpLink])

// Cache configuration with proper type policies
const cache = new InMemoryCache({
  typePolicies: {
    Query: {
      fields: {
        discoverKOLs: {
          keyArgs: ['filters'],
          merge(existing = { kols: [], total: 0, hasMore: false, facets: {} }, incoming) {
            return {
              ...incoming,
              kols: [...existing.kols, ...incoming.kols]
            }
          }
        },
        myKOLs: {
          keyArgs: ['status'],
          merge(existing = { kols: [], total: 0, hasMore: false }, incoming) {
            return {
              ...incoming,
              kols: [...existing.kols, ...incoming.kols]
            }
          }
        }
      }
    },
    KOL: {
      fields: {
        campaigns: {
          merge(existing = [], incoming = []) {
            return incoming
          }
        }
      }
    }
  }
})

export const apolloClient = new ApolloClient({
  link: splitLink,
  cache,
  defaultOptions: {
    watchQuery: { 
      errorPolicy: 'all',
      fetchPolicy: 'cache-and-network'
    },
    query: { 
      errorPolicy: 'all',
      fetchPolicy: 'cache-first'
    },
    mutate: {
      errorPolicy: 'all'
    }
  },
  connectToDevTools: process.env.NODE_ENV === 'development'
})

// Helper function to reset Apollo cache
export const resetApolloCache = () => {
  return apolloClient.clearStore()
}