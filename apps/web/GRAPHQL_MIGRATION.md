# KOL Platform - GraphQL Migration Documentation

## Overview

This document outlines the complete migration from tRPC to Apollo GraphQL client for the KOL Platform frontend. The migration provides enhanced type safety, better developer experience, real-time capabilities, and improved performance.

## Architecture Changes

### Before (tRPC)
```
Frontend (React) -> tRPC Client -> HTTP -> Backend (FastAPI with tRPC router)
```

### After (Apollo GraphQL)
```
Frontend (React) -> Apollo Client -> GraphQL HTTP/WebSocket -> Backend (FastAPI with GraphQL)
```

## Key Components

### 1. Apollo Client Setup (`/src/lib/apollo.ts`)

**Features:**
- HTTP link for queries/mutations
- WebSocket link for real-time subscriptions
- Authentication via HTTP-only cookies (Better-Auth)
- Error handling with user notifications
- Optimized caching policies
- Development tools integration

**Configuration:**
```typescript
export const apolloClient = new ApolloClient({
  link: splitLink, // HTTP + WebSocket
  cache: new InMemoryCache({
    typePolicies: {
      // Optimized caching for KOL and campaign data
    }
  }),
  defaultOptions: {
    watchQuery: { errorPolicy: 'all', fetchPolicy: 'cache-and-network' },
    query: { errorPolicy: 'all', fetchPolicy: 'cache-first' }
  }
})
```

### 2. GraphQL Operations

#### KOL Operations (`/src/lib/graphql/kol.graphql.ts`)
- `DISCOVER_KOLS` - Advanced KOL discovery with filtering
- `MATCH_KOLS_TO_BRIEF` - AI-powered semantic matching
- `GET_KOL_BY_ID` - Detailed KOL profiles
- `GET_MY_KOLS` - User's KOL portfolio
- `OPTIMIZE_BUDGET` - Budget optimization algorithms

#### Campaign Operations (`/src/lib/graphql/campaign.graphql.ts`)
- `GET_ALL_CAMPAIGNS` - Campaign management
- `CREATE_CAMPAIGN` - Campaign creation
- `UPDATE_CAMPAIGN` - Campaign updates
- `APPROVE_CONTENT` - Content approval workflow
- `CAMPAIGN_ACTIVITY_SUBSCRIPTION` - Real-time updates

#### Sophisticated Queries (`/src/lib/graphql/sophisticated-queries.ts`)
- `ENHANCED_KOL_MATCHING_QUERY` - Advanced matching with ML scoring
- `BUDGET_OPTIMIZATION_QUERY` - Comprehensive budget optimization
- `GET_DASHBOARD_DATA` - Dashboard analytics
- `REAL_TIME_CAMPAIGN_MONITORING` - Live campaign monitoring

### 3. React Hooks Integration

#### Comprehensive Hook System (`/src/hooks/useKOLOperations.ts`)

**Available Hooks:**
- `useKOLDiscovery()` - KOL discovery with filtering
- `useKOLMatching(brief)` - Semantic KOL matching
- `useEnhancedKOLMatching()` - Advanced matching with ML
- `useKOLDetails(id)` - Individual KOL profiles
- `useMyKOLs()` - User's KOL portfolio
- `useKOLPlan()` - KOL plan management
- `useBudgetOptimization()` - Budget optimization
- `useCampaigns()` - Campaign CRUD operations
- `useCampaignMonitoring()` - Real-time monitoring
- `useDashboard()` - Dashboard data
- `useKOLPlatform()` - Combined operations

**Usage Example:**
```typescript
function KOLDiscoveryComponent() {
  const { kols, loading, error, loadMore } = useKOLDiscovery({
    categories: ['lifestyle', 'fitness'],
    minFollowers: 10000,
    location: 'Thailand'
  });

  return (
    <div>
      {kols.map(kol => <KOLCard key={kol.id} kol={kol} />)}
      {loading && <LoadingSpinner />}
    </div>
  );
}
```

### 4. Type Safety

#### TypeScript Integration (`/src/lib/graphql/types.ts`)
- Complete type definitions for all GraphQL operations
- Interface definitions for KOL, Campaign, Activity, etc.
- Input types for filters, mutations, and subscriptions
- Response types with proper null handling

#### Code Generation (`codegen.ts`)
```bash
npm run generate-types  # Generate TypeScript types from schema
npm run codegen:watch   # Watch mode for development
```

### 5. Updated Components

#### Dashboard (`/src/components/pages/Dashboard.tsx`)
- Real-time KPI updates
- GraphQL-powered activity feed
- Campaign performance metrics
- Loading states and error handling

#### Campaigns (`/src/components/pages/Campaigns.tsx`)
- CRUD operations with GraphQL
- Advanced filtering and search
- Optimistic updates
- Real-time status updates

#### KOL Discovery (`/src/components/kol/KOLDiscovery.tsx`)
- Semantic search capabilities
- Advanced filtering
- Infinite scrolling
- Match scoring visualization

### 6. Real-time Features

#### Subscriptions
- Campaign activity updates
- Live metrics monitoring
- Content approval notifications
- KOL status changes

**Implementation:**
```typescript
const { latestActivity, liveMetrics } = useCampaignMonitoring(campaignId);
```

### 7. Performance Optimizations

#### Caching Strategy
- Query-level caching with TTL
- Optimistic updates for mutations
- Background refetching
- Cache invalidation patterns

#### Bundle Optimization
- Code splitting by feature
- Lazy loading of GraphQL operations
- Tree shaking of unused queries
- Image optimization for KOL avatars

### 8. Error Handling

#### Comprehensive Error Management
- GraphQL error parsing
- Network error handling
- User-friendly error messages
- Retry mechanisms
- Offline support planning

### 9. Development Experience

#### Developer Tools
- Apollo DevTools integration
- GraphQL schema introspection
- Query performance monitoring
- Real-time subscription debugging

#### Code Quality
- ESLint rules for GraphQL
- Prettier formatting
- TypeScript strict mode
- Automated testing setup

## Migration Benefits

### 1. Type Safety
- End-to-end type safety from schema to UI
- Compile-time error detection
- IntelliSense support
- Reduced runtime errors

### 2. Developer Experience
- Single query language for all operations
- Powerful developer tools
- Code generation
- Schema-first development

### 3. Performance
- Efficient caching
- Query batching
- Optimistic updates
- Real-time capabilities

### 4. Scalability
- Modular operation structure
- Reusable fragments
- Subscription-based real-time features
- Efficient data fetching

## Environment Variables

```env
NEXT_PUBLIC_GRAPHQL_ENDPOINT=http://localhost:8000/graphql
NEXT_PUBLIC_WS_ENDPOINT=ws://localhost:8000/graphql
GRAPHQL_SCHEMA_URL=http://localhost:8000/graphql
```

## Scripts

```json
{
  "dev": "next dev --turbopack --port=3001",
  "build": "next build",
  "codegen": "graphql-codegen --config codegen.ts",
  "codegen:watch": "graphql-codegen --config codegen.ts --watch",
  "generate-types": "npm run codegen"
}
```

## File Structure

```
src/
├── lib/
│   ├── apollo.ts                 # Apollo Client configuration
│   └── graphql/
│       ├── kol.graphql.ts       # KOL operations
│       ├── campaign.graphql.ts   # Campaign operations
│       ├── sophisticated-queries.ts # Advanced queries
│       ├── fragments.ts          # Reusable fragments
│       ├── types.ts             # TypeScript definitions
│       └── client-utils.ts      # GraphQL utilities
├── hooks/
│   ├── useKOLOperations.ts      # Comprehensive KOL hooks
│   └── useDebounce.ts          # Utility hooks
├── components/
│   ├── providers.tsx            # Apollo Provider setup
│   ├── pages/                   # Updated page components
│   └── kol/                     # KOL-specific components
└── app/
    ├── layout.tsx              # Updated app layout
    ├── page.tsx               # Updated home page
    └── [routes]/              # Updated route pages
```

## Testing Strategy

### Unit Tests
- GraphQL operation testing
- Hook testing with MockedProvider
- Component integration tests
- Error handling validation

### Integration Tests
- End-to-end GraphQL flows
- Real-time subscription testing
- Cache behavior validation
- Performance benchmarking

## Deployment Considerations

### Production Optimizations
- GraphQL endpoint security
- Query complexity analysis
- Rate limiting
- Monitoring and logging

### Environment Configuration
- Separate endpoints for dev/staging/prod
- Schema registry integration
- CDN configuration for static assets

## Troubleshooting

### Common Issues
1. **WebSocket Connection Failures**: Check WS endpoint configuration
2. **Cache Inconsistencies**: Verify cache policies and invalidation
3. **Type Generation Errors**: Ensure schema accessibility
4. **Performance Issues**: Monitor query complexity and implement pagination

### Debug Tools
- Apollo DevTools
- GraphQL Playground
- Network tab inspection
- Console error analysis

## Future Enhancements

### Planned Features
1. Offline support with cache persistence
2. Advanced subscription filtering
3. Query complexity analysis
4. Automated performance monitoring
5. Schema federation for microservices
6. Enhanced real-time collaboration features

## Conclusion

This migration from tRPC to Apollo GraphQL provides a robust, scalable, and developer-friendly foundation for the KOL Platform. The implementation includes comprehensive type safety, real-time capabilities, optimized performance, and excellent developer experience.

The new architecture supports the platform's growth while maintaining code quality and ensuring smooth user experiences across all KOL discovery, campaign management, and budget optimization workflows.