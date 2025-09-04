"use client"

import { useState } from 'react'
import { useQuery } from '@apollo/client'
import { DISCOVER_KOLS, MATCH_KOLS_TO_BRIEF } from '@/lib/graphql/kol.graphql'
import { Card, CardContent, Badge, Button, Input } from '@/components/ui/ui-components'
import { useDebounce } from '@/hooks/useDebounce'

// AIDEV-NOTE: 250903170007 - KOL Discovery component updated to use Apollo GraphQL

export default function KOLDiscovery() {
  const [filters, setFilters] = useState({
    categories: [],
    minFollowers: 1000,
    location: 'Thailand'
  })
  const [brief, setBrief] = useState('')
  const debouncedBrief = useDebounce(brief, 500)

  // GraphQL queries
  const { data: kolsData, loading: kolsLoading } = useQuery(DISCOVER_KOLS, {
    variables: { filters, pagination: { limit: 20, offset: 0 } }
  })

  const { data: matchData, loading: matchLoading } = useQuery(MATCH_KOLS_TO_BRIEF, {
    variables: { brief: debouncedBrief },
    skip: !debouncedBrief
  })

  return (
    <div className="space-y-6">
      {/* Semantic Search */}
      <Card>
        <CardContent className="p-6">
          <Input
            placeholder="Describe your ideal KOL (e.g., 'Office Worker', 'Beauty Guru')"
            value={brief}
            onChange={(e) => setBrief(e.target.value)}
          />
          {matchLoading && <div>Finding matches...</div>}
        </CardContent>
      </Card>

      {/* Results */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {(matchData?.matchKOLsToBrief || kolsData?.discoverKOLs.kols)?.map((item: any) => (
          <Card key={item.kol?.id || item.id}>
            <CardContent className="p-6">
              <div className="flex items-center gap-3">
                <img src={item.kol?.avatar || item.avatar} className="w-12 h-12 rounded-full" />
                <div>
                  <p className="font-semibold">{item.kol?.handle || item.handle}</p>
                  <p className="text-sm text-muted-foreground">{item.kol?.followers || item.followers}</p>
                </div>
              </div>
              
              {item.matchScore && (
                <div className="mt-4">
                  <Badge variant="success">Match: {Math.round(item.matchScore * 100)}%</Badge>
                  <p className="text-xs mt-1">{item.reasoning}</p>
                </div>
              )}
              
              <div className="flex gap-2 mt-4">
                {(item.kol?.categories || item.categories)?.map((cat: string) => (
                  <Badge key={cat} variant="outline">{cat}</Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}