"use client"

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Badge, Button, Input, Textarea } from '@/components/ui/ui-components'
import { useKOLMatching, useKOLPlan } from '@/hooks/useKOLs'
import { useDebounce } from '@/hooks/useDebounce'

export default function KOLMatchingPage() {
  const [brief, setBrief] = useState('')
  const [constraints, setConstraints] = useState({
    budget: '',
    minReach: '',
    tierMix: ''
  })
  
  const debouncedBrief = useDebounce(brief, 800)
  const { matches, loading, hasResults } = useKOLMatching(debouncedBrief)
  const { addToPlan, planCount } = useKOLPlan()

  return (
    <div className="space-y-6">
      {/* PoC2: Primary ML Matching Interface */}
      <Card>
        <CardHeader>
          <CardTitle>AI KOL Matching (PoC2 - Priority 1)</CardTitle>
          <p className="text-sm text-muted-foreground">
            Describe your target audience. ML will find matching KOLs with reasoning.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="Example: 'Office Worker', 'University students who cook in condo', 'Beauty enthusiasts in Bangkok'"
            value={brief}
            onChange={(e) => setBrief(e.target.value)}
            rows={3}
          />
          
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium">Budget (THB)</label>
              <Input 
                placeholder="500,000" 
                value={constraints.budget}
                onChange={(e) => setConstraints(prev => ({ ...prev, budget: e.target.value }))}
              />
            </div>
            <div>
              <label className="text-sm font-medium">Min Reach</label>
              <Input 
                placeholder="1M followers" 
                value={constraints.minReach}
                onChange={(e) => setConstraints(prev => ({ ...prev, minReach: e.target.value }))}
              />
            </div>
            <div>
              <label className="text-sm font-medium">Tier Mix</label>
              <Input 
                placeholder="1 macro, 10 micro" 
                value={constraints.tierMix}
                onChange={(e) => setConstraints(prev => ({ ...prev, tierMix: e.target.value }))}
              />
            </div>
          </div>

          {loading && (
            <div className="flex items-center gap-2 text-blue-600">
              <div className="animate-spin w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
              ML processing your brief...
            </div>
          )}
        </CardContent>
      </Card>

      {/* Dynamic ML Results */}
      {hasResults && (
        <Card>
          <CardHeader>
            <CardTitle>ML Matched KOLs ({matches.length} found)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {matches.map((match: any) => (
                <Card key={match.kol.id} className="border-l-4 border-l-green-500">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      <img 
                        src={`https://placehold.co/48x48/E2E8F0/4A5568?text=${match.kol.handle.charAt(1)}`}
                        className="w-12 h-12 rounded-full" 
                      />
                      <div>
                        <p className="font-semibold">{match.kol.handle}</p>
                        <p className="text-sm text-muted-foreground">{match.kol.followers}</p>
                      </div>
                    </div>
                    
                    <div className="mt-3">
                      <Badge variant="success">
                        {Math.round(match.matchScore * 100)}% Match
                      </Badge>
                      <Badge variant="outline" className="ml-2">
                        {Math.round(match.confidence * 100)}% Confidence
                      </Badge>
                    </div>
                    
                    <p className="text-xs mt-2 text-gray-600 bg-gray-50 p-2 rounded">
                      <strong>AI Reasoning:</strong> {match.reasoning}
                    </p>
                    
                    <div className="flex gap-2 mt-3">
                      {match.kol.categories?.map((cat: string) => (
                        <Badge key={cat} variant="outline">{cat}</Badge>
                      ))}
                    </div>
                    
                    <Button 
                      onClick={() => addToPlan(match.kol)}
                      className="w-full mt-3"
                      size="sm"
                    >
                      Add to Plan
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {brief && !loading && !hasResults && (
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-muted-foreground">
              No matching KOLs found for "{brief}". Try refining your brief.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}