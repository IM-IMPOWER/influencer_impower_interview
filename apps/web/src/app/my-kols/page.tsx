"use client"

import { Card, CardContent, CardHeader, CardTitle, Badge, Button } from '@/components/ui/ui-components'
import { useKOLPlan } from '@/hooks/useKOLs'
import { Trash2 } from 'lucide-react'

export default function MyKOLsPage() {
  const { planKOLs, removeFromPlan, clearPlan, planCount } = useKOLPlan()

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>My KOL Plan ({planCount})</CardTitle>
            <p className="text-sm text-muted-foreground">
              KOLs dynamically added from discovery and matching
            </p>
          </div>
          {planCount > 0 && (
            <Button onClick={clearPlan} variant="outline" size="sm">
              Clear All
            </Button>
          )}
        </CardHeader>
        <CardContent>
          {planCount === 0 ? (
            <div className="text-center p-8">
              <p className="text-muted-foreground">
                No KOLs in your plan yet. Add KOLs from discovery or matching.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {planKOLs.map(kol => (
                <Card key={kol.id} className="relative">
                  <Button
                    onClick={() => removeFromPlan(kol.id)}
                    className="absolute top-2 right-2 p-2 h-8 w-8"
                    variant="ghost"
                    size="sm"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                  
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      <img 
                        src={`https://placehold.co/48x48/E2E8F0/4A5568?text=${kol.handle?.charAt(1) || 'K'}`}
                        className="w-12 h-12 rounded-full" 
                      />
                      <div>
                        <p className="font-semibold">{kol.handle}</p>
                        <p className="text-sm text-muted-foreground">{kol.followers}</p>
                      </div>
                    </div>
                    
                    <div className="mt-3 flex justify-between text-sm">
                      <span>Engagement: {kol.engagement}</span>
                      {kol.rates?.post && (
                        <span>Post: ฿{kol.rates.post.toLocaleString()}</span>
                      )}
                    </div>
                    
                    <div className="flex flex-wrap gap-1 mt-2">
                      {kol.categories?.slice(0, 2).map(cat => (
                        <Badge key={cat} variant="outline" className="text-xs">
                          {cat}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Plan Summary */}
      {planCount > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Plan Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div>
                <p className="text-2xl font-bold">{planCount}</p>
                <p className="text-sm text-muted-foreground">Total KOLs</p>
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {planKOLs.reduce((sum, kol) => {
                    const followers = parseInt(kol.followers.replace(/[^0-9]/g, '')) || 0
                    return sum + followers
                  }, 0).toLocaleString()}
                </p>
                <p className="text-sm text-muted-foreground">Est. Reach</p>
              </div>
              <div>
                <p className="text-2xl font-bold">
                  ฿{planKOLs.reduce((sum, kol) => sum + (kol.rates?.post || 0), 0).toLocaleString()}
                </p>
                <p className="text-sm text-muted-foreground">Est. Cost</p>
              </div>
              <div>
                <Button className="w-full">Export Plan</Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}