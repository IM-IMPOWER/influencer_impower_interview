"use client"

import React from 'react'
import { useQuery } from '@apollo/client'
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { GET_DASHBOARD_DATA } from '@/lib/graphql/sophisticated-queries'
import { GET_ALL_CAMPAIGNS } from '@/lib/graphql/campaign.graphql'
import { Activity, TrendingUp, TrendingDown, Users, DollarSign, Target, BarChart3 } from 'lucide-react'
import { toast } from 'sonner'

// AIDEV-NOTE: 250903170010 - Dashboard component migrated to Apollo GraphQL with real-time data

interface DashboardProps {
  navigate?: (page: string, id?: string) => void
}

const getStatusVariant = (status: string) => {
  switch (status.toLowerCase()) {
    case 'active': return 'default'
    case 'completed': return 'success'
    case 'planning': return 'warning'
    case 'pending': return 'secondary'
    default: return 'outline'
  }
}

const getChangeIcon = (changeType: 'increase' | 'decrease') => {
  return changeType === 'increase' ? 
    <TrendingUp className="h-4 w-4 text-green-500" /> : 
    <TrendingDown className="h-4 w-4 text-red-500" />
}

const getKPIIcon = (title: string) => {
  switch (title.toLowerCase()) {
    case 'active campaigns': return <Target className="h-5 w-5" />
    case 'total kols': return <Users className="h-5 w-5" />
    case 'total spend': return <DollarSign className="h-5 w-5" />
    case 'avg roi': return <BarChart3 className="h-5 w-5" />
    default: return <Activity className="h-5 w-5" />
  }
}

export const Dashboard: React.FC<DashboardProps> = ({ navigate }) => {
  // Fetch dashboard data with GraphQL
  const { data: dashboardData, loading: dashboardLoading, error: dashboardError } = useQuery(GET_DASHBOARD_DATA, {
    variables: { timeRange: '30d' },
    pollInterval: 30000, // Refresh every 30 seconds
    errorPolicy: 'all'
  })

  // Fetch campaigns data
  const { data: campaignData, loading: campaignsLoading } = useQuery(GET_ALL_CAMPAIGNS, {
    variables: { 
      filters: { status: 'Active' },
      pagination: { limit: 4, offset: 0 }
    },
    errorPolicy: 'all'
  })

  // Handle errors with toast notifications
  React.useEffect(() => {
    if (dashboardError) {
      toast.error('Failed to load dashboard data')
    }
  }, [dashboardError])

  const kpis = dashboardData?.dashboard?.kpis
  const activities = dashboardData?.dashboard?.recentActivities || []
  const campaigns = campaignData?.campaigns?.campaigns || []

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {dashboardLoading ? (
          Array(4).fill(0).map((_, i) => (
            <Card key={i}>
              <CardHeader className="pb-3">
                <Skeleton className="h-4 w-24" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-8 w-16" />
              </CardContent>
              <CardFooter>
                <Skeleton className="h-4 w-32" />
              </CardFooter>
            </Card>
          ))
        ) : kpis ? (
          Object.entries(kpis).map(([key, value]) => {
            const title = key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())
            const formattedValue = typeof value === 'number' ? 
              (key.includes('Spend') ? `$${value.toLocaleString()}` : value.toLocaleString()) :
              value
            
            // Mock change data - in real implementation this would come from the API
            const mockChange = Math.random() > 0.5 ? 'increase' : 'decrease'
            const mockChangeValue = `${(Math.random() * 20).toFixed(1)}%`
            
            return (
              <Card key={key}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">{title}</CardTitle>
                  {getKPIIcon(title)}
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{formattedValue}</div>
                </CardContent>
                <CardFooter className="flex items-center text-xs text-muted-foreground">
                  {getChangeIcon(mockChange)}
                  <span className={`ml-1 ${mockChange === 'increase' ? 'text-green-600' : 'text-red-600'}`}>
                    {mockChangeValue}
                  </span>
                  <span className="ml-1">from last period</span>
                </CardFooter>
              </Card>
            )
          })
        ) : (
          <div className="col-span-4 text-center text-muted-foreground">
            No KPI data available
          </div>
        )}
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Active Campaigns */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Active Campaigns
            </CardTitle>
          </CardHeader>
          <CardContent>
            {campaignsLoading ? (
              <div className="space-y-4">
                {Array(3).fill(0).map((_, i) => (
                  <div key={i} className="flex items-center space-x-4">
                    <Skeleton className="h-4 flex-1" />
                    <Skeleton className="h-6 w-16" />
                    <Skeleton className="h-4 w-8" />
                    <Skeleton className="h-2 w-24" />
                  </div>
                ))}
              </div>
            ) : campaigns.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Campaign</th>
                      <th className="text-left p-2">Status</th>
                      <th className="text-center p-2">KOLs</th>
                      <th className="text-left p-2">Progress</th>
                    </tr>
                  </thead>
                  <tbody>
                    {campaigns.map((campaign: any) => (
                      <tr 
                        key={campaign.id} 
                        className="border-b hover:bg-muted/50 cursor-pointer transition-colors" 
                        onClick={() => navigate?.('CampaignDetail', campaign.id)}
                      >
                        <td className="p-2 font-medium">{campaign.name}</td>
                        <td className="p-2">
                          <Badge variant={getStatusVariant(campaign.status)}>
                            {campaign.status}
                          </Badge>
                        </td>
                        <td className="p-2 text-center">{campaign.kols || 0}</td>
                        <td className="p-2">
                          <div className="flex items-center gap-2">
                            <div className="w-full bg-secondary rounded-full h-2">
                              <div 
                                className="bg-primary h-2 rounded-full transition-all" 
                                style={{ width: `${campaign.progress || 0}%` }}
                              />
                            </div>
                            <span className="text-xs text-muted-foreground whitespace-nowrap">
                              {campaign.progress || 0}%
                            </span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Target className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No active campaigns found</p>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="mt-2"
                  onClick={() => navigate?.('Campaigns')}
                >
                  Create Campaign
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Recent Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            {dashboardLoading ? (
              <div className="space-y-4">
                {Array(4).fill(0).map((_, i) => (
                  <div key={i} className="flex items-start space-x-3">
                    <Skeleton className="h-3 w-3 rounded-full mt-1.5" />
                    <div className="flex-1 space-y-2">
                      <Skeleton className="h-3 w-full" />
                      <Skeleton className="h-3 w-16" />
                    </div>
                  </div>
                ))}
              </div>
            ) : activities.length > 0 ? (
              <ul className="space-y-4">
                {activities.map((activity: any) => (
                  <li key={activity.id || activity.text} className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-3 h-3 mt-1.5 bg-primary rounded-full" />
                    <div className="min-w-0 flex-1">
                      <p className="text-sm break-words">{activity.text}</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(activity.time).toLocaleString()}
                      </p>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No recent activity</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}