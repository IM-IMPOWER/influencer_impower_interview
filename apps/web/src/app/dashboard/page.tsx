"use client"

import { useEffect, useState } from 'react'
import { Card, CardContent, CardFooter, CardHeader, CardTitle, Badge, Table, TableBody, TableCell, TableRow } from '@/components/ui/ui-components'
import { apiClient } from '@/services/api'
import { KPIData, Campaign, Activity } from '@/types'
import { getStatusVariant, getRelativeTime } from '@/lib/utils'
import Link from 'next/link'
import { Skeleton } from '@/components/ui/skeleton'

export default function DashboardPage() {
  const [kpiData, setKpiData] = useState<KPIData[]>([])
  const [campaigns, setCampaigns] = useState<Campaign[]>([])
  const [activities, setActivities] = useState<Activity[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [kpiResponse, campaignsResponse, activitiesResponse] = await Promise.all([
          apiClient.getKPIData(),
          apiClient.getCampaigns(1, 4),
          apiClient.getRecentActivities()
        ])
        
        setKpiData(kpiResponse)
        setCampaigns(campaignsResponse.data)
        setActivities(activitiesResponse)
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) {
    return <DashboardSkeleton />
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {kpiData.map((item) => (
          <Card key={item.title}>
            <CardHeader>
              <CardTitle className="text-sm font-medium">{item.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{item.value}</div>
            </CardContent>
            <CardFooter>
              <span className={item.changeType === 'increase' ? 'text-green-600' : 'text-red-600'}>
                {item.change}
              </span>
              <span className="ml-1 text-sm text-muted-foreground">from last period</span>
            </CardFooter>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Active Campaigns</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableBody>
                {campaigns.map((campaign) => (
                  <TableRow key={campaign.id} className="cursor-pointer">
                    <TableCell className="font-medium">
                      <Link href={`/campaigns/${campaign.id}`} className="hover:underline">
                        {campaign.name}
                      </Link>
                    </TableCell>
                    <TableCell>
                      <Badge variant={getStatusVariant(campaign.status)}>{campaign.status}</Badge>
                    </TableCell>
                    <TableCell className="text-center">{campaign.kols}</TableCell>
                    <TableCell>
                      <div className="w-full bg-secondary rounded-full h-2">
                        <div 
                          className="bg-primary h-2 rounded-full" 
                          style={{ width: `${campaign.progress}%` }}
                        />
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {activities.map((activity) => (
                <div key={activity.id} className="flex items-start space-x-3">
                  <div className="flex-shrink-0 w-2 h-2 mt-2 bg-primary rounded-full" />
                  <div>
                    <p className="text-sm">{activity.text}</p>
                    <p className="text-xs text-muted-foreground">{getRelativeTime(activity.time)}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i}>
            <CardHeader>
              <Skeleton className="h-4 w-24" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-8 w-16" />
            </CardContent>
            <CardFooter>
              <Skeleton className="h-4 w-32" />
            </CardFooter>
          </Card>
        ))}
      </div>
      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <Skeleton className="h-6 w-32" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-48 w-full" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-28" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-32 w-full" />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}