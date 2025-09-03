"use client"

import React, { useState } from 'react'
import { useQuery, useMutation } from '@apollo/client'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Skeleton } from '@/components/ui/skeleton'
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger,
  DropdownMenuSeparator
} from '@/components/ui/dropdown-menu'
import { GET_ALL_CAMPAIGNS, CREATE_CAMPAIGN, DELETE_CAMPAIGN } from '@/lib/graphql/campaign.graphql'
import { 
  Search, 
  Filter, 
  Plus, 
  MoreVertical, 
  Edit, 
  Trash2, 
  Target, 
  Users, 
  DollarSign,
  TrendingUp
} from 'lucide-react'
import { toast } from 'sonner'

// AIDEV-NOTE: 250903170011 - Campaigns component migrated to Apollo GraphQL with CRUD operations

interface CampaignsProps {
  navigate?: (page: string, id?: string) => void
}

const getStatusVariant = (status: string) => {
  switch (status?.toLowerCase()) {
    case 'active': return 'default'
    case 'completed': return 'success' 
    case 'planning': return 'warning'
    case 'pending': return 'secondary'
    default: return 'outline'
  }
}

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  }).format(amount)
}

export const Campaigns: React.FC<CampaignsProps> = ({ navigate }) => {
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  
  // GraphQL queries
  const { data, loading, error, refetch } = useQuery(GET_ALL_CAMPAIGNS, {
    variables: {
      filters: statusFilter !== 'all' ? { status: statusFilter } : undefined,
      pagination: { limit: 20, offset: 0 }
    },
    errorPolicy: 'all',
    pollInterval: 60000 // Refresh every minute
  })

  // GraphQL mutations
  const [deleteCampaign] = useMutation(DELETE_CAMPAIGN, {
    onCompleted: () => {
      toast.success('Campaign deleted successfully')
      refetch()
    },
    onError: (error) => {
      toast.error(`Failed to delete campaign: ${error.message}`)
    }
  })

  const [createCampaign] = useMutation(CREATE_CAMPAIGN, {
    onCompleted: (data) => {
      toast.success('Campaign created successfully')
      navigate?.('CampaignDetail', data.createCampaign.id)
    },
    onError: (error) => {
      toast.error(`Failed to create campaign: ${error.message}`)
    }
  })

  const handleDelete = async (campaignId: string, campaignName: string) => {
    if (window.confirm(`Are you sure you want to delete "${campaignName}"? This action cannot be undone.`)) {
      try {
        await deleteCampaign({ variables: { id: campaignId } })
      } catch (error) {
        // Error handled by onError callback
      }
    }
  }

  const handleCreateCampaign = async () => {
    try {
      await createCampaign({
        variables: {
          input: {
            name: 'New Campaign',
            brief: 'Campaign description',
            budget: 10000,
            status: 'Planning'
          }
        }
      })
    } catch (error) {
      // Error handled by onError callback
    }
  }

  // Filter campaigns by search term
  const campaigns = data?.campaigns?.campaigns || []
  const filteredCampaigns = campaigns.filter((campaign: any) => 
    campaign.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    campaign.brief?.toLowerCase().includes(searchTerm.toLowerCase())
  )

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] text-center">
        <div className="text-destructive mb-4">Failed to load campaigns</div>
        <Button onClick={() => refetch()} variant="outline">
          Try Again
        </Button>
      </div>
    )
  }

  return (
    <Card>
      <CardHeader className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            All Campaigns
          </CardTitle>
          <p className="text-sm text-muted-foreground mt-1">
            Manage, track, and analyze all your campaigns.
          </p>
        </div>
        
        <div className="flex flex-col sm:flex-row gap-2 w-full md:w-auto">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search campaigns..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 w-full md:w-64"
            />
          </div>
          
          {/* Status Filter */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="flex items-center gap-2">
                <Filter className="h-4 w-4" />
                {statusFilter === 'all' ? 'All' : statusFilter}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuItem onClick={() => setStatusFilter('all')}>
                All Status
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => setStatusFilter('Active')}>
                Active
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setStatusFilter('Planning')}>
                Planning
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setStatusFilter('Completed')}>
                Completed
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setStatusFilter('Pending')}>
                Pending
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          
          {/* Create Campaign Button */}
          <Button onClick={handleCreateCampaign} className="flex items-center gap-2">
            <Plus className="h-4 w-4" />
            Create Campaign
          </Button>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="overflow-x-auto border rounded-lg">
          <table className="w-full text-sm">
            <thead className="text-xs uppercase bg-muted/50">
              <tr className="text-left">
                <th className="p-4">Campaign</th>
                <th className="p-4">Status</th>
                <th className="p-4">Budget</th>
                <th className="p-4">KOLs</th>
                <th className="p-4">Progress</th>
                <th className="p-4">Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                Array(5).fill(0).map((_, i) => (
                  <tr key={i} className="border-b">
                    <td className="p-4"><Skeleton className="h-4 w-32" /></td>
                    <td className="p-4"><Skeleton className="h-6 w-16" /></td>
                    <td className="p-4"><Skeleton className="h-4 w-20" /></td>
                    <td className="p-4"><Skeleton className="h-4 w-8" /></td>
                    <td className="p-4"><Skeleton className="h-2 w-24" /></td>
                    <td className="p-4"><Skeleton className="h-4 w-16" /></td>
                  </tr>
                ))
              ) : filteredCampaigns.length > 0 ? (
                filteredCampaigns.map((campaign: any) => (
                  <tr key={campaign.id} className="border-b hover:bg-muted/25 transition-colors">
                    <td className="p-4">
                      <div className="font-medium">{campaign.name}</div>
                      {campaign.brief && (
                        <div className="text-xs text-muted-foreground mt-1 max-w-xs truncate">
                          {campaign.brief}
                        </div>
                      )}
                    </td>
                    <td className="p-4">
                      <Badge variant={getStatusVariant(campaign.status)}>
                        {campaign.status}
                      </Badge>
                    </td>
                    <td className="p-4">
                      {formatCurrency(campaign.budget || 0)}
                    </td>
                    <td className="p-4 text-center">
                      <span className="flex items-center gap-1">
                        <Users className="h-3 w-3" />
                        {campaign.kols || 0}
                      </span>
                    </td>
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        <div className="w-16 bg-secondary rounded-full h-2">
                          <div 
                            className="bg-primary h-2 rounded-full transition-all" 
                            style={{ width: `${campaign.progress || 0}%` }}
                          />
                        </div>
                        <span className="text-xs whitespace-nowrap">
                          {campaign.progress || 0}%
                        </span>
                      </div>
                    </td>
                    <td className="p-4">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="sm">
                            <MoreVertical className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem 
                            onClick={() => navigate?.('CampaignDetail', campaign.id)}
                            className="flex items-center gap-2"
                          >
                            <Edit className="h-4 w-4" />
                            View Details
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem 
                            onClick={() => handleDelete(campaign.id, campaign.name)}
                            className="flex items-center gap-2 text-destructive focus:text-destructive"
                          >
                            <Trash2 className="h-4 w-4" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={6} className="p-8 text-center">
                    <div className="flex flex-col items-center">
                      <Target className="h-16 w-16 text-muted-foreground opacity-50 mb-4" />
                      <h3 className="text-lg font-medium mb-2">
                        {searchTerm || statusFilter !== 'all' ? 'No campaigns found' : 'No campaigns yet'}
                      </h3>
                      <p className="text-muted-foreground mb-4">
                        {searchTerm || statusFilter !== 'all' 
                          ? 'Try adjusting your search or filters'
                          : 'Create your first campaign to get started'
                        }
                      </p>
                      {(!searchTerm && statusFilter === 'all') && (
                        <Button onClick={handleCreateCampaign} className="flex items-center gap-2">
                          <Plus className="h-4 w-4" />
                          Create Your First Campaign
                        </Button>
                      )}
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}