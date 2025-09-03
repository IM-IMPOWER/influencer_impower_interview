"use client"

import { useState } from 'react'
import { useQuery, useMutation } from '@apollo/client'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Badge, Table, TableBody, TableRow, TableCell } from '@/components/ui/ui-components'
import { CREATE_CAMPAIGN, GET_CAMPAIGNS } from '@/queries/campaign.graphql'
import { Campaign } from '@/types'
import { getStatusVariant } from '@/lib/utils'
import { Plus } from 'lucide-react'

export default function CampaignsPage() {
  const [showList, setShowList] = useState(false)
  const [newCampaign, setNewCampaign] = useState({ name: '', budget: '', brief: '' })
  
  const { data: campaignsData, loading } = useQuery(GET_CAMPAIGNS, {
    skip: !showList, // Only fetch after "Create Campaign" clicked
    pollInterval: 3000 // Dynamic updates
  })

  const [createCampaign] = useMutation(CREATE_CAMPAIGN, {
    onCompleted: () => {
      setShowList(true)
      setNewCampaign({ name: '', budget: '', brief: '' })
    }
  })

  const handleCreateCampaign = async () => {
    if (!newCampaign.name || !newCampaign.budget) return
    
    await createCampaign({
      variables: { input: newCampaign }
    })
  }

  return (
    <div className="space-y-6">
      {/* Campaign Creation Form */}
      <Card>
        <CardHeader>
          <CardTitle>Create New Campaign</CardTitle>
          <p className="text-sm text-muted-foreground">
            Click "Create Campaign" to see your campaign list dynamically
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Input
              placeholder="Campaign name"
              value={newCampaign.name}
              onChange={(e) => setNewCampaign(prev => ({ ...prev, name: e.target.value }))}
            />
            <Input
              placeholder="Budget (THB)"
              value={newCampaign.budget}
              onChange={(e) => setNewCampaign(prev => ({ ...prev, budget: e.target.value }))}
            />
            <Input
              placeholder="Brief (e.g., Office Worker)"
              value={newCampaign.brief}
              onChange={(e) => setNewCampaign(prev => ({ ...prev, brief: e.target.value }))}
            />
          </div>
          
          <Button 
            onClick={handleCreateCampaign}
            disabled={!newCampaign.name || !newCampaign.budget}
            className="w-full"
          >
            <Plus className="w-4 h-4 mr-2" />
            Create Campaign & Show List
          </Button>
        </CardContent>
      </Card>

      {/* Dynamic Campaign List - Only shows after creation */}
      {showList && (
        <Card>
          <CardHeader>
            <CardTitle>Your Campaigns ({campaignsData?.campaigns?.length || 0})</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center p-8">Loading campaigns...</div>
            ) : (
              <Table>
                <TableBody>
                  {campaignsData?.campaigns?.map((campaign: Campaign) => (
                    <TableRow key={campaign.id}>
                      <TableCell className="font-medium">{campaign.name}</TableCell>
                      <TableCell>
                        <Badge variant={getStatusVariant(campaign.status)}>
                          {campaign.status}
                        </Badge>
                      </TableCell>
                      <TableCell>{campaign.budget}</TableCell>
                      <TableCell>{campaign.kols} KOLs</TableCell>
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
            )}
          </CardContent>
        </Card>
      )}

      {!showList && (
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-muted-foreground">
              Create your first campaign to see the dynamic campaign list
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}