import React, { useState } from 'react';
import { Card, CardContent, Badge } from '../ui';
import { ArrowLeft } from '../icons';
import { allCampaigns, campaignKOLs, contentForApproval } from '../../data/mockData';
import { getStatusVariant } from '../../utils';

interface CampaignDetailProps {
  campaignId: string;
  navigate: (page: string, id?: string) => void;
}

export const CampaignDetail: React.FC<CampaignDetailProps> = ({ campaignId, navigate }) => {
  const campaign = allCampaigns.find(c => c.id === campaignId);
  const kols = campaignKOLs[campaignId] || [];
  const contents = contentForApproval[campaignId] || [];
  const [activeTab, setActiveTab] = useState<'KOLs' | 'Content' | 'Analytics'>('KOLs');

  if (!campaign) return <div>Campaign not found</div>;

  return (
    <div className="space-y-6">
      <button 
        onClick={() => navigate('Campaigns')} 
        className="flex items-center gap-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
      >
        <ArrowLeft className="w-4 h-4" /> Back to Campaigns
      </button>
      <div className="flex flex-col md:flex-row justify-between items-start gap-4">
        <div>
          <h1 className="text-3xl font-bold">{campaign.name}</h1>
          <Badge variant={getStatusVariant(campaign.status)}>{campaign.status}</Badge>
        </div>
        <div className="grid grid-cols-2 gap-4 text-right">
          <div className="font-semibold">Budget</div>
          <div>{campaign.budget}</div>
          <div className="font-semibold">Total Reach</div>
          <div>{campaign.totalReach}</div>
        </div>
      </div>
      
      <Card>
        <div className="border-b dark:border-gray-800">
          <div className="flex space-x-4 px-6">
            {(['KOLs', 'Content', 'Analytics'] as const).map(tab => (
              <button 
                key={tab} 
                onClick={() => setActiveTab(tab)} 
                className={`py-3 px-1 border-b-2 font-medium ${
                  activeTab === tab 
                    ? 'border-blue-500 text-blue-600' 
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>
        <CardContent className="pt-6">
          {activeTab === 'KOLs' && (
            <table className="w-full text-sm">
              <tbody>
                {kols.map(kol => (
                  <tr key={kol.id} className="border-b dark:border-gray-800">
                    <td className="p-4 flex items-center gap-3">
                      <img 
                        src={`https://placehold.co/40x40/E2E8F0/4A5568?text=${kol.name.charAt(0)}`} 
                        className="w-10 h-10 rounded-full"
                      />
                      <div>
                        <p className="font-medium">{kol.name}</p>
                        <p className="text-xs text-gray-500">{kol.handle}</p>
                      </div>
                    </td>
                    <td className="p-4 text-center">{kol.followers}</td>
                    <td className="p-4 text-center">{kol.engagement}</td>
                    <td className="p-4 text-right">
                      <button 
                        className="font-medium text-blue-600 hover:underline" 
                        onClick={() => navigate('KOLProfile', kol.id)}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {activeTab === 'Content' && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {contents.map(c => (
                <Card key={c.id}>
                  <img src={c.thumbnail} alt="content" className="rounded-t-lg" />
                  <CardContent className="pt-4">
                    <p className="font-semibold">{c.kol.name}</p>
                    <Badge variant={
                      c.status === 'Approved' ? 'success' : 
                      (c.status === 'Pending' ? 'warning' : 'danger')
                    }>
                      {c.status}
                    </Badge>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
          {activeTab === 'Analytics' && (
            <div className="text-center py-12 text-gray-500">
              Analytics charts will be displayed here.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};