import React, { useState } from 'react';
import { Card, CardContent, Badge } from '../ui';
import { ArrowLeft } from '../icons';
import { kolDiscoveryData } from '../../data/mockData';
import { getStatusVariant } from '../../utils/index.ts';
import { KOL } from '../../types';

interface KOLProfileProps {
  kolId: string;
  navigate: (page: string, id?: string) => void;
  onAddToPlan: (kol: KOL) => void;
}

export const KOLProfile: React.FC<KOLProfileProps> = ({ kolId, navigate, onAddToPlan }) => {
  const kol = kolDiscoveryData.find(k => k.id === kolId);
  const [activeTab, setActiveTab] = useState<'Outreach' | 'Campaign History' | 'Analytics'>('Outreach');

  if (!kol) return <div>KOL not found</div>;

  return (
    <div className="space-y-6">
      <button 
        onClick={() => navigate('KOL Discovery')} 
        className="flex items-center gap-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
      >
        <ArrowLeft className="w-4 h-4" /> Back to Discovery
      </button>
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col md:flex-row items-center gap-6">
            <img 
              src={`https://placehold.co/100x100/E2E8F0/4A5568?text=${kol.name.charAt(0)}`} 
              className="w-24 h-24 rounded-full"
            />
            <div className="flex-1 text-center md:text-left">
              <h1 className="text-3xl font-bold">{kol.name}</h1>
              <p className="text-blue-500">{kol.handle}</p>
            </div>
            <div className="flex gap-2">
              <button className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700">
                Invite to Campaign
              </button>
              <button 
                onClick={() => onAddToPlan(kol)} 
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
              >
                Add to Plan
              </button>
            </div>
          </div>
        </CardContent>
      </Card>
      <Card>
        <div className="border-b dark:border-gray-800">
          <div className="flex space-x-4 px-6">
            {(['Outreach', 'Campaign History', 'Analytics'] as const).map(tab => (
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
          {activeTab === 'Outreach' && (
            <div className="space-y-4">
              <p className="font-semibold">Conversation History</p>
              <div className="border rounded-lg p-4 space-y-2 text-sm h-48 overflow-y-auto bg-gray-50 dark:bg-gray-800/50">
                <p><Badge variant="warning">Contacted</Badge> Sent initial outreach via TikTok DM. (2 days ago)</p>
                <p><Badge variant="info">Negotiating</Badge> Replied with rate card. (1 day ago)</p>
              </div>
              <textarea 
                placeholder="Write a message or use a template..." 
                className="w-full text-sm border-gray-300 rounded-md shadow-sm dark:bg-gray-800 dark:border-gray-700" 
                rows={4}
              ></textarea>
              <div className="flex justify-end gap-2">
                <button className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200">
                  Use Template
                </button>
                <button className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700">
                  Send Message
                </button>
              </div>
            </div>
          )}
          {activeTab === 'Campaign History' && (
            <table className="w-full text-sm">
              <tbody>
                {kol.campaignHistory.map(c => (
                  <tr key={c.id} className="border-b dark:border-gray-800">
                    <td className="p-4 font-medium">{c.name}</td>
                    <td className="p-4">
                      <Badge variant={getStatusVariant(c.status)}>{c.status}</Badge>
                    </td>
                    <td className="p-4 text-right">
                      <button 
                        onClick={() => navigate('CampaignDetail', c.id)} 
                        className="font-medium text-blue-600 hover:underline"
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
                {kol.campaignHistory.length === 0 && (
                  <tr>
                    <td colSpan={3} className="p-4 text-center text-gray-500">
                      No campaign history.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          )}
          {activeTab === 'Analytics' && (
            <div className="text-center py-12 text-gray-500">
              KOL performance analytics will be displayed here.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};