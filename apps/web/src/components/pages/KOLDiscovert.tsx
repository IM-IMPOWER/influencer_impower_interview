import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Badge } from '../ui';
import { Filter } from '../icons';
import { kolDiscoveryData } from '../../data/mockData';
import { KOL } from '../../types';

interface KOLDiscoveryProps {
  navigate: (page: string, id?: string) => void;
  onAddToPlan: (kol: KOL) => void;
}

export const KOLDiscovery: React.FC<KOLDiscoveryProps> = ({ navigate, onAddToPlan }) => (
  <div className="flex flex-col lg:flex-row gap-6">
    <div className="lg:w-1/4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="w-5 h-5" /> Filters
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="text-sm font-medium">Brief</label>
            <input 
              type="text" 
              placeholder="e.g., Office Worker, Beauty" 
              className="mt-1 block w-full text-sm border-gray-300 rounded-md shadow-sm dark:bg-gray-800 dark:border-gray-700"
            />
          </div>
          <div>
            <label className="text-sm font-medium">Platform</label>
            <select className="mt-1 block w-full text-sm border-gray-300 rounded-md shadow-sm dark:bg-gray-800 dark:border-gray-700">
              <option>Any</option>
              <option>TikTok</option>
              <option>Instagram</option>
            </select>
          </div>
          <div>
            <label className="text-sm font-medium">Followers</label>
            <input type="range" className="w-full"/>
          </div>
          <button className="w-full px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700">
            Search
          </button>
        </CardContent>
      </Card>
    </div>
    <div className="flex-1">
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {kolDiscoveryData.map(kol => (
          <Card key={kol.id}>
            <CardContent className="pt-6">
              <div className="flex items-center space-x-4">
                <img 
                  src={`https://placehold.co/64x64/E2E8F0/4A5568?text=${kol.name.charAt(0)}`} 
                  alt={kol.name} 
                  className="w-16 h-16 rounded-full" 
                />
                <div>
                  <p className="font-semibold">{kol.name}</p>
                  <p className="text-sm text-blue-500">{kol.handle}</p>
                </div>
              </div>
              <div className="flex justify-around text-center my-4 border-y dark:border-gray-800 py-3">
                <div>
                  <p className="font-bold text-lg">{kol.followers}</p>
                  <p className="text-xs text-gray-500 uppercase">Followers</p>
                </div>
                <div>
                  <p className="font-bold text-lg">{kol.engagement}</p>
                  <p className="text-xs text-gray-500 uppercase">Engagement</p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                {kol.categories.map(cat => <Badge key={cat}>{cat}</Badge>)}
              </div>
              <div className="flex gap-2 mt-4">
                <button 
                  onClick={() => navigate('KOLProfile', kol.id)} 
                  className="flex-1 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700"
                >
                  View Profile
                </button>
                <button 
                  onClick={() => onAddToPlan(kol)} 
                  className="flex-1 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
                >
                  Add to Plan
                </button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  </div>
);