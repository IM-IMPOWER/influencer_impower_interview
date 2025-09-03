import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui';

export const BudgetOptimizer: React.FC = () => (
  <Card>
    <CardHeader>
      <CardTitle>Budget Optimizer</CardTitle>
      <p className="text-sm text-gray-500 mt-1">Define your constraints to get a suggested KOL plan.</p>
    </CardHeader>
    <CardContent className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="text-sm font-medium">Total Budget ($)</label>
          <input 
            type="number" 
            placeholder="50000" 
            className="mt-1 block w-full text-sm border-gray-300 rounded-md shadow-sm dark:bg-gray-800 dark:border-gray-700"
          />
        </div>
        <div>
          <label className="text-sm font-medium">Minimum Total Reach</label>
          <input 
            type="number" 
            placeholder="10000000" 
            className="mt-1 block w-full text-sm border-gray-300 rounded-md shadow-sm dark:bg-gray-800 dark:border-gray-700"
          />
        </div>
      </div>
      <div>
        <label className="text-sm font-medium">Category Mix</label>
        <p className="text-xs text-gray-500">Allocate percentage for each category.</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-2">
          <div>
            <label className="text-xs">Fashion</label>
            <input 
              type="number" 
              placeholder="40" 
              className="mt-1 block w-full text-sm border-gray-300 rounded-md"
            />
          </div>
          <div>
            <label className="text-xs">Tech</label>
            <input 
              type="number" 
              placeholder="30" 
              className="mt-1 block w-full text-sm border-gray-300 rounded-md"
            />
          </div>
          <div>
            <label className="text-xs">Food</label>
            <input 
              type="number" 
              placeholder="20" 
              className="mt-1 block w-full text-sm border-gray-300 rounded-md"
            />
          </div>
          <div>
            <label className="text-xs">Travel</label>
            <input 
              type="number" 
              placeholder="10" 
              className="mt-1 block w-full text-sm border-gray-300 rounded-md"
            />
          </div>
        </div>
      </div>
      <div className="text-right">
        <button className="px-6 py-2 font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700">
          Generate Plan
        </button>
      </div>
    </CardContent>
  </Card>
);