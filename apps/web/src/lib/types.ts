export interface KPIData {
  title: string;
  value: string;
  change: string;
  changeType: 'increase' | 'decrease';
}

export interface Campaign {
  id: string;
  name: string;
  status: 'Active' | 'Pending' | 'Completed' | 'Planning';
  kols: number;
  progress: number;
  budget: string;
  totalReach: string;
  avgEngagement: string;
  createdAt: string;
  updatedAt: string;
}

export interface Activity {
  id: string;
  text: string;
  time: string;
  type: 'content_approval' | 'offer_accepted' | 'post_published';
}

export interface KOL {
  id: string;
  name: string;
  handle: string;
  avatar: string;
  followers: string;
  engagement: string;
  categories: string[];
  campaignHistory: Campaign[];
  email?: string;
  phone?: string;
  location?: string;
  rates?: {
    post: number;
    story: number;
    reel: number;
  };
}

export interface Content {
  id: string;
  kol: KOL;
  campaignId: string;
  thumbnail: string;
  status: 'Pending' | 'Approved' | 'Changes Requested';
  contentType: 'post' | 'story' | 'reel' | 'video';
  scheduledDate?: string;
  approvedAt?: string;
  feedback?: string;
}

export interface ViewState {
  page: string;
  id: string | null;
}

export interface ApiResponse<T> {
  data: T;
  message: string;
  success: boolean;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  totalPages: number;
}

export interface FilterOptions {
  categories?: string[];
  minFollowers?: number;
  maxFollowers?: number;
  minEngagement?: number;
  maxEngagement?: number;
  location?: string;
  platform?: string[];
}

export interface BudgetOptimizationRequest {
  totalBudget: number;
  minTotalReach: number;
  categoryMix: Record<string, number>;
  maxKOLs?: number;
  targetDemographics?: string[];
}

export interface BudgetOptimizationResponse {
  recommendedKOLs: KOL[];
  totalCost: number;
  projectedReach: number;
  projectedEngagement: number;
  categoryBreakdown: Record<string, number>;
}


export type StatusVariant = 'default' | 'success' | 'warning' | 'danger';

export type BadgeVariant = 'default' | 'success' | 'warning' | 'danger' | 'info';

export interface IconProps {
  className?: string;
}