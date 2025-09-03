import { 
  KPIData, 
  Campaign, 
  Activity, 
  KOL, 
  Content, 
  ApiResponse, 
  PaginatedResponse, 
  FilterOptions,
  BudgetOptimizationRequest,
  BudgetOptimizationResponse 
} from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;
const KOL_SERVICE_URL = process.env.NEXT_PUBLIC_KOL_SERVICE_URL;
const CAMPAIGN_SERVICE_URL = process.env.NEXT_PUBLIC_CAMPAIGN_SERVICE_URL;
const ANALYTICS_SERVICE_URL = process.env.NEXT_PUBLIC_ANALYTICS_SERVICE_URL;
const CONTENT_SERVICE_URL = process.env.NEXT_PUBLIC_CONTENT_SERVICE_URL;

class ApiClient {
  private async request<T>(url: string, options?: RequestInit): Promise<ApiResponse<T>> {
    const token = localStorage.getItem('auth_token');
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // Dashboard APIs
  async getKPIData(): Promise<KPIData[]> {
    const response = await this.request<KPIData[]>(`${ANALYTICS_SERVICE_URL}/dashboard/kpi`);
    return response.data;
  }

  async getRecentActivities(): Promise<Activity[]> {
    const response = await this.request<Activity[]>(`${API_BASE_URL}/activities/recent`);
    return response.data;
  }

  // Campaign APIs
  async getCampaigns(page = 1, limit = 10): Promise<PaginatedResponse<Campaign>> {
    const response = await this.request<PaginatedResponse<Campaign>>(
      `${CAMPAIGN_SERVICE_URL}/campaigns?page=${page}&limit=${limit}`
    );
    return response.data;
  }

  async getCampaignById(id: string): Promise<Campaign> {
    const response = await this.request<Campaign>(`${CAMPAIGN_SERVICE_URL}/campaigns/${id}`);
    return response.data;
  }

  async createCampaign(campaign: Partial<Campaign>): Promise<Campaign> {
    const response = await this.request<Campaign>(`${CAMPAIGN_SERVICE_URL}/campaigns`, {
      method: 'POST',
      body: JSON.stringify(campaign),
    });
    return response.data;
  }

  async updateCampaign(id: string, campaign: Partial<Campaign>): Promise<Campaign> {
    const response = await this.request<Campaign>(`${CAMPAIGN_SERVICE_URL}/campaigns/${id}`, {
      method: 'PUT',
      body: JSON.stringify(campaign),
    });
    return response.data;
  }

  // KOL APIs
  async getKOLs(filters: FilterOptions = {}, page = 1, limit = 20): Promise<PaginatedResponse<KOL>> {
    const params = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
      ...Object.fromEntries(
        Object.entries(filters).map(([key, value]) => [key, Array.isArray(value) ? value.join(',') : value.toString()])
      ),
    });

    const response = await this.request<PaginatedResponse<KOL>>(
      `${KOL_SERVICE_URL}/kols?${params}`
    );
    return response.data;
  }

  async getKOLById(id: string): Promise<KOL> {
    const response = await this.request<KOL>(`${KOL_SERVICE_URL}/kols/${id}`);
    return response.data;
  }

  async getCampaignKOLs(campaignId: string): Promise<KOL[]> {
    const response = await this.request<KOL[]>(`${CAMPAIGN_SERVICE_URL}/campaigns/${campaignId}/kols`);
    return response.data;
  }

  async inviteKOLToCampaign(campaignId: string, kolId: string, terms: any): Promise<void> {
    await this.request(`${CAMPAIGN_SERVICE_URL}/campaigns/${campaignId}/invite`, {
      method: 'POST',
      body: JSON.stringify({ kolId, terms }),
    });
  }

  // Content APIs
  async getCampaignContent(campaignId: string): Promise<Content[]> {
    const response = await this.request<Content[]>(`${CONTENT_SERVICE_URL}/campaigns/${campaignId}/content`);
    return response.data;
  }

  async approveContent(contentId: string, feedback?: string): Promise<void> {
    await this.request(`${CONTENT_SERVICE_URL}/content/${contentId}/approve`, {
      method: 'POST',
      body: JSON.stringify({ feedback }),
    });
  }

  async requestContentChanges(contentId: string, feedback: string): Promise<void> {
    await this.request(`${CONTENT_SERVICE_URL}/content/${contentId}/request-changes`, {
      method: 'POST',
      body: JSON.stringify({ feedback }),
    });
  }

  // Budget Optimization
  async optimizeBudget(request: BudgetOptimizationRequest): Promise<BudgetOptimizationResponse> {
    const response = await this.request<BudgetOptimizationResponse>(
      `${ANALYTICS_SERVICE_URL}/budget/optimize`,
      {
        method: 'POST',
        body: JSON.stringify(request),
      }
    );
    return response.data;
  }

  // Analytics
  async getCampaignAnalytics(campaignId: string, dateRange?: { start: string; end: string }) {
    const params = dateRange ? `?start=${dateRange.start}&end=${dateRange.end}` : '';
    const response = await this.request(
      `${ANALYTICS_SERVICE_URL}/campaigns/${campaignId}/analytics${params}`
    );
    return response.data;
  }

  async getKOLAnalytics(kolId: string, campaignId?: string) {
    const params = campaignId ? `?campaignId=${campaignId}` : '';
    const response = await this.request(
      `${ANALYTICS_SERVICE_URL}/kols/${kolId}/analytics${params}`
    );
    return response.data;
  }
}

export const apiClient = new ApiClient();