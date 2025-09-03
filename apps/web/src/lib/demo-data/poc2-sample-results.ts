// AIDEV-NOTE: 250903170000 - Sample data for POC2 testing and demonstration purposes

import type { KOLData } from "@/components/kol/kol-card";

export interface POC2DemoResults {
  extractedRequirements: {
    targetKolTiers: string[];
    targetCategories: string[];
    totalBudget: number;
    minFollowerCount: number;
    maxFollowerCount: number;
    minEngagementRate: number;
    targetDemographics: string;
    targetLocations: string[];
    targetLanguages: string[];
    requireBrandSafe: boolean;
    requireVerified: boolean;
    campaignObjective: string;
    contentTheme: string;
    keywords: string[];
    excludedKeywords: string[];
    deliverables: string[];
    timeline: string;
    specialRequirements: string[];
  };
  matchedKols: KOLData[];
  briefAnalysis: {
    extractedThemes: string[];
    brandTone: string;
    targetAudience: string;
    contentRequirements: string[];
    budgetInsights: string[];
    timelineAnalysis: string;
    feasibilityScore: number;
    complexityLevel: string;
  };
  totalMatches: number;
  processingTimeSeconds: number;
  briefParsingConfidence: number;
  matchingAlgorithm: string;
  dataQualityWarnings: string[];
}

export const samplePOC2Results: POC2DemoResults = {
  extractedRequirements: {
    targetKolTiers: ["Micro", "Mid-tier"],
    targetCategories: ["Beauty & Skincare", "Lifestyle & Wellness", "Sustainability & Environment"],
    totalBudget: 25000,
    minFollowerCount: 10000,
    maxFollowerCount: 500000,
    minEngagementRate: 0.03,
    targetDemographics: "Women aged 25-40, income $40K-$100K, interested in sustainable living",
    targetLocations: ["United States", "Canada", "UK", "Australia"],
    targetLanguages: ["English"],
    requireBrandSafe: true,
    requireVerified: false,
    campaignObjective: "Drive awareness and sales for sustainable skincare line",
    contentTheme: "Clean beauty meets effective results",
    keywords: ["sustainable", "clean beauty", "eco-friendly", "skincare", "organic"],
    excludedKeywords: ["competing brands", "controversial", "adult content"],
    deliverables: ["feed posts", "stories", "unboxing videos", "tutorials"],
    timeline: "6 weeks",
    specialRequirements: ["Authentic product usage", "Educational content", "FTC disclosure"]
  },
  matchedKols: [
    {
      id: "kol_001",
      username: "cleanbeautyemma",
      displayName: "Emma Rodriguez",
      platform: "Instagram",
      tier: "Mid-tier",
      primaryCategory: "Beauty & Skincare",
      isVerified: true,
      isBrandSafe: true,
      profileImageUrl: "https://images.unsplash.com/photo-1494790108755-2616b332e044",
      followerCount: 145000,
      engagementRate: 0.047,
      averageViews: 8500,
      ratePerPost: 2800,
      score: {
        overallScore: 0.89,
        scoreComponents: {
          roiScore: 0.92,
          audienceQualityScore: 0.88,
          brandSafetyScore: 0.95,
          contentRelevanceScore: 0.91,
          demographicFitScore: 0.87,
          reliabilityScore: 0.85,
          overallConfidence: 0.90
        },
        semanticMatching: {
          similarityScore: 0.94,
          contentMatchScore: 0.89,
          brandAffinityScore: 0.92,
          matchedContentCategories: ["Clean Beauty", "Sustainable Living", "Skincare Reviews"],
          semanticKeywords: ["clean beauty", "sustainable skincare", "eco-friendly", "organic", "natural ingredients"]
        },
        performancePrediction: {
          predictedReach: 125000,
          predictedEngagement: 5875,
          predictedConversions: 235,
          predictedRoi: 3.8,
          predictionConfidence: 0.87
        }
      },
      matchReasons: ["High audience alignment", "Strong engagement rates", "Relevant content history"],
      fitExplanation: "Emma has established credibility in clean beauty with an engaged audience that matches your target demographic. Her content consistently promotes sustainable brands and she has a track record of driving conversions for eco-friendly products.",
      potentialConcerns: ["Premium pricing for her tier", "Limited availability in Q1"],
      recommendedApproach: "Offer exclusive early access to build authentic relationship and encourage detailed skincare routine integration."
    },
    {
      id: "kol_002",
      username: "eco_lifestyle_sarah",
      displayName: "Sarah Chen",
      platform: "Instagram",
      tier: "Micro",
      primaryCategory: "Lifestyle & Wellness",
      isVerified: false,
      isBrandSafe: true,
      profileImageUrl: "https://images.unsplash.com/photo-1438761681033-6461ffad8d80",
      followerCount: 68000,
      engagementRate: 0.065,
      averageViews: 4200,
      ratePerPost: 1200,
      score: {
        overallScore: 0.84,
        scoreComponents: {
          roiScore: 0.88,
          audienceQualityScore: 0.91,
          brandSafetyScore: 0.89,
          contentRelevanceScore: 0.82,
          demographicFitScore: 0.85,
          reliabilityScore: 0.79,
          overallConfidence: 0.84
        },
        semanticMatching: {
          similarityScore: 0.86,
          contentMatchScore: 0.79,
          brandAffinityScore: 0.88,
          matchedContentCategories: ["Sustainable Living", "Wellness", "Green Beauty"],
          semanticKeywords: ["eco-friendly", "sustainable", "wellness", "green living", "conscious beauty"]
        },
        performancePrediction: {
          predictedReach: 58000,
          predictedEngagement: 3770,
          predictedConversions: 142,
          predictedRoi: 4.2,
          predictionConfidence: 0.82
        }
      },
      matchReasons: ["Excellent engagement rate", "Cost-effective", "Authentic sustainability focus"],
      fitExplanation: "Sarah's micro-influencer status delivers highly engaged, niche audience with genuine interest in sustainable products. Her authentic approach to eco-living content creates trust and drives action.",
      potentialConcerns: ["Smaller reach compared to macro influencers"],
      recommendedApproach: "Focus on long-term partnership and authentic product integration into her daily routine content."
    },
    {
      id: "kol_003",
      username: "greenbeautyguide",
      displayName: "Jessica Wong",
      platform: "YouTube",
      tier: "Mid-tier",
      primaryCategory: "Beauty & Skincare",
      isVerified: true,
      isBrandSafe: true,
      profileImageUrl: "https://images.unsplash.com/photo-1489424731084-a5d8b219a5bb",
      followerCount: 285000,
      engagementRate: 0.038,
      averageViews: 15600,
      ratePerPost: 4200,
      score: {
        overallScore: 0.81,
        scoreComponents: {
          roiScore: 0.78,
          audienceQualityScore: 0.85,
          brandSafetyScore: 0.92,
          contentRelevanceScore: 0.88,
          demographicFitScore: 0.79,
          reliabilityScore: 0.83,
          overallConfidence: 0.81
        },
        semanticMatching: {
          similarityScore: 0.91,
          contentMatchScore: 0.85,
          brandAffinityScore: 0.87,
          matchedContentCategories: ["Green Beauty", "Product Reviews", "Skincare Education"],
          semanticKeywords: ["green beauty", "clean skincare", "product reviews", "beauty education", "sustainable"]
        },
        performancePrediction: {
          predictedReach: 195000,
          predictedEngagement: 7410,
          predictedConversions: 312,
          predictedRoi: 2.9,
          predictionConfidence: 0.79
        }
      },
      matchReasons: ["Strong YouTube presence", "Educational content style", "High brand safety"],
      fitExplanation: "Jessica specializes in green beauty education with detailed product reviews. Her YouTube format allows for comprehensive product demonstrations and builds strong trust with viewers.",
      potentialConcerns: ["Higher cost per engagement", "Longer content production timeline"],
      recommendedApproach: "Provide comprehensive product information and allow creative freedom for authentic review format."
    },
    {
      id: "kol_004",
      username: "mindfulbeautyco",
      displayName: "Aaliyah Johnson",
      platform: "TikTok",
      tier: "Micro",
      primaryCategory: "Beauty & Skincare",
      isVerified: false,
      isBrandSafe: true,
      profileImageUrl: "https://images.unsplash.com/photo-1531123897727-8f129e1688ce",
      followerCount: 42000,
      engagementRate: 0.089,
      averageViews: 12000,
      ratePerPost: 800,
      score: {
        overallScore: 0.86,
        scoreComponents: {
          roiScore: 0.94,
          audienceQualityScore: 0.83,
          brandSafetyScore: 0.87,
          contentRelevanceScore: 0.85,
          demographicFitScore: 0.88,
          reliabilityScore: 0.81,
          overallConfidence: 0.86
        },
        semanticMatching: {
          similarityScore: 0.83,
          contentMatchScore: 0.88,
          brandAffinityScore: 0.81,
          matchedContentCategories: ["Skincare Routines", "Beauty Tips", "Wellness"],
          semanticKeywords: ["mindful beauty", "skincare routine", "self-care", "natural beauty", "wellness"]
        },
        performancePrediction: {
          predictedReach: 78000,
          predictedEngagement: 6940,
          predictedConversions: 198,
          predictedRoi: 5.1,
          predictionConfidence: 0.84
        }
      },
      matchReasons: ["Exceptional engagement rate", "TikTok viral potential", "Cost-effective"],
      fitExplanation: "Aaliyah's TikTok content focuses on mindful beauty practices with exceptional engagement rates. Her authentic approach to skincare resonates with younger demographics interested in sustainable products.",
      potentialConcerns: ["TikTok platform dependency", "Younger skewing audience"],
      recommendedApproach: "Leverage TikTok's viral nature with authentic before/after content and educational skincare tips."
    },
    {
      id: "kol_005",
      username: "sustainablestyle_maya",
      displayName: "Maya Patel",
      platform: "Instagram",
      tier: "Micro",
      primaryCategory: "Sustainability & Environment",
      isVerified: false,
      isBrandSafe: true,
      profileImageUrl: "https://images.unsplash.com/photo-1534528741775-53994a69daeb",
      followerCount: 33000,
      engagementRate: 0.072,
      averageViews: 2800,
      ratePerPost: 650,
      score: {
        overallScore: 0.77,
        scoreComponents: {
          roiScore: 0.85,
          audienceQualityScore: 0.89,
          brandSafetyScore: 0.91,
          contentRelevanceScore: 0.74,
          demographicFitScore: 0.73,
          reliabilityScore: 0.76,
          overallConfidence: 0.77
        },
        semanticMatching: {
          similarityScore: 0.79,
          contentMatchScore: 0.71,
          brandAffinityScore: 0.92,
          matchedContentCategories: ["Sustainable Living", "Zero Waste", "Eco Products"],
          semanticKeywords: ["sustainability", "zero waste", "eco-friendly", "conscious living", "green products"]
        },
        performancePrediction: {
          predictedReach: 28000,
          predictedEngagement: 2016,
          predictedConversions: 89,
          predictedRoi: 4.6,
          predictionConfidence: 0.75
        }
      },
      matchReasons: ["Strong sustainability focus", "Highly engaged niche audience", "Budget-friendly"],
      fitExplanation: "Maya's content is perfectly aligned with sustainability values and attracts conscious consumers who prioritize eco-friendly products. Her audience is highly engaged and conversion-focused.",
      potentialConcerns: ["Smaller overall reach", "Limited beauty-specific content"],
      recommendedApproach: "Position products within broader sustainable lifestyle context and emphasize eco-friendly packaging."
    }
  ],
  briefAnalysis: {
    extractedThemes: [
      "Sustainable Beauty",
      "Clean Skincare",
      "Environmental Consciousness",
      "Product Efficacy",
      "Natural Ingredients",
      "Eco-Friendly Packaging"
    ],
    brandTone: "Educational, authentic, environmentally conscious",
    targetAudience: "Environmentally conscious women aged 25-40 with disposable income",
    contentRequirements: [
      "Before/after product demonstrations",
      "Educational content about sustainability",
      "Authentic usage testimonials",
      "FTC compliant sponsored disclosure"
    ],
    budgetInsights: [
      "70% allocation to micro and mid-tier influencers maximizes reach",
      "Cost per engagement target of $0.50 is achievable",
      "Expected ROI of 300% is realistic based on similar campaigns"
    ],
    timelineAnalysis: "6-week timeline is adequate for comprehensive campaign execution",
    feasibilityScore: 0.88,
    complexityLevel: "Medium - requires authentic product testing period"
  },
  totalMatches: 47,
  processingTimeSeconds: 3.2,
  briefParsingConfidence: 0.91,
  matchingAlgorithm: "Semantic Vector Matching v2.1 with ROI Optimization",
  dataQualityWarnings: [
    "Limited recent engagement data for 3 KOLs",
    "Some audience demographic data is estimated",
    "Rate information is based on industry benchmarks for 2 KOLs"
  ]
};

// Sample extracted requirements for different types of campaigns
export const sampleRequirements = {
  techStartup: {
    targetKolTiers: ["Micro", "Mid-tier"],
    targetCategories: ["Technology", "Business", "Innovation"],
    totalBudget: 15000,
    campaignObjective: "Drive awareness for new SaaS platform among entrepreneurs",
    keywords: ["startup", "productivity", "business tools", "entrepreneur"],
  },
  fashionBrand: {
    targetKolTiers: ["Mid-tier", "Macro"],
    targetCategories: ["Fashion", "Style", "Lifestyle"],
    totalBudget: 45000,
    campaignObjective: "Launch new sustainable fashion collection",
    keywords: ["sustainable fashion", "ethical clothing", "style", "fashion"],
  },
  foodBrand: {
    targetKolTiers: ["Nano", "Micro"],
    targetCategories: ["Food", "Health", "Lifestyle"],
    totalBudget: 8000,
    campaignObjective: "Promote healthy snack alternatives to health-conscious consumers",
    keywords: ["healthy snacks", "nutrition", "wellness", "organic food"],
  }
};