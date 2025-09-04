"use client";

import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { 
  Users, 
  CheckCircle, 
  ExternalLink, 
  TrendingUp,
  Eye,
  Heart,
  MessageCircle,
  DollarSign,
  Shield,
  Star,
  Target
} from "lucide-react";
import { cn } from "@/lib/utils";

// AIDEV-NOTE: 250903170000 - Comprehensive KOL card component for POC2 results display

export interface KOLScore {
  overallScore: number;
  scoreComponents: {
    roiScore: number;
    audienceQualityScore: number;
    brandSafetyScore: number;
    contentRelevanceScore: number;
    demographicFitScore: number;
    reliabilityScore?: number;
    overallConfidence: number;
  };
  semanticMatching?: {
    similarityScore: number;
    contentMatchScore: number;
    brandAffinityScore: number;
    matchedContentCategories: string[];
    semanticKeywords: string[];
  };
  performancePrediction?: {
    predictedReach: number;
    predictedEngagement: number;
    predictedConversions: number;
    predictedRoi: number;
    predictionConfidence: number;
  };
}

export interface KOLData {
  id: string;
  username: string;
  displayName: string;
  platform: string;
  tier: string;
  primaryCategory: string;
  isVerified: boolean;
  isBrandSafe: boolean;
  profileImageUrl?: string;
  followerCount: number;
  engagementRate: number;
  averageViews: number;
  ratePerPost: number;
  score: KOLScore;
  matchReasons?: string[];
  fitExplanation?: string;
  potentialConcerns?: string[];
  recommendedApproach?: string;
}

export interface KOLCardProps {
  kol: KOLData;
  variant?: "default" | "compact" | "detailed";
  showMatchInsights?: boolean;
  showPredictions?: boolean;
  showActions?: boolean;
  onContact?: (kolId: string) => void;
  onViewProfile?: (kolId: string) => void;
  onAddToShortlist?: (kolId: string) => void;
  className?: string;
}

export function KOLCard({
  kol,
  variant = "default",
  showMatchInsights = false,
  showPredictions = false,
  showActions = true,
  onContact,
  onViewProfile,
  onAddToShortlist,
  className
}: KOLCardProps) {
  // Utility functions
  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0
    }).format(amount);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const getTierBadgeVariant = (tier: string) => {
    switch (tier.toLowerCase()) {
      case 'nano': return 'outline';
      case 'micro': return 'secondary';
      case 'mid': return 'default';
      case 'macro': return 'success';
      case 'mega': return 'warning';
      default: return 'outline';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getPlatformIcon = (platform: string) => {
    // Could add platform-specific icons here
    return <Users className="h-4 w-4" />;
  };

  if (variant === "compact") {
    return (
      <Card className={cn("hover:shadow-md transition-shadow", className)}>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 bg-muted rounded-full flex items-center justify-center">
                {getPlatformIcon(kol.platform)}
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h4 className="font-semibold text-sm">{kol.displayName}</h4>
                  {kol.isVerified && (
                    <CheckCircle className="h-3 w-3 text-blue-500" />
                  )}
                </div>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span>@{kol.username}</span>
                  <span>•</span>
                  <Badge variant={getTierBadgeVariant(kol.tier)} className="text-xs">
                    {kol.tier}
                  </Badge>
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className={`text-lg font-bold ${getScoreColor(kol.score.overallScore)}`}>
                {(kol.score.overallScore * 100).toFixed(0)}
              </div>
              <div className="text-xs text-muted-foreground">Score</div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn("overflow-hidden hover:shadow-md transition-shadow", className)}>
      <CardContent className="p-6">
        <div className={cn(
          "grid gap-6",
          variant === "detailed" ? "lg:grid-cols-4" : "md:grid-cols-3"
        )}>
          {/* Profile Section */}
          <div className={variant === "detailed" ? "lg:col-span-1" : ""}>
            <div className="flex items-start gap-3">
              <div className="h-12 w-12 bg-muted rounded-full flex items-center justify-center flex-shrink-0">
                {kol.profileImageUrl ? (
                  <img 
                    src={kol.profileImageUrl} 
                    alt={kol.displayName}
                    className="h-12 w-12 rounded-full object-cover"
                  />
                ) : (
                  getPlatformIcon(kol.platform)
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h4 className="font-semibold truncate">{kol.displayName}</h4>
                  {kol.isVerified && (
                    <CheckCircle className="h-4 w-4 text-blue-500 flex-shrink-0" />
                  )}
                </div>
                <p className="text-sm text-muted-foreground mb-2">@{kol.username}</p>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant={getTierBadgeVariant(kol.tier)}>
                    {kol.tier}
                  </Badge>
                  <Badge variant="outline">{kol.platform}</Badge>
                  {kol.isBrandSafe && (
                    <Badge variant="success" className="text-xs">
                      <Shield className="h-3 w-3 mr-1" />
                      Safe
                    </Badge>
                  )}
                </div>
                <div className="mt-2">
                  <p className="text-xs text-muted-foreground">{kol.primaryCategory}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Metrics Section */}
          <div className={variant === "detailed" ? "lg:col-span-1" : ""}>
            <div className="grid grid-cols-2 gap-3">
              <div className="flex items-center gap-2">
                <Users className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Followers</p>
                  <p className="font-semibold text-sm">{formatNumber(kol.followerCount)}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Heart className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Engagement</p>
                  <p className="font-semibold text-sm">{formatPercentage(kol.engagementRate)}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Eye className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Avg Views</p>
                  <p className="font-semibold text-sm">{formatNumber(kol.averageViews)}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <DollarSign className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Rate</p>
                  <p className="font-semibold text-sm">{formatCurrency(kol.ratePerPost)}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Scoring Section */}
          <div className={variant === "detailed" ? "lg:col-span-1" : ""}>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Overall Score</span>
                <span className={`text-lg font-bold ${getScoreColor(kol.score.overallScore)}`}>
                  {(kol.score.overallScore * 100).toFixed(0)}
                </span>
              </div>
              <Progress value={kol.score.overallScore * 100} className="h-2" />
              
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">ROI:</span>
                  <span className="font-medium">{(kol.score.scoreComponents.roiScore * 100).toFixed(0)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Relevance:</span>
                  <span className="font-medium">{(kol.score.scoreComponents.contentRelevanceScore * 100).toFixed(0)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Quality:</span>
                  <span className="font-medium">{(kol.score.scoreComponents.audienceQualityScore * 100).toFixed(0)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Safety:</span>
                  <span className="font-medium">{(kol.score.scoreComponents.brandSafetyScore * 100).toFixed(0)}</span>
                </div>
              </div>

              <div className="text-center text-xs text-muted-foreground">
                Confidence: {formatPercentage(kol.score.scoreComponents.overallConfidence)}
              </div>
            </div>
          </div>

          {/* Predictions & Actions Section */}
          {(showPredictions || showActions) && (
            <div className={variant === "detailed" ? "lg:col-span-1" : ""}>
              <div className="space-y-3">
                {showPredictions && kol.score.performancePrediction && (
                  <>
                    <div className="text-center p-3 bg-muted/50 rounded-lg">
                      <div className="flex items-center justify-center gap-1 mb-1">
                        <TrendingUp className="h-4 w-4 text-green-600" />
                        <span className="text-xs text-muted-foreground">Predicted ROI</span>
                      </div>
                      <p className="font-bold text-lg text-green-600">
                        {formatPercentage(kol.score.performancePrediction.predictedRoi)}
                      </p>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-muted-foreground">Reach:</span>
                        <p className="font-medium">{formatNumber(kol.score.performancePrediction.predictedReach)}</p>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Conversions:</span>
                        <p className="font-medium">{formatNumber(kol.score.performancePrediction.predictedConversions)}</p>
                      </div>
                    </div>
                  </>
                )}

                {showActions && (
                  <div className="flex gap-2">
                    <Button 
                      size="sm" 
                      className="flex-1"
                      onClick={() => onContact?.(kol.id)}
                    >
                      Contact
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => onViewProfile?.(kol.id)}
                    >
                      <ExternalLink className="h-3 w-3" />
                    </Button>
                    {onAddToShortlist && (
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => onAddToShortlist(kol.id)}
                      >
                        <Star className="h-3 w-3" />
                      </Button>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Match Insights Section */}
        {showMatchInsights && (kol.fitExplanation || kol.potentialConcerns?.length || kol.score.semanticMatching?.semanticKeywords?.length) && (
          <>
            <Separator className="my-4" />
            <div className="space-y-3">
              {kol.fitExplanation && (
                <div>
                  <h5 className="font-medium mb-1 text-sm flex items-center gap-2">
                    <Target className="h-4 w-4 text-green-600" />
                    Why This KOL Fits
                  </h5>
                  <p className="text-sm text-muted-foreground">{kol.fitExplanation}</p>
                </div>
              )}

              {kol.potentialConcerns?.length > 0 && (
                <div>
                  <h5 className="font-medium mb-1 text-sm text-yellow-600">Considerations</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    {kol.potentialConcerns.slice(0, 3).map((concern, idx) => (
                      <li key={idx} className="text-xs flex items-start gap-2">
                        <span className="text-yellow-600 mt-1">•</span>
                        <span>{concern}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              {kol.score.semanticMatching?.semanticKeywords?.length > 0 && (
                <div>
                  <h5 className="font-medium mb-2 text-sm">Matched Keywords</h5>
                  <div className="flex flex-wrap gap-1">
                    {kol.score.semanticMatching.semanticKeywords.slice(0, 6).map((keyword, idx) => (
                      <Badge key={idx} variant="outline" className="text-xs">
                        {keyword}
                      </Badge>
                    ))}
                    {kol.score.semanticMatching.semanticKeywords.length > 6 && (
                      <Badge variant="outline" className="text-xs">
                        +{kol.score.semanticMatching.semanticKeywords.length - 6} more
                      </Badge>
                    )}
                  </div>
                </div>
              )}

              {kol.recommendedApproach && (
                <div>
                  <h5 className="font-medium mb-1 text-sm">Recommended Approach</h5>
                  <p className="text-xs text-muted-foreground">{kol.recommendedApproach}</p>
                </div>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}