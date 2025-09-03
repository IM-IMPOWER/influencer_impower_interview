"use client";

import React, { useState, useCallback } from "react";
import { useMutation, useQuery } from "@apollo/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import { Separator } from "@/components/ui/separator";
import { FileUpload } from "@/components/ui/file-upload";
import { KOLCard, type KOLData } from "@/components/kol/kol-card";
import { KOLResultsSkeleton } from "@/components/kol/kol-card-skeleton";
import { 
  MATCH_KOLS_TO_BRIEF, 
  GET_BRIEF_PROCESSING_STATUS 
} from "@/lib/graphql/sophisticated-queries";
import { 
  Upload, 
  FileText, 
  Brain, 
  Target, 
  TrendingUp, 
  Users, 
  CheckCircle, 
  AlertCircle,
  Download,
  Filter,
  Star,
  FileDown,
  Info,
  ExternalLink,
  Zap,
  FileCheck,
  Clock,
  DollarSign
} from "lucide-react";
import { toast } from "sonner";
import Link from "next/link";

// AIDEV-NOTE: 250903170000 - POC2 main page component with comprehensive file upload and KOL matching interface

// Using KOLData interface from the KOL card component
type MatchedKOL = KOLData;

interface ExtractedRequirements {
  targetKolTiers: string[];
  targetCategories: string[];
  totalBudget: number;
  minFollowerCount?: number;
  maxFollowerCount?: number;
  minEngagementRate?: number;
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
}

interface BriefAnalysis {
  extractedThemes: string[];
  brandTone: string;
  targetAudience: string;
  contentRequirements: string[];
  budgetInsights: string[];
  timelineAnalysis: string;
  feasibilityScore: number;
  complexityLevel: string;
}

interface ProcessingStatus {
  id: string;
  status: "pending" | "processing" | "completed" | "error";
  progress: number;
  message: string;
  startedAt: string;
  completedAt?: string;
  errorDetails?: string;
  resultUrl?: string;
}

export default function POC2Page() {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [matchResults, setMatchResults] = useState<{
    extractedRequirements: ExtractedRequirements;
    matchedKols: MatchedKOL[];
    briefAnalysis: BriefAnalysis;
    totalMatches: number;
    processingTimeSeconds: number;
    briefParsingConfidence: number;
    matchingAlgorithm: string;
    dataQualityWarnings: string[];
  } | null>(null);
  const [selectedKOLs, setSelectedKOLs] = useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = useState<"score" | "followers" | "engagement" | "rate">("score");
  const [filterTier, setFilterTier] = useState<string>("all");

  // AIDEV-NOTE: 250903170000 - Apollo mutations and queries for POC2 functionality
  const [matchKolsToBrief, { loading: matchingLoading }] = useMutation(MATCH_KOLS_TO_BRIEF, {
    onCompleted: (data) => {
      if (data.matchKolsToBrief.success) {
        setMatchResults(data.matchKolsToBrief);
        toast.success("KOL matching completed successfully!");
      } else {
        toast.error(data.matchKolsToBrief.message || "Failed to match KOLs");
      }
    },
    onError: (error) => {
      toast.error(`Error: ${error.message}`);
    }
  });

  const { data: statusData } = useQuery(GET_BRIEF_PROCESSING_STATUS, {
    variables: { processingId },
    skip: !processingId,
    pollInterval: 2000, // Poll every 2 seconds while processing
  });

  // AIDEV-NOTE: 250903170000 - File upload drag and drop handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const mdFile = files.find(file => 
      file.name.toLowerCase().endsWith('.md') || 
      file.name.toLowerCase().endsWith('.markdown')
    );
    
    if (mdFile) {
      if (mdFile.size > 10 * 1024 * 1024) { // 10MB limit
        toast.error("File size must be less than 10MB");
        return;
      }
      setUploadedFile(mdFile);
      toast.success(`Selected: ${mdFile.name}`);
    } else {
      toast.error("Please upload a .md or .markdown file");
    }
  }, []);

  // File selection is now handled by the FileUpload component

  const handleProcessBrief = useCallback(async () => {
    if (!uploadedFile) {
      toast.error("Please select a file first");
      return;
    }

    try {
      await matchKolsToBrief({
        variables: {
          briefFile: uploadedFile,
          confidenceThreshold: 0.7,
          limit: 50,
          enableSemanticMatching: true
        }
      });
    } catch (error) {
      console.error("Error processing brief:", error);
    }
  }, [uploadedFile, matchKolsToBrief]);

  const handleClearResults = useCallback(() => {
    setMatchResults(null);
    setUploadedFile(null);
    setProcessingId(null);
    setSelectedKOLs(new Set());
  }, []);

  const handleContactKOL = useCallback((kolId: string) => {
    // TODO: Implement contact functionality
    toast.success(`Contacting KOL ${kolId}`);
  }, []);

  const handleViewProfile = useCallback((kolId: string) => {
    // TODO: Implement profile view functionality
    window.open(`/kol/${kolId}`, '_blank');
  }, []);

  const handleAddToShortlist = useCallback((kolId: string) => {
    setSelectedKOLs(prev => {
      const newSet = new Set(prev);
      if (newSet.has(kolId)) {
        newSet.delete(kolId);
        toast.success("Removed from shortlist");
      } else {
        newSet.add(kolId);
        toast.success("Added to shortlist");
      }
      return newSet;
    });
  }, []);

  const handleFileSelect = useCallback((file: File) => {
    setUploadedFile(file);
    toast.success(`Selected: ${file.name}`);
  }, []);

  const handleFileRemove = useCallback(() => {
    setUploadedFile(null);
  }, []);

  // AIDEV-NOTE: 250903170000 - Format utility functions
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

  // Utility functions moved to KOL card component

  return (
    <div className="container mx-auto max-w-7xl px-4 py-8">
      {/* Header Section */}
      <div className="text-center mb-12">
        <div className="flex items-center justify-center gap-3 mb-4">
          <Brain className="h-8 w-8 text-primary" />
          <h1 className="text-4xl font-bold tracking-tight">
            POC2: KOL-to-Brief Matching
          </h1>
        </div>
        <p className="text-xl text-muted-foreground mb-6 max-w-3xl mx-auto">
          Upload your campaign brief as a markdown file and get AI-powered KOL recommendations 
          with advanced semantic matching and performance predictions
        </p>
        
        {/* Feature Highlights */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto">
          <div className="flex items-center justify-center gap-2 p-3 bg-muted/50 rounded-lg">
            <FileText className="h-5 w-5 text-primary" />
            <span className="text-sm font-medium">Brief Analysis</span>
          </div>
          <div className="flex items-center justify-center gap-2 p-3 bg-muted/50 rounded-lg">
            <Target className="h-5 w-5 text-primary" />
            <span className="text-sm font-medium">Smart Matching</span>
          </div>
          <div className="flex items-center justify-center gap-2 p-3 bg-muted/50 rounded-lg">
            <TrendingUp className="h-5 w-5 text-primary" />
            <span className="text-sm font-medium">ROI Prediction</span>
          </div>
          <div className="flex items-center justify-center gap-2 p-3 bg-muted/50 rounded-lg">
            <Users className="h-5 w-5 text-primary" />
            <span className="text-sm font-medium">Audience Fit</span>
          </div>
        </div>
      </div>

      {!matchResults ? (
        <>
          {/* File Upload Section */}
          <Card className="max-w-2xl mx-auto mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Upload Campaign Brief
              </CardTitle>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Info className="h-4 w-4" />
                <span>Need a template?</span>
                <a 
                  href="/sample-campaign-brief.md" 
                  download="sample-campaign-brief.md"
                  className="text-primary hover:underline flex items-center gap-1"
                >
                  <FileDown className="h-3 w-3" />
                  Download Sample
                </a>
              </div>
            </CardHeader>
            <CardContent>
              <FileUpload
                onFileSelect={handleFileSelect}
                onFileRemove={handleFileRemove}
                accept=".md,.markdown"
                maxSize={10 * 1024 * 1024} // 10MB
                placeholder="Drop your campaign brief here"
                description="or click to browse files (.md, .markdown)"
                disabled={matchingLoading}
              />
              
              {uploadedFile && (
                <div className="flex gap-2 justify-center mt-4">
                  <Button onClick={handleProcessBrief} disabled={matchingLoading}>
                    {matchingLoading ? (
                      <>
                        <Spinner size="sm" className="mr-2" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Zap className="h-4 w-4 mr-2" />
                        Analyze & Match KOLs
                      </>
                    )}
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* How it Works Section */}
          <Card className="max-w-4xl mx-auto">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                How POC2 Works
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="text-center space-y-3">
                  <div className="h-12 w-12 bg-primary/10 rounded-full flex items-center justify-center mx-auto">
                    <span className="font-bold text-primary">1</span>
                  </div>
                  <h3 className="font-semibold">Brief Analysis</h3>
                  <p className="text-sm text-muted-foreground">
                    AI extracts campaign requirements, target audience, budget, and key themes from your markdown brief
                  </p>
                </div>
                <div className="text-center space-y-3">
                  <div className="h-12 w-12 bg-primary/10 rounded-full flex items-center justify-center mx-auto">
                    <span className="font-bold text-primary">2</span>
                  </div>
                  <h3 className="font-semibold">Semantic Matching</h3>
                  <p className="text-sm text-muted-foreground">
                    Advanced algorithms match KOLs based on content relevance, audience fit, and performance metrics
                  </p>
                </div>
                <div className="text-center space-y-3">
                  <div className="h-12 w-12 bg-primary/10 rounded-full flex items-center justify-center mx-auto">
                    <span className="font-bold text-primary">3</span>
                  </div>
                  <h3 className="font-semibold">Smart Recommendations</h3>
                  <p className="text-sm text-muted-foreground">
                    Get ranked KOL suggestions with detailed scoring, ROI predictions, and actionable insights
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      ) : (
        // Results Section
        <div className="space-y-8">
          {/* Results Header */}
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold flex items-center gap-2">
                <CheckCircle className="h-6 w-6 text-green-600" />
                Matching Results
                {demoMode && (
                  <Badge variant="outline" className="ml-2">
                    Demo Mode
                  </Badge>
                )}
              </h2>
              <p className="text-muted-foreground">
                Found {matchResults.totalMatches} matching KOLs in {matchResults.processingTimeSeconds.toFixed(2)}s
                {demoMode && " (simulated)"}
              </p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={handleClearResults}>
                New Analysis
              </Button>
              <Button>
                <Download className="h-4 w-4 mr-2" />
                Export Results
              </Button>
            </div>
          </div>

          {/* Brief Analysis Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Campaign Brief Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div>
                  <h4 className="font-medium text-sm text-muted-foreground mb-2">CAMPAIGN OBJECTIVE</h4>
                  <p className="font-semibold">{matchResults.extractedRequirements.campaignObjective}</p>
                </div>
                <div>
                  <h4 className="font-medium text-sm text-muted-foreground mb-2">TOTAL BUDGET</h4>
                  <p className="font-semibold">{formatCurrency(matchResults.extractedRequirements.totalBudget)}</p>
                </div>
                <div>
                  <h4 className="font-medium text-sm text-muted-foreground mb-2">TARGET CATEGORIES</h4>
                  <div className="flex flex-wrap gap-1">
                    {matchResults.extractedRequirements.targetCategories.slice(0, 2).map((category, idx) => (
                      <Badge key={idx} variant="outline" className="text-xs">
                        {category}
                      </Badge>
                    ))}
                    {matchResults.extractedRequirements.targetCategories.length > 2 && (
                      <Badge variant="outline" className="text-xs">
                        +{matchResults.extractedRequirements.targetCategories.length - 2} more
                      </Badge>
                    )}
                  </div>
                </div>
                <div>
                  <h4 className="font-medium text-sm text-muted-foreground mb-2">PARSING CONFIDENCE</h4>
                  <div className="flex items-center gap-2">
                    <Progress value={matchResults.briefParsingConfidence * 100} className="flex-1" />
                    <span className="text-sm font-medium">
                      {formatPercentage(matchResults.briefParsingConfidence)}
                    </span>
                  </div>
                </div>
              </div>

              {matchResults.briefAnalysis.extractedThemes.length > 0 && (
                <div className="mt-6 pt-6 border-t">
                  <h4 className="font-medium mb-3">Content Themes</h4>
                  <div className="flex flex-wrap gap-2">
                    {matchResults.briefAnalysis.extractedThemes.map((theme, idx) => (
                      <Badge key={idx} variant="secondary">
                        {theme}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Warning alerts */}
          {matchResults.dataQualityWarnings.length > 0 && (
            <Alert variant="warning">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Data Quality Notices</AlertTitle>
              <AlertDescription>
                <ul className="list-disc list-inside space-y-1 mt-2">
                  {matchResults.dataQualityWarnings.map((warning, idx) => (
                    <li key={idx} className="text-sm">{warning}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}

          {/* KOL Results Grid */}
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold">Recommended KOLs</h3>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm">
                  <Filter className="h-4 w-4 mr-2" />
                  Filter
                </Button>
                <select 
                  className="px-3 py-1 text-sm border rounded-md"
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                >
                  <option value="score">Sort by Score</option>
                  <option value="followers">Sort by Followers</option>
                  <option value="engagement">Sort by Engagement</option>
                  <option value="rate">Sort by Rate</option>
                </select>
              </div>
            </div>

            {matchingLoading ? (
              <KOLResultsSkeleton count={5} />
            ) : (
              <div className="space-y-4">
                {matchResults.matchedKols.map((kol, index) => (
                  <KOLCard
                    key={kol.id}
                    kol={kol}
                    variant="detailed"
                    showMatchInsights={true}
                    showPredictions={true}
                    showActions={true}
                    onContact={handleContactKOL}
                    onViewProfile={handleViewProfile}
                    onAddToShortlist={handleAddToShortlist}
                    className={selectedKOLs.has(kol.id) ? "ring-2 ring-primary" : ""}
                  />
                ))}
              </div>
            )}

            {/* Shortlist Summary */}
            {selectedKOLs.size > 0 && (
              <Card className="mt-6">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Star className="h-5 w-5 text-yellow-500" />
                      <span className="font-medium">{selectedKOLs.size} KOLs shortlisted</span>
                    </div>
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" onClick={() => setSelectedKOLs(new Set())}>
                        Clear All
                      </Button>
                      <Button size="sm">
                        <Download className="h-4 w-4 mr-2" />
                        Export Shortlist
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Summary Statistics */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Campaign Projections
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-4 gap-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-primary">
                    {formatNumber(
                      matchResults.matchedKols.reduce((sum, kol) => sum + kol.score.performancePrediction.predictedReach, 0)
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground">Total Predicted Reach</p>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {formatPercentage(
                      matchResults.matchedKols.reduce((sum, kol) => sum + kol.score.performancePrediction.predictedRoi, 0) / matchResults.matchedKols.length
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground">Average ROI</p>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {formatCurrency(
                      matchResults.matchedKols.reduce((sum, kol) => sum + kol.ratePerPost, 0)
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground">Total Investment</p>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {formatNumber(
                      matchResults.matchedKols.reduce((sum, kol) => sum + kol.score.performancePrediction.predictedConversions, 0)
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground">Expected Conversions</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Processing Status */}
      {matchingLoading && (
        <Card className="mt-8">
          <CardContent className="p-6">
            <div className="flex items-center justify-center space-x-4">
              <Spinner size="md" />
              <div className="text-center">
                <h3 className="font-medium mb-2">Analyzing Your Campaign Brief</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Our AI is extracting requirements and matching with the best KOLs...
                </p>
                <div className="flex items-center justify-center space-x-6 text-sm">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span>Brief parsed</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Clock className="h-4 w-4 animate-spin text-blue-500" />
                    <span>Finding matches</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                    <span>Scoring results</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quick Navigation */}
      <div className="mt-12 pt-8 border-t">
        <h3 className="text-lg font-semibold mb-4 text-center">Explore More Features</h3>
        <div className="grid md:grid-cols-3 gap-4 max-w-3xl mx-auto">
          <Link href="/kol-matching">
            <Card className="cursor-pointer hover:shadow-md transition-shadow">
              <CardContent className="p-4 text-center">
                <Users className="h-6 w-6 mx-auto mb-2 text-primary" />
                <h4 className="font-medium mb-1">KOL Discovery</h4>
                <p className="text-xs text-muted-foreground">Browse and filter KOLs</p>
              </CardContent>
            </Card>
          </Link>
          <Link href="/campaigns">
            <Card className="cursor-pointer hover:shadow-md transition-shadow">
              <CardContent className="p-4 text-center">
                <TrendingUp className="h-6 w-6 mx-auto mb-2 text-primary" />
                <h4 className="font-medium mb-1">Campaign Manager</h4>
                <p className="text-xs text-muted-foreground">Manage active campaigns</p>
              </CardContent>
            </Card>
          </Link>
          <Link href="/budget-optimizer">
            <Card className="cursor-pointer hover:shadow-md transition-shadow">
              <CardContent className="p-4 text-center">
                <DollarSign className="h-6 w-6 mx-auto mb-2 text-primary" />
                <h4 className="font-medium mb-1">Budget Optimizer</h4>
                <p className="text-xs text-muted-foreground">Optimize spending plans</p>
              </CardContent>
            </Card>
          </Link>
        </div>
      </div>
    </div>
  );
}