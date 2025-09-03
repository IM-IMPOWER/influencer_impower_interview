"use client";

import { useQuery } from "@apollo/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { HEALTH_CHECK } from "@/lib/graphql/sophisticated-queries";
import { Activity, Database, Zap, Brain } from "lucide-react";
import Link from "next/link";

// AIDEV-NOTE: 250903170013 - Home page migrated to Apollo GraphQL with enhanced health monitoring

const TITLE_TEXT = `
 ██╗  ██╗ ██████╗ ██╗         ██████╗ ██╗      █████╗ ████████╗███████╗ ██████╗ ██████╗ ███╗   ███╗
 ██║ ██╔╝██╔═══██╗██║         ██╔══██╗██║     ██╔══██╗╚══██╔══╝██╔════╝██╔═══██╗██╔══██╗████╗ ████║
 █████╔╝ ██║   ██║██║         ██████╔╝██║     ███████║   ██║   █████╗  ██║   ██║██████╔╝██╔████╔██║
 ██╔═██╗ ██║   ██║██║         ██╔═══╝ ██║     ██╔══██║   ██║   ██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║
 ██║  ██╗╚██████╔╝███████╗    ██║     ███████╗██║  ██║   ██║   ██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║
 ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝
                                                                                                       
         ██╗███╗   ██╗███████╗██╗     ██╗   ██╗███████╗███╗   ██╗ ██████╗███████╗██████╗             
         ██║████╗  ██║██╔════╝██║     ██║   ██║██╔════╝████╗  ██║██╔════╝██╔════╝██╔══██╗            
         ██║██╔██╗ ██║█████╗  ██║     ██║   ██║█████╗  ██╔██╗ ██║██║     █████╗  ██████╔╝            
         ██║██║╚██╗██║██╔══╝  ██║     ██║   ██║██╔══╝  ██║╚██╗██║██║     ██╔══╝  ██╔══██╗            
         ██║██║ ╚████║██║     ███████╗╚██████╔╝███████╗██║ ╚████║╚██████╗███████╗██║  ██║            
         ╚═╝╚═╝  ╚═══╝╚═╝     ╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚═╝  ╚═╝            
 `;

const getStatusIcon = (service: string) => {
  switch (service.toLowerCase()) {
    case 'database': return <Database className="h-4 w-4" />
    case 'redis': return <Zap className="h-4 w-4" />
    case 'ml_service': return <Brain className="h-4 w-4" />
    default: return <Activity className="h-4 w-4" />
  }
}

export default function Home() {
  const { data, loading, error, refetch } = useQuery(HEALTH_CHECK, {
    pollInterval: 30000, // Check health every 30 seconds
    errorPolicy: 'all'
  });

  const healthData = data?.healthCheck;
  const services = healthData?.services;
  const isHealthy = healthData?.status === 'healthy';

  return (
    <div className="container mx-auto max-w-6xl px-4 py-8">
      {/* ASCII Art Title */}
      <div className="mb-8 overflow-x-auto">
        <pre className="font-mono text-xs text-muted-foreground whitespace-pre">
          {TITLE_TEXT}
        </pre>
      </div>

      {/* Welcome Section */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-4">
          Welcome to KOL Platform
        </h1>
        <p className="text-xl text-muted-foreground mb-8">
          AI-powered influencer marketing with intelligent KOL matching and budget optimization
        </p>
        <div className="flex flex-wrap justify-center gap-4">
          <Link href="/poc2">
            <Button size="lg" className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              POC2: Brief Matching
            </Button>
          </Link>
          <Link href="/kol-matching">
            <Button variant="outline" size="lg">
              Discover KOLs
            </Button>
          </Link>
          <Link href="/campaigns">
            <Button variant="outline" size="lg">
              View Campaigns
            </Button>
          </Link>
        </div>
      </div>

      {/* System Status */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              System Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Overall Status */}
              <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                <div className="flex items-center gap-3">
                  <div
                    className={`h-3 w-3 rounded-full ${
                      loading 
                        ? "bg-yellow-500 animate-pulse" 
                        : isHealthy 
                          ? "bg-green-500" 
                          : "bg-red-500"
                    }`}
                  />
                  <span className="font-medium">Platform Status</span>
                </div>
                <Badge variant={isHealthy ? "success" : error ? "destructive" : "warning"}>
                  {loading ? "Checking..." : isHealthy ? "Operational" : "Issues Detected"}
                </Badge>
              </div>

              {/* Individual Services */}
              {services && Object.entries(services).map(([service, status]) => (
                <div key={service} className="flex items-center justify-between p-2">
                  <div className="flex items-center gap-2">
                    {getStatusIcon(service)}
                    <span className="capitalize">
                      {service.replace('_', ' ')}
                    </span>
                  </div>
                  <Badge variant={status === 'healthy' ? 'success' : 'destructive'}>
                    {status === 'healthy' ? 'Online' : 'Offline'}
                  </Badge>
                </div>
              ))}

              {/* Refresh Button */}
              <div className="pt-2 border-t">
                <Button 
                  onClick={() => refetch()} 
                  variant="outline" 
                  size="sm" 
                  className="w-full"
                  disabled={loading}
                >
                  {loading ? "Refreshing..." : "Refresh Status"}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* System Information */}
        <Card>
          <CardHeader>
            <CardTitle>Platform Information</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {healthData?.version && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Version:</span>
                  <span className="font-mono">{healthData.version}</span>
                </div>
              )}
              {healthData?.timestamp && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Last Check:</span>
                  <span className="font-mono text-sm">
                    {new Date(healthData.timestamp).toLocaleString()}
                  </span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-muted-foreground">API Type:</span>
                <Badge variant="outline">GraphQL</Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Client:</span>
                <Badge variant="outline">Apollo Client</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Navigation */}
      <div className="mt-12">
        <h2 className="text-2xl font-semibold mb-6 text-center">Quick Access</h2>
        <div className="grid gap-4 md:grid-cols-4">
          <Link href="/poc2">
            <Card className="cursor-pointer hover:shadow-md transition-shadow">
              <CardContent className="p-6 text-center">
                <Brain className="h-8 w-8 mx-auto mb-3 text-primary" />
                <h3 className="font-medium mb-2">POC2: Brief Matching</h3>
                <p className="text-sm text-muted-foreground">Upload briefs for AI KOL matching</p>
              </CardContent>
            </Card>
          </Link>
          
          <Link href="/kol-matching">
            <Card className="cursor-pointer hover:shadow-md transition-shadow">
              <CardContent className="p-6 text-center">
                <Activity className="h-8 w-8 mx-auto mb-3 text-primary" />
                <h3 className="font-medium mb-2">KOL Discovery</h3>
                <p className="text-sm text-muted-foreground">Browse and filter KOLs</p>
              </CardContent>
            </Card>
          </Link>
          
          <Link href="/campaigns">
            <Card className="cursor-pointer hover:shadow-md transition-shadow">
              <CardContent className="p-6 text-center">
                <Database className="h-8 w-8 mx-auto mb-3 text-primary" />
                <h3 className="font-medium mb-2">Campaigns</h3>
                <p className="text-sm text-muted-foreground">Manage your marketing campaigns</p>
              </CardContent>
            </Card>
          </Link>
          
          <Link href="/dashboard">
            <Card className="cursor-pointer hover:shadow-md transition-shadow">
              <CardContent className="p-6 text-center">
                <Activity className="h-8 w-8 mx-auto mb-3 text-primary" />
                <h3 className="font-medium mb-2">Dashboard</h3>
                <p className="text-sm text-muted-foreground">Analytics and insights</p>
              </CardContent>
            </Card>
          </Link>
        </div>
      </div>
    </div>
  );
}
