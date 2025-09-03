import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

// AIDEV-NOTE: 250903170000 - Loading skeleton for KOL cards during data fetching

export interface KOLCardSkeletonProps {
  variant?: "default" | "compact" | "detailed";
  className?: string;
}

export function KOLCardSkeleton({ 
  variant = "default", 
  className 
}: KOLCardSkeletonProps) {
  if (variant === "compact") {
    return (
      <Card className={className}>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Skeleton className="h-10 w-10 rounded-full" />
              <div className="space-y-2">
                <Skeleton className="h-4 w-24" />
                <Skeleton className="h-3 w-20" />
              </div>
            </div>
            <Skeleton className="h-6 w-8" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardContent className="p-6">
        <div className="grid gap-6 lg:grid-cols-4">
          {/* Profile Section */}
          <div className="lg:col-span-1">
            <div className="flex items-start gap-3">
              <Skeleton className="h-12 w-12 rounded-full flex-shrink-0" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-3 w-24" />
                <div className="flex gap-2">
                  <Skeleton className="h-6 w-16 rounded-full" />
                  <Skeleton className="h-6 w-20 rounded-full" />
                  <Skeleton className="h-6 w-12 rounded-full" />
                </div>
                <Skeleton className="h-3 w-28" />
              </div>
            </div>
          </div>

          {/* Metrics Section */}
          <div className="lg:col-span-1">
            <div className="grid grid-cols-2 gap-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="flex items-center gap-2">
                  <Skeleton className="h-4 w-4" />
                  <div className="space-y-1">
                    <Skeleton className="h-3 w-12" />
                    <Skeleton className="h-4 w-16" />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Scoring Section */}
          <div className="lg:col-span-1">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Skeleton className="h-4 w-20" />
                <Skeleton className="h-6 w-8" />
              </div>
              <Skeleton className="h-2 w-full rounded-full" />
              <div className="grid grid-cols-2 gap-2">
                {Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className="flex justify-between">
                    <Skeleton className="h-3 w-12" />
                    <Skeleton className="h-3 w-6" />
                  </div>
                ))}
              </div>
              <div className="text-center">
                <Skeleton className="h-3 w-24 mx-auto" />
              </div>
            </div>
          </div>

          {/* Actions Section */}
          <div className="lg:col-span-1">
            <div className="space-y-3">
              <div className="p-3 bg-muted/50 rounded-lg text-center">
                <Skeleton className="h-3 w-20 mx-auto mb-2" />
                <Skeleton className="h-6 w-16 mx-auto" />
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <Skeleton className="h-3 w-12 mb-1" />
                  <Skeleton className="h-4 w-16" />
                </div>
                <div>
                  <Skeleton className="h-3 w-16 mb-1" />
                  <Skeleton className="h-4 w-12" />
                </div>
              </div>
              <div className="flex gap-2">
                <Skeleton className="h-8 flex-1" />
                <Skeleton className="h-8 w-8" />
                <Skeleton className="h-8 w-8" />
              </div>
            </div>
          </div>
        </div>

        {/* Match Insights Skeleton */}
        {variant === "detailed" && (
          <div className="mt-6 pt-6 border-t space-y-3">
            <div>
              <Skeleton className="h-4 w-32 mb-2" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
            </div>
            <div>
              <Skeleton className="h-4 w-24 mb-2" />
              <div className="flex flex-wrap gap-1">
                {Array.from({ length: 6 }).map((_, i) => (
                  <Skeleton key={i} className="h-6 w-16 rounded-full" />
                ))}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function KOLResultsSkeleton({ count = 3 }: { count?: number }) {
  return (
    <div className="space-y-6">
      {Array.from({ length: count }).map((_, i) => (
        <KOLCardSkeleton key={i} variant="detailed" />
      ))}
    </div>
  );
}