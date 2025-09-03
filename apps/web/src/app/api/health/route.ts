// AIDEV-NOTE: 250903170025 Next.js health check endpoint for monitoring
// Provides health status for load balancers and monitoring systems

import { NextRequest, NextResponse } from 'next/server'

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy'
  timestamp: string
  uptime: number
  version: string
  checks: {
    [key: string]: {
      status: 'healthy' | 'degraded' | 'unhealthy'
      message?: string
      responseTime?: number
    }
  }
}

const startTime = Date.now()

export async function GET(request: NextRequest) {
  const health: HealthStatus = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: Date.now() - startTime,
    version: process.env.npm_package_version || '1.0.0',
    checks: {}
  }

  // AIDEV-NOTE: Check API backend connectivity
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  try {
    const apiCheckStart = Date.now()
    const apiResponse = await fetch(`${apiUrl}/api/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      // Short timeout for health checks
      signal: AbortSignal.timeout(5000)
    })
    
    const apiResponseTime = Date.now() - apiCheckStart
    
    if (apiResponse.ok) {
      health.checks.api = {
        status: 'healthy',
        message: 'API backend is responding',
        responseTime: apiResponseTime
      }
    } else {
      health.checks.api = {
        status: 'degraded',
        message: `API backend returned status ${apiResponse.status}`,
        responseTime: apiResponseTime
      }
      health.status = 'degraded'
    }
  } catch (error) {
    health.checks.api = {
      status: 'unhealthy',
      message: `API backend unreachable: ${error instanceof Error ? error.message : 'Unknown error'}`
    }
    health.status = 'degraded' // Web app can still serve static content
  }

  // AIDEV-NOTE: Check Node.js process health
  const memUsage = process.memoryUsage()
  const memUsageMB = Math.round(memUsage.heapUsed / 1024 / 1024)
  const memThresholdMB = 500 // 500MB threshold
  
  health.checks.memory = {
    status: memUsageMB > memThresholdMB ? 'degraded' : 'healthy',
    message: `Memory usage: ${memUsageMB}MB`,
    responseTime: 0
  }

  if (memUsageMB > memThresholdMB) {
    health.status = 'degraded'
  }

  // AIDEV-NOTE: Check environment variables
  const requiredEnvVars = ['NEXT_PUBLIC_API_URL']
  const missingEnvVars = requiredEnvVars.filter(envVar => !process.env[envVar])
  
  if (missingEnvVars.length > 0) {
    health.checks.configuration = {
      status: 'degraded',
      message: `Missing environment variables: ${missingEnvVars.join(', ')}`
    }
    health.status = 'degraded'
  } else {
    health.checks.configuration = {
      status: 'healthy',
      message: 'All required environment variables are set'
    }
  }

  // AIDEV-NOTE: Return appropriate HTTP status code
  const statusCode = health.status === 'healthy' ? 200 : 
                    health.status === 'degraded' ? 200 : 503

  return NextResponse.json(health, { 
    status: statusCode,
    headers: {
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Content-Type': 'application/json'
    }
  })
}

// AIDEV-NOTE: Simple liveness check for Kubernetes
export async function HEAD(request: NextRequest) {
  return new Response(null, { status: 200 })
}