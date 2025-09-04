/** @type {import('next').NextConfig} */
const nextConfig = {
  // AIDEV-NOTE: 250903170017 - Next.js config optimized for GraphQL and Apollo Client
  
  experimental: {
    // Enable experimental features for better performance
    optimizePackageImports: ['@apollo/client', 'graphql', 'lucide-react'],
  },
  
  // Environment variables for GraphQL endpoints
  env: {
    NEXT_PUBLIC_GRAPHQL_ENDPOINT: process.env.NEXT_PUBLIC_GRAPHQL_ENDPOINT || 'http://localhost:8000/graphql',
    NEXT_PUBLIC_WS_ENDPOINT: process.env.NEXT_PUBLIC_WS_ENDPOINT || 'ws://localhost:8000/graphql',
  },
  
  // Webpack configuration for GraphQL files
  webpack: (config, { isServer }) => {
    // Handle .graphql files
    config.module.rules.push({
      test: /\.(graphql|gql)$/,
      exclude: /node_modules/,
      use: [
        {
          loader: 'graphql-tag/loader',
        },
      ],
    });
    
    // Optimize Apollo Client for client-side
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
      };
    }
    
    return config;
  },
  
  // Optimize images for KOL avatars and content
  images: {
    domains: [
      'localhost',
      'cdn.example.com', // Replace with your CDN domain
      'images.unsplash.com', // For demo purposes
      'avatars.githubusercontent.com', // For demo purposes
    ],
    formats: ['image/webp', 'image/avif'],
  },
  
  // Headers for CORS and security
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          {
            key: 'Access-Control-Allow-Origin',
            value: process.env.ALLOWED_ORIGIN || '*',
          },
          {
            key: 'Access-Control-Allow-Methods',
            value: 'GET,OPTIONS,PATCH,DELETE,POST,PUT',
          },
          {
            key: 'Access-Control-Allow-Headers',
            value: 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, Authorization',
          },
        ],
      },
    ];
  },
  
  // Rewrites for API proxying (if needed)
  async rewrites() {
    if (process.env.NODE_ENV === 'development') {
      return [
        {
          source: '/graphql',
          destination: process.env.NEXT_PUBLIC_GRAPHQL_ENDPOINT || 'http://localhost:8000/graphql',
        },
      ];
    }
    return [];
  },
  
  // Redirects
  async redirects() {
    return [
      {
        source: '/trpc/:path*',
        destination: '/', // Redirect old tRPC routes to home
        permanent: false,
      },
    ];
  },
  
  // Compiler options
  compiler: {
    // Remove console logs in production
    removeConsole: process.env.NODE_ENV === 'production',
  },
  
  // Output configuration
  output: 'standalone',
  
  // TypeScript configuration
  typescript: {
    // Type checking is handled by separate process
    ignoreBuildErrors: false,
  },
  
  // ESLint configuration
  eslint: {
    ignoreDuringBuilds: false,
  },
  
  // Performance optimizations
  poweredByHeader: false,
  generateEtags: true,
  compress: true,
};

module.exports = nextConfig;