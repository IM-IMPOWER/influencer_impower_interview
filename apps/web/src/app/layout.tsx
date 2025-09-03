import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import Providers from '@/components/providers'
import './globals.css'

// AIDEV-NOTE: 250903170005 - Updated layout to use centralized providers with Apollo GraphQL

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'KOL Platform - Influencer Campaign Management',
  description: 'Find the right Key Opinion Leaders for your campaigns with AI-powered matching and budget optimization.',
  keywords: 'influencer, KOL, campaign, marketing, social media, budget optimization'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className} suppressHydrationWarning>
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  )
}