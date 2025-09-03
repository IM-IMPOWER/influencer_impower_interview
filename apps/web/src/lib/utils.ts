import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import { Campaign } from "@/types"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export const getStatusVariant = (status: Campaign['status']) => {
  switch (status) {
    case 'Active': return 'success'
    case 'Pending':
    case 'Planning': return 'warning'
    case 'Completed': return 'default'
    default: return 'default'
  }
}

export const formatCurrency = (amount: string | number) => {
  const num = typeof amount === 'string' ? parseFloat(amount.replace(/[$,]/g, '')) : amount
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(num)
}

export const formatNumber = (num: string | number) => {
  const value = typeof num === 'string' ? parseFloat(num.replace(/[^0-9.-]/g, '')) : num
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K`
  return value.toString()
}

export const formatDate = (date: string) => {
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  }).format(new Date(date))
}

export const getRelativeTime = (date: string) => {
  const now = new Date()
  const past = new Date(date)
  const diff = now.getTime() - past.getTime()
  
  const minutes = Math.floor(diff / 60000)
  const hours = Math.floor(diff / 3600000)
  const days = Math.floor(diff / 86400000)
  
  if (minutes < 60) return `${minutes}m ago`
  if (hours < 24) return `${hours}h ago`
  return `${days}d ago`
}