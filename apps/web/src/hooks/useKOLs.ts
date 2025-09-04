"use client"

import { useState, useEffect } from 'react'
import { useQuery, useMutation } from '@apollo/client'
import { MATCH_KOLS_TO_BRIEF, DISCOVER_KOLS } from '@/lib/graphql/kol.graphql'
import { KOL } from '@/lib/types'

// AIDEV-NOTE: 250903170006 - Enhanced KOL hooks with Apollo GraphQL operations

export function useKOLMatching(brief: string) {
  const { data, loading, error } = useQuery(MATCH_KOLS_TO_BRIEF, {
    variables: { brief },
    skip: !brief || brief.length < 3,
    pollInterval: 2000 // Refresh every 2s as ML processes
  })

  return {
    matches: data?.matchKOLsToBrief || [],
    loading,
    error,
    hasResults: data?.matchKOLsToBrief?.length > 0
  }
}

export function useKOLPlan() {
  const [planKOLs, setPlanKOLs] = useState<KOL[]>([])

  const addToPlan = (kol: KOL) => {
    if (!planKOLs.find(k => k.id === kol.id)) {
      setPlanKOLs(prev => [...prev, kol])
    }
  }

  const removeFromPlan = (kolId: string) => {
    setPlanKOLs(prev => prev.filter(k => k.id !== kolId))
  }

  const clearPlan = () => setPlanKOLs([])

  return {
    planKOLs,
    addToPlan,
    removeFromPlan,
    clearPlan,
    planCount: planKOLs.length
  }
}