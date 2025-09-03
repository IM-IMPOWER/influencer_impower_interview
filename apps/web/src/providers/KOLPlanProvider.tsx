"use client"

import { createContext, useContext, useState, ReactNode } from 'react'
import { KOL } from '@/types'

interface KOLPlanContextType {
  planKOLs: KOL[]
  addToPlan: (kol: KOL) => void
  removeFromPlan: (kolId: string) => void
  clearPlan: () => void
  planCount: number
}

const KOLPlanContext = createContext<KOLPlanContextType | undefined>(undefined)

export function KOLPlanProvider({ children }: { children: ReactNode }) {
  const [planKOLs, setPlanKOLs] = useState<KOL[]>([])

  const addToPlan = (kol: KOL) => {
    setPlanKOLs(prev => {
      if (prev.find(k => k.id === kol.id)) return prev
      return [...prev, kol]
    })
  }

  const removeFromPlan = (kolId: string) => {
    setPlanKOLs(prev => prev.filter(k => k.id !== kolId))
  }

  const clearPlan = () => setPlanKOLs([])

  return (
    <KOLPlanContext.Provider value={{
      planKOLs,
      addToPlan,
      removeFromPlan,
      clearPlan,
      planCount: planKOLs.length
    }}>
      {children}
    </KOLPlanContext.Provider>
  )
}

export function useKOLPlan() {
  const context = useContext(KOLPlanContext)
  if (!context) {
    throw new Error('useKOLPlan must be used within KOLPlanProvider')
  }
  return context
}