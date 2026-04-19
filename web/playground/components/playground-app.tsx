"use client"

import dynamic from "next/dynamic"
import { useState } from "react"

import { Playground } from "@/components/playground"
import type { BaCode } from "@/lib/us-grid-geo"

// MapLibre GL ships ~500 KB gzipped — dynamic-import so the chart card
// renders before the map chunk finishes loading.
const GridMap = dynamic(
  () => import("@/components/grid-map").then((m) => m.GridMap),
  {
    ssr: false,
    loading: () => (
      <div className="border-border bg-muted/30 h-[420px] w-full animate-pulse rounded-lg border" />
    ),
  },
)

export function PlaygroundApp() {
  const [ba, setBa] = useState<BaCode>("PJM")
  const [horizon, setHorizon] = useState<number>(24)

  return (
    <div className="space-y-6">
      <GridMap horizon={horizon} selected={ba} onSelect={setBa} />
      <Playground
        ba={ba}
        onBaChange={setBa}
        horizon={horizon}
        onHorizonChange={setHorizon}
      />
    </div>
  )
}
