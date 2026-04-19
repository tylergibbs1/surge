"use client"

import dynamic from "next/dynamic"
import { useSearchParams } from "next/navigation"
import { useState } from "react"

import { Playground } from "@/components/playground"
import { BAS, type BaCode } from "@/lib/us-grid-geo"

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

const VALID_BAS = new Set<string>(BAS as readonly string[])

export function PlaygroundApp() {
  // Honor ?ba=XXX&horizon=NN from the URL — this is how /grid deep-links
  // into the map view after a card click.
  const params = useSearchParams()
  const baParam = params.get("ba")?.toUpperCase()
  const horizonParam = Number.parseInt(params.get("horizon") ?? "", 10)
  const initialBa: BaCode =
    baParam && VALID_BAS.has(baParam) ? (baParam as BaCode) : "PJM"
  const initialHorizon =
    Number.isInteger(horizonParam) && horizonParam >= 1 && horizonParam <= 168
      ? horizonParam
      : 24

  const [ba, setBa] = useState<BaCode>(initialBa)
  const [horizon, setHorizon] = useState<number>(initialHorizon)

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
