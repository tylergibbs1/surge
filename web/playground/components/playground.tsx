"use client"

import dynamic from "next/dynamic"
import { useMemo, useState } from "react"
import useSWR from "swr"

import { ForecastChartSkeleton } from "@/components/forecast-chart-skeleton"
import { RefreshIcon } from "@/components/refresh-icon"
import type { BaCode } from "@/lib/us-grid-geo"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { BAS, type ForecastResponse } from "@/lib/types"

// Recharts + our chart component weigh ~100KB gzipped and are only useful
// after a forecast arrives. Lazy-load; render the skeleton meanwhile.
const ForecastChart = dynamic(
  () => import("@/components/forecast-chart").then((m) => m.ForecastChart),
  { ssr: false, loading: () => <ForecastChartSkeleton /> },
)

async function fetcher(url: string): Promise<ForecastResponse> {
  const r = await fetch(url)
  if (!r.ok) {
    const body = await r.json().catch(() => ({}))
    throw new Error(body.detail ?? body.error ?? `HTTP ${r.status}`)
  }
  return r.json()
}

export function Playground({
  ba,
  onBaChange,
  horizon,
  onHorizonChange,
}: {
  ba: BaCode
  onBaChange: (ba: BaCode) => void
  horizon: number
  onHorizonChange: (h: number) => void
}) {
  // State is now lifted to the page so the map and the chart stay in sync.

  const { data, error, isLoading, mutate } = useSWR<ForecastResponse>(
    `/api/forecast?ba=${ba}&horizon=${horizon}`,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 60_000,
      keepPreviousData: true,
    },
  )

  const stats = useMemo(() => {
    if (!data) return null
    const peak = Math.max(...data.points.map((p) => p.median_mw))
    const peakPt = data.points.find((p) => p.median_mw === peak)!
    const mean = data.points.reduce((s, p) => s + p.median_mw, 0) / data.points.length
    const first = data.points[0]
    const piPct =
      ((first.p90_mw - first.p10_mw) / first.median_mw) * 100
    return { peak, peakPt, mean, piPct }
  }, [data])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Forecast</CardTitle>
        <CardDescription>
          Dashed line is our median forecast, shaded band is the 80% uncertainty
          range. Hover the chart for per-hour values.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <label className="text-muted-foreground text-xs font-medium uppercase">
              Balancing authority
            </label>
            <Select value={ba} onValueChange={(v) => onBaChange(v as BaCode)}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {BAS.map((b) => (
                  <SelectItem key={b.code} value={b.code}>
                    <span className="font-mono">{b.code}</span>
                    <span className="text-muted-foreground ml-2 text-xs">
                      {b.name}
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-muted-foreground text-xs font-medium uppercase tabular-nums">
              Forecast horizon — {horizon} hours ({(horizon / 24).toFixed(1)} days)
            </label>
            <Slider
              min={1}
              max={168}
              step={1}
              value={[horizon]}
              onValueChange={(v) => onHorizonChange(v[0])}
            />
          </div>
        </div>

        <div className="flex items-center justify-between">
          <p className="text-muted-foreground text-xs">
            {data ? (
              <>
                Forecast starts{" "}
                <span className="font-mono">
                  {new Date(data.points[0]?.ts_utc).toISOString().slice(0, 16)}Z
                </span>{" "}
                · {horizon} hourly steps · model{" "}
                <span className="font-mono">{data.model}</span>
              </>
            ) : error ? (
              <span className="text-destructive">error: {error.message}</span>
            ) : (
              "Loading forecast…"
            )}
          </p>
          <Button
            onClick={() => mutate()}
            disabled={isLoading}
            size="sm"
            variant="outline"
            className="gap-2 transition-transform duration-100 active:scale-[0.96]"
          >
            <RefreshIcon loading={isLoading} />
            Refresh
          </Button>
        </div>

        {data ? <ForecastChart forecast={data} /> : <ForecastChartSkeleton />}

        {stats ? (
          <div className="text-muted-foreground grid grid-cols-2 gap-4 border-t pt-4 text-sm tabular-nums md:grid-cols-3">
            <div>
              <div className="text-xs uppercase">Peak</div>
              <div className="text-foreground font-mono text-base">
                {(stats.peak / 1000).toFixed(1)} GW
              </div>
              <div className="text-xs">
                at {new Date(stats.peakPt.ts_utc).toISOString().slice(11, 16)}Z
              </div>
            </div>
            <div>
              <div className="text-xs uppercase">Mean</div>
              <div className="text-foreground font-mono text-base">
                {(stats.mean / 1000).toFixed(1)} GW
              </div>
            </div>
            <div>
              <div className="text-xs uppercase">PI width</div>
              <div className="text-foreground font-mono text-base">
                {stats.piPct.toFixed(1)}%
              </div>
              <div className="text-xs">at step 1</div>
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}
