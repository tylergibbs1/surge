"use client"

import dynamic from "next/dynamic"
import { useMemo } from "react"
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

// We always request the maximum horizon (168 h) from the inference layer
// so the slider just truncates a pre-computed series — no re-inference on
// drag. Effective result: ONE model call per BA per hour-ish (thanks to
// SWR dedup + the upstream 5-min Cache-Control).
const MAX_HORIZON = 168

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

  // Single fetch at max horizon; slider just slices. SWR's key stays
  // stable regardless of slider value so rapid drags don't even stream.
  const { data: full, error, isLoading, mutate } = useSWR<ForecastResponse>(
    `/api/forecast?ba=${ba}&horizon=${MAX_HORIZON}`,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 300_000,      // matches API Cache-Control
      keepPreviousData: true,
    },
  )

  // Truncate locally to the user's selected horizon — zero network calls.
  const data = useMemo<ForecastResponse | null>(() => {
    if (!full) return null
    if (horizon >= full.points.length) return full
    return { ...full, horizon, points: full.points.slice(0, horizon) }
  }, [full, horizon])

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
