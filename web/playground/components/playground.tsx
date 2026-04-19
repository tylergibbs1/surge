"use client"

import dynamic from "next/dynamic"
import { useMemo } from "react"
import useSWR from "swr"

import { ErrorBanner } from "@/components/error-banner"
import { ForecastChartSkeleton } from "@/components/forecast-chart-skeleton"
import { BA_LABEL, BA_UTC_OFFSET, type BaCode } from "@/lib/us-grid-geo"
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

function fmtRelative(fromIso: string): string {
  const ms = Date.now() - new Date(fromIso).getTime()
  if (ms < 60_000) return "just now"
  const m = Math.round(ms / 60_000)
  if (m < 60) return `${m}m ago`
  const h = Math.round(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.round(h / 24)}d ago`
}

function fmtLocalPeakTime(tsUtc: string, ba: BaCode): string {
  // Approximate local-time display using the BA's standard-time offset.
  const d = new Date(tsUtc)
  const local = new Date(d.getTime() + BA_UTC_OFFSET[ba] * 3600 * 1000)
  return local.toLocaleString("en-US", {
    weekday: "short",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
    timeZone: "UTC",
  })
}

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
  const { data: full, error } = useSWR<ForecastResponse>(
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
    // Single-pass peak + mean.
    let peak = -Infinity
    let peakPt = data.points[0]
    let sum = 0
    for (const p of data.points) {
      sum += p.median_mw
      if (p.median_mw > peak) { peak = p.median_mw; peakPt = p }
    }
    const mean = sum / data.points.length
    const first = data.points[0]
    const piPct = ((first.p90_mw - first.p10_mw) / first.median_mw) * 100
    // Peak above mean — replaces "vs yesterday" (we don't have yesterday's
    // data on the client) with something just as informative.
    const peakDeltaPct = ((peak - mean) / mean) * 100
    return { peak, peakPt, mean, piPct, peakDeltaPct }
  }, [data])

  // Plain-English takeaway for the card title. Prefers "peak at X
  // tomorrow" when stats are available; falls back to the BA name while
  // data is loading so the title isn't empty.
  const title = stats
    ? `${BA_LABEL[ba]}: peak ${(stats.peak / 1000).toFixed(1)} GW · ${fmtLocalPeakTime(stats.peakPt.ts_utc, ba)} local`
    : `${BA_LABEL[ba]} load forecast`

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>
          Hourly probabilistic load forecast · GW · EIA-930 demand series ·
          model <span className="font-mono" translate="no">{data?.model ?? "—"}</span>
          {data ? (
            <>
              {" "}
              · issued{" "}
              <span className="font-mono tabular-nums">
                {new Date(data.as_of_utc).toISOString().slice(0, 16)}Z
              </span>
            </>
          ) : null}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Hero strip: eyebrow label + peak number + delta + live dot */}
        <div className="border-b pb-3">
          <div className="text-muted-foreground flex items-center justify-between text-[11px] font-medium tracking-wide uppercase">
            <span>
              {BA_LABEL[ba]} · Day-ahead load
            </span>
            <span className="flex items-center gap-1.5 font-mono normal-case tabular-nums">
              <span
                className="bg-chart-1 inline-block size-1.5 rounded-full"
                style={{ boxShadow: "0 0 6px var(--chart-1)" }}
              />
              {data ? `updated ${fmtRelative(data.as_of_utc)}` : "loading…"}
            </span>
          </div>
          <div className="mt-1.5 flex items-baseline gap-3">
            <span className="text-foreground text-4xl font-semibold tracking-tight tabular-nums">
              {stats ? (stats.peak / 1000).toFixed(1) : "—"}
            </span>
            <span className="text-muted-foreground text-sm">GW peak</span>
            {stats ? (
              <>
                <span className="text-muted-foreground">·</span>
                <span className="text-muted-foreground font-mono text-sm tabular-nums">
                  {fmtLocalPeakTime(stats.peakPt.ts_utc, ba)} local
                </span>
                <span className="text-muted-foreground">·</span>
                <span className="font-mono text-sm tabular-nums text-emerald-500">
                  ▲ {stats.peakDeltaPct.toFixed(1)}%
                </span>
                <span className="text-muted-foreground text-xs">vs window mean</span>
              </>
            ) : null}
          </div>
        </div>

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
                    <span className="font-mono" translate="no">{b.code}</span>
                    <span className="text-muted-foreground ml-2 text-xs">
                      {b.name}
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-3 pt-1">
            <label
              id="horizon-label"
              className="text-muted-foreground block text-xs font-medium uppercase tracking-wider tabular-nums"
            >
              Forecast horizon — {horizon} hours ({(horizon / 24).toFixed(1)} days)
            </label>
            <div className="pt-1">
              <Slider
                aria-labelledby="horizon-label"
                min={1}
                max={168}
                step={1}
                value={[horizon]}
                onValueChange={(v) => onHorizonChange(v[0])}
              />
            </div>
          </div>
        </div>

        <p className="text-muted-foreground text-xs">
          {data ? (
            <>
              Forecast starts{" "}
              <span className="font-mono">
                {new Date(data.points[0]?.ts_utc).toISOString().slice(0, 16)}Z
              </span>{" "}
              · {horizon} hourly steps · model{" "}
              <span className="font-mono" translate="no">{data.model}</span>
            </>
          ) : error ? null : (
            "Loading forecast…"
          )}
        </p>

        {error && !data ? (
          <ErrorBanner
            title="Couldn't load forecast"
            detail={error.message}
          />
        ) : null}

        {data ? <ForecastChart forecast={data} ba={ba} /> : <ForecastChartSkeleton />}

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

        {/* Provenance footer: source, assumptions, last-observed. Keeps the
            chart self-explanatory without forcing the reader to dig through
            the repo README. */}
        {data ? (
          <div className="text-muted-foreground border-t pt-3 text-[11px] leading-relaxed">
            <p className="flex flex-wrap gap-x-3 gap-y-1">
              <span>
                Source:{" "}
                <a
                  href="https://www.eia.gov/electricity/gridmonitor/"
                  target="_blank"
                  rel="noreferrer"
                  className="underline-offset-4 hover:underline"
                >
                  EIA-930 hourly demand
                </a>
              </span>
              <span className="text-foreground/20" aria-hidden="true">·</span>
              <span>
                Context ends{" "}
                <span className="font-mono tabular-nums">
                  {new Date(data.context_end_utc).toISOString().slice(0, 16)}Z
                </span>
              </span>
              <span className="text-foreground/20" aria-hidden="true">·</span>
              <span>Raw load — not weather-normalized, net of BTM solar</span>
            </p>
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}
