"use client"

import dynamic from "next/dynamic"
import { useEffect, useMemo, useState } from "react"
import { preload } from "swr"
import useSWRImmutable from "swr/immutable"

import { swrFetcher } from "@/components/swr-provider"

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

// CSV export of the currently-rendered forecast slice. Spreadsheets need
// a quoted string for the timestamp (Excel mangles ISO-without-quotes into
// its own date format); numbers stay numeric so formulas work out of the
// box. temp_c is blank when the API didn't ship one rather than "null",
// which Excel would import as the literal string.
function toCsv(fc: ForecastResponse): string {
  const header = "ts_utc,median_mw,p10_mw,p90_mw,temp_c"
  const rows = fc.points.map((p) => {
    const t = p.temp_c == null ? "" : p.temp_c.toFixed(2)
    return `"${p.ts_utc}",${p.median_mw},${p.p10_mw},${p.p90_mw},${t}`
  })
  return [header, ...rows].join("\n") + "\n"
}

function downloadCsv(fc: ForecastResponse): void {
  const csv = toCsv(fc)
  const asOf = fc.as_of_utc.slice(0, 13).replace(/[-:]/g, "") // 20260419T20
  const fname = `surge-${fc.ba}-${asOf}Z-h${fc.horizon}.csv`
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = fname
  a.rel = "noopener"
  document.body.appendChild(a)
  a.click()
  a.remove()
  // Revoke on next tick — immediate revoke races the download on some
  // browsers (Safari in particular).
  setTimeout(() => URL.revokeObjectURL(url), 0)
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
  // Immutable because the daily bake publishes once at ~06:15 UTC — no
  // value in revalidating mid-session. Global fetcher + defaults come
  // from <SwrProvider>.
  const { data: full, error } = useSWRImmutable<ForecastResponse>(
    `/api/forecast?ba=${ba}&horizon=${MAX_HORIZON}`,
  )

  // Kill the parent→child waterfall: fire the two child-chart fetches
  // (actuals + EIA DF) the moment `ba` is known, without waiting for
  // the lazy-loaded Recharts chunk to mount ForecastChart. By the time
  // the chart calls useSWRImmutable on the same keys, the responses
  // are already in SWR's cache — the hook is a synchronous read.
  useEffect(() => {
    preload(`/api/actuals?ba=${ba}&hours=48`, swrFetcher)
    preload(`/api/eia-forecast?ba=${ba}&hours=168`, swrFetcher)
  }, [ba])

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

  // Peak across the FULL 168h window, independent of slider position. This
  // answers "when does the grid hit peak this week?" even when the user has
  // the horizon scoped to 24h — otherwise that number only ever reflects
  // the visible slice.
  const weekPeak = useMemo(() => {
    if (!full || full.points.length <= 24) return null
    let peak = -Infinity
    let peakPt = full.points[0]
    for (const p of full.points) {
      if (p.median_mw > peak) { peak = p.median_mw; peakPt = p }
    }
    return { peak, peakPt }
  }, [full])

  const [copied, setCopied] = useState(false)
  const onCopyShareLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard API can fail on insecure contexts / older Safari. The
      // URL is already in the address bar — silently dropping the feedback
      // is the least-surprising fallback.
    }
  }

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
          <div className="text-muted-foreground grid grid-cols-2 gap-4 border-t pt-4 text-sm tabular-nums md:grid-cols-4">
            <div>
              <div className="text-xs uppercase">Peak (window)</div>
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
            {weekPeak ? (
              <div>
                <div className="text-xs uppercase">Peak (next 7d)</div>
                <div className="text-foreground font-mono text-base">
                  {(weekPeak.peak / 1000).toFixed(1)} GW
                </div>
                <div className="text-xs">
                  {new Date(weekPeak.peakPt.ts_utc).toLocaleString("en-US", {
                    weekday: "short",
                    hour: "numeric",
                    hour12: true,
                    timeZone: "UTC",
                  })} UTC
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        {data ? (
          <div className="flex flex-wrap items-center gap-2 pt-1 text-[11px]">
            <button
              type="button"
              onClick={() => downloadCsv(data)}
              className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1.5 rounded-md bg-foreground/[0.03] px-2.5 py-1.5 uppercase tracking-wider ring-1 ring-foreground/5 transition-colors duration-150 hover:bg-foreground/[0.06] focus:outline-none focus-visible:ring-2 focus-visible:ring-foreground/40 active:scale-[0.97]"
              aria-label={`Download ${horizon}-hour forecast as CSV`}
            >
              <svg aria-hidden="true" viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M8 2v8.5M4.5 7 8 10.5 11.5 7M2.5 13.5h11" />
              </svg>
              Download CSV
            </button>
            <button
              type="button"
              onClick={onCopyShareLink}
              className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1.5 rounded-md bg-foreground/[0.03] px-2.5 py-1.5 uppercase tracking-wider ring-1 ring-foreground/5 transition-colors duration-150 hover:bg-foreground/[0.06] focus:outline-none focus-visible:ring-2 focus-visible:ring-foreground/40 active:scale-[0.97]"
              aria-label="Copy share link"
            >
              {copied ? (
                <>
                  <svg aria-hidden="true" viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M3 8.5 6.5 12l7-7.5" />
                  </svg>
                  Copied
                </>
              ) : (
                <>
                  <svg aria-hidden="true" viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M6.5 9.5a2 2 0 0 0 2.83 0l2.83-2.83a2 2 0 0 0-2.83-2.83l-.71.71M9.5 6.5a2 2 0 0 0-2.83 0L3.84 9.34a2 2 0 0 0 2.83 2.83l.71-.71" />
                  </svg>
                  Share link
                </>
              )}
            </button>
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
