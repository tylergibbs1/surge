"use client"

import { useEffect, useState } from "react"
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceDot,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import {
  ChartContainer,
  type ChartConfig,
} from "@/components/ui/chart"
import type { ForecastResponse } from "@/lib/types"
import { BA_UTC_OFFSET, type BaCode } from "@/lib/us-grid-geo"

const chartConfig = {
  median: { label: "Forecast", color: "var(--chart-1)" },
  band:   { label: "80% probability", color: "var(--chart-1)" },
  temp:   { label: "Temperature", color: "var(--chart-4)" },
} satisfies ChartConfig

type Row = {
  ts: string
  hourLabel: string
  localHour: number
  median: number
  lo: number
  hi: number
  bandHeight: number
  tempC: number | null
}

function toLocalHour(isoTs: string, utcOffset: number): number {
  const d = new Date(isoTs)
  const h = d.getUTCHours() + utcOffset
  return ((h % 24) + 24) % 24
}

function buildRows(fc: ForecastResponse, ba: BaCode): Row[] {
  const off = BA_UTC_OFFSET[ba] ?? -5
  return fc.points.map((p) => {
    const d = new Date(p.ts_utc)
    const localHour = toLocalHour(p.ts_utc, off)
    return {
      ts: d.toISOString(),
      hourLabel: d.toLocaleString("en-US", {
        weekday: "short",
        hour: "numeric",
        hour12: true,
        timeZone: "UTC",
      }),
      localHour,
      median: p.median_mw / 1000,
      lo: p.p10_mw / 1000,
      hi: p.p90_mw / 1000,
      bandHeight: (p.p90_mw - p.p10_mw) / 1000,
      tempC: p.temp_c ?? null,
    }
  })
}

// Group consecutive nighttime rows (local hour ≥ 20 or < 6) into reference
// areas for the x-axis. Indexes refer to the row positions, which is what
// Recharts' ReferenceArea needs when the XAxis uses categorical data.
function nightBands(rows: Row[]): Array<{ x1: string; x2: string }> {
  const bands: Array<{ x1: string; x2: string }> = []
  let start: number | null = null
  for (let i = 0; i < rows.length; i++) {
    const night = rows[i].localHour >= 20 || rows[i].localHour < 6
    if (night && start === null) start = i
    if ((!night || i === rows.length - 1) && start !== null) {
      const end = night ? i : i - 1
      bands.push({ x1: rows[start].hourLabel, x2: rows[end].hourLabel })
      start = null
    }
  }
  return bands
}

function fmtRelative(fromIso: string): string {
  const ms = Date.now() - new Date(fromIso).getTime()
  if (ms < 60_000) return "just now"
  const m = Math.round(ms / 60_000)
  if (m < 60) return `${m}m ago`
  const h = Math.round(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.round(h / 24)}d ago`
}

function CustomTooltip({
  active, payload,
}: {
  active?: boolean
  payload?: Array<{ payload: Row }>
}) {
  if (!active || !payload?.length) return null
  const r = payload[0].payload
  return (
    <div className="bg-popover text-popover-foreground rounded-md border p-3 font-mono text-xs shadow-lg backdrop-blur">
      <div className="text-foreground text-lg font-semibold tracking-tight tabular-nums">
        {r.median.toFixed(1)} GW
      </div>
      <div className="text-muted-foreground mt-0.5 tabular-nums">
        80% range: {r.lo.toFixed(1)}–{r.hi.toFixed(1)} GW
      </div>
      {r.tempC !== null && (
        <div className="text-muted-foreground mt-0.5 tabular-nums">
          temp: {r.tempC.toFixed(1)}°C
        </div>
      )}
      <div className="text-muted-foreground mt-1.5 border-t pt-1.5">
        {r.hourLabel} UTC
      </div>
    </div>
  )
}

// Tick the clock once a minute so the "now" indicator slides across the
// chart without a full page refresh. Returns the current epoch-ms.
function useNowTick(periodMs = 60_000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), periodMs)
    return () => window.clearInterval(id)
  }, [periodMs])
  return now
}

// Find the row closest to `nowMs`. Returns null if every row is outside
// the chart window (e.g. the baked forecast is older than it shows — now
// already marched off the right edge).
function findNowRow(
  rows: Row[],
  nowMs: number,
): { row: Row; index: number } | null {
  let bestIdx = -1
  let bestDist = Infinity
  for (let i = 0; i < rows.length; i++) {
    const d = Math.abs(Date.parse(rows[i].ts) - nowMs)
    if (d < bestDist) {
      bestDist = d
      bestIdx = i
    }
  }
  if (bestIdx === -1) return null
  // Only render the marker if "now" is within ~45 min of a chart tick;
  // anything further means we're drawing a stale or out-of-range chart
  // and the dot would be misleading.
  if (bestDist > 45 * 60_000) return null
  return { row: rows[bestIdx], index: bestIdx }
}

export function ForecastChart({
  forecast,
  ba,
}: {
  forecast: ForecastResponse
  ba: BaCode
}) {
  const rows = buildRows(forecast, ba)
  const nowMs = useNowTick()
  if (rows.length === 0) return null

  // Single-pass stats for the peak marker.
  let peak = rows[0]
  for (const r of rows) if (r.median > peak.median) peak = r

  const nowHit = findNowRow(rows, nowMs)

  const tickEvery = Math.max(1, Math.floor(rows.length / 8))
  const bands = nightBands(rows)
  const hasTemp = rows.some((r) => r.tempC !== null)

  // Temperature axis bounds — pad a little for readability.
  let tMin = Infinity, tMax = -Infinity
  for (const r of rows) if (r.tempC !== null) {
    if (r.tempC < tMin) tMin = r.tempC
    if (r.tempC > tMax) tMax = r.tempC
  }
  const tPad = Math.max(2, (tMax - tMin) * 0.15)

  return (
    <ChartContainer config={chartConfig} className="h-[380px] w-full">
      <ComposedChart data={rows} margin={{ top: 20, right: hasTemp ? 48 : 16, left: 4, bottom: 4 }}>
        <defs>
          {/* Area gradient: line color at the top, fading to ~0 at the
              bottom. Vercel-style — the chart has *presence* without the
              hard edge of a stacked band. */}
          <linearGradient id="forecastFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="var(--color-median)" stopOpacity={0.45} />
            <stop offset="60%"  stopColor="var(--color-median)" stopOpacity={0.18} />
            <stop offset="100%" stopColor="var(--color-median)" stopOpacity={0.02} />
          </linearGradient>
          <linearGradient id="bandGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="var(--color-band)" stopOpacity={0.22} />
            <stop offset="100%" stopColor="var(--color-band)" stopOpacity={0.04} />
          </linearGradient>
        </defs>

        {/* Night bands — shade overnight hours so the diurnal shape is
            visible at a glance. */}
        {bands.map((b, i) => (
          <ReferenceArea
            key={`night-${i}`}
            x1={b.x1}
            x2={b.x2}
            fill="oklch(0.985 0 0)"
            fillOpacity={0.025}
            stroke="none"
            ifOverflow="extendDomain"
          />
        ))}

        <CartesianGrid strokeDasharray="2 4" stroke="var(--border)" opacity={0.4} vertical={false} />

        <XAxis
          dataKey="hourLabel"
          axisLine={false}
          tickLine={false}
          interval={tickEvery - 1}
          tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
          tickMargin={10}
        />
        <YAxis
          yAxisId="load"
          axisLine={false}
          tickLine={false}
          width={44}
          tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
          tickFormatter={(v: number) => `${v.toFixed(0)}`}
          label={{
            value: "GW", position: "insideLeft", offset: 8,
            style: { fill: "var(--muted-foreground)", fontSize: 11, textAnchor: "start" },
          }}
        />
        {hasTemp && (
          <YAxis
            yAxisId="temp"
            orientation="right"
            axisLine={false}
            tickLine={false}
            width={40}
            domain={[tMin - tPad, tMax + tPad]}
            tick={{ fill: "var(--chart-4)", fontSize: 11, opacity: 0.7 }}
            tickFormatter={(v: number) => `${v.toFixed(0)}°`}
          />
        )}

        <Tooltip
          cursor={{ stroke: "var(--border)", strokeDasharray: "3 3" }}
          content={<CustomTooltip />}
        />

        {/* 80% band: invisible floor at `lo`, visible delta above. */}
        <Area
          yAxisId="load"
          type="monotone"
          dataKey="lo"
          stackId="band"
          stroke="none"
          fill="transparent"
          isAnimationActive={false}
        />
        <Area
          yAxisId="load"
          type="monotone"
          dataKey="bandHeight"
          stackId="band"
          stroke="none"
          fill="url(#bandGradient)"
          isAnimationActive={false}
        />

        {/* Hero gradient area under the median line. */}
        <Area
          yAxisId="load"
          type="monotone"
          dataKey="median"
          stroke="var(--color-median)"
          strokeWidth={2.2}
          fill="url(#forecastFill)"
          dot={false}
          activeDot={{ r: 4, strokeWidth: 0, fill: "var(--color-median)" }}
          isAnimationActive={false}
        />

        {/* Temperature ghost line on secondary axis. */}
        {hasTemp && (
          <Line
            yAxisId="temp"
            type="monotone"
            dataKey="tempC"
            stroke="var(--color-temp)"
            strokeWidth={1.25}
            strokeDasharray="3 3"
            strokeOpacity={0.55}
            dot={false}
            isAnimationActive={false}
          />
        )}

        {/* Peak marker — dot + stacked label. */}
        <ReferenceDot
          yAxisId="load"
          x={peak.hourLabel}
          y={peak.median}
          r={4}
          fill="var(--color-median)"
          stroke="var(--background)"
          strokeWidth={2}
          label={{
            value: `${peak.median.toFixed(1)} GW`,
            position: "top",
            fill: "var(--foreground)",
            fontSize: 11,
            fontWeight: 600,
            offset: 10,
            // SVG <text> — Tailwind's tabular-nums class doesn't apply
            // through Recharts' Label, so set the CSS prop directly.
            style: { fontVariantNumeric: "tabular-nums" },
          }}
        />

        {/* "Now" indicator — vertical guideline + pulsing dot where real
            time intersects the forecast curve. Slides rightward once a
            minute (useNowTick above). Suppressed when the chart is of a
            past/future forecast where "now" falls outside the window. */}
        {nowHit ? (
          <>
            <ReferenceLine
              x={nowHit.row.hourLabel}
              stroke="var(--foreground)"
              strokeOpacity={0.35}
              strokeDasharray="2 4"
              strokeWidth={1}
              ifOverflow="extendDomain"
              label={{
                value: "now",
                position: "insideTop",
                fill: "var(--muted-foreground)",
                fontSize: 10,
                fontWeight: 500,
                offset: 6,
              }}
            />
            <ReferenceDot
              yAxisId="load"
              x={nowHit.row.hourLabel}
              y={nowHit.row.median}
              // Shape override: one static dot + one scaled ping. Keeps
              // the marker distinct from the peak dot (static, no pulse).
              shape={(props: { cx?: number; cy?: number }) => (
                <g>
                  <circle
                    cx={props.cx}
                    cy={props.cy}
                    r={9}
                    fill="var(--color-median)"
                    opacity={0.35}
                    className="surge-now-pulse"
                  />
                  <circle
                    cx={props.cx}
                    cy={props.cy}
                    r={4}
                    fill="var(--color-median)"
                    stroke="var(--background)"
                    strokeWidth={2}
                  />
                </g>
              )}
            />
          </>
        ) : null}
      </ComposedChart>
    </ChartContainer>
  )
}
