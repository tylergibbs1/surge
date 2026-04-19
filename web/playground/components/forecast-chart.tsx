"use client"

import { useEffect, useMemo, useState } from "react"
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
import useSWR from "swr"

import {
  ChartContainer,
  type ChartConfig,
} from "@/components/ui/chart"
import type { ForecastResponse } from "@/lib/types"
import { BA_UTC_OFFSET, type BaCode } from "@/lib/us-grid-geo"

const chartConfig = {
  load:     { label: "Actual load",       color: "var(--chart-1)" },
  median:   { label: "Forecast (median)", color: "var(--chart-1)" },
  band:     { label: "80% probability",   color: "var(--chart-1)" },
  temp:     { label: "Temperature",       color: "var(--chart-4)" },
} satisfies ChartConfig

// Actuals window we ask the API for. 48 h is enough context that a reader
// can see a full weekend-vs-weekday rhythm running into the forecast
// without pushing the forecast window off the right edge.
const ACTUALS_HOURS = 48

type ActualsResponse = {
  ba: string
  as_of_utc: string
  hours: number
  units: string
  points: Array<{ ts_utc: string; load_mw: number }>
}

type Row = {
  ts_ms: number
  // Actuals (defined for historical rows only).
  load?: number
  // Forecast (defined for forecast rows only).
  median?: number
  lo?: number
  hi?: number
  bandHeight?: number
  localHour: number
}

type TempRow = {
  ts_ms: number
  tempC: number
}

function toLocalHour(ms: number, utcOffset: number): number {
  const d = new Date(ms)
  const h = d.getUTCHours() + utcOffset
  return ((h % 24) + 24) % 24
}

function buildRows(
  fc: ForecastResponse,
  actuals: ActualsResponse | undefined,
  ba: BaCode,
): { rows: Row[]; forecastStartMs: number; lastActualMs: number | null } {
  const off = BA_UTC_OFFSET[ba] ?? -5
  const rows: Row[] = []
  const forecastFirst = fc.points[0]
  const forecastStartMs = forecastFirst ? Date.parse(forecastFirst.ts_utc) : 0

  if (actuals) {
    // Dedupe on ts_ms defensively — duplicate timestamps (from a cached
    // response or a stale upstream dedupe miss) crash Recharts with
    // "two children with the same key" on XAxis tick generation.
    const seen = new Set<number>()
    for (const a of actuals.points) {
      const ts_ms = Date.parse(a.ts_utc)
      // Trim any actuals that overlap with the forecast window — the
      // forecast is the authority beyond its start.
      if (ts_ms >= forecastStartMs) continue
      if (seen.has(ts_ms)) continue
      seen.add(ts_ms)
      rows.push({
        ts_ms,
        load: a.load_mw / 1000,
        localHour: toLocalHour(ts_ms, off),
      })
    }
  }

  const lastActualMs = rows.length ? rows[rows.length - 1].ts_ms : null

  // Bridge row: stamp the last actual's `median` so the forecast line
  // starts where the actual line ended. Otherwise the dashed line
  // "teleports" up or down between the last observation and the first
  // forecast step.
  if (lastActualMs !== null && forecastFirst) {
    const lastLoadGw = rows[rows.length - 1].load
    rows[rows.length - 1] = {
      ...rows[rows.length - 1],
      median: lastLoadGw,
      lo: lastLoadGw,
      hi: lastLoadGw,
      bandHeight: 0,
    }
  }

  for (const p of fc.points) {
    const ts_ms = Date.parse(p.ts_utc)
    rows.push({
      ts_ms,
      median: p.median_mw / 1000,
      lo: p.p10_mw / 1000,
      hi: p.p90_mw / 1000,
      bandHeight: (p.p90_mw - p.p10_mw) / 1000,
      localHour: toLocalHour(ts_ms, off),
    })
  }
  return { rows, forecastStartMs, lastActualMs }
}

function buildTempRows(fc: ForecastResponse): TempRow[] {
  const out: TempRow[] = []
  for (const p of fc.points) {
    if (p.temp_c == null) continue
    out.push({ ts_ms: Date.parse(p.ts_utc), tempC: p.temp_c })
  }
  return out
}

// Shade overnight hours (local 20:00–06:00) across the combined window.
function nightBands(rows: Row[]): Array<{ x1: number; x2: number }> {
  const bands: Array<{ x1: number; x2: number }> = []
  let start: number | null = null
  for (let i = 0; i < rows.length; i++) {
    const night = rows[i].localHour >= 20 || rows[i].localHour < 6
    if (night && start === null) start = i
    if ((!night || i === rows.length - 1) && start !== null) {
      const end = night ? i : i - 1
      bands.push({ x1: rows[start].ts_ms, x2: rows[end].ts_ms })
      start = null
    }
  }
  return bands
}

function fmtTick(ms: number): string {
  const d = new Date(ms)
  const h = d.getUTCHours()
  // Show day-of-week marker at local midnight boundaries (UTC 00 is a
  // simplification — good enough for a visual x-axis).
  if (h === 0) {
    return d.toLocaleString("en-US", {
      weekday: "short",
      timeZone: "UTC",
    })
  }
  return d.toLocaleString("en-US", {
    hour: "numeric",
    hour12: true,
    timeZone: "UTC",
  })
}

function fmtTooltipTs(ms: number): string {
  return new Date(ms).toLocaleString("en-US", {
    weekday: "short",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
    timeZone: "UTC",
  })
}

function CustomTooltip({
  active, payload,
}: {
  active?: boolean
  payload?: Array<{ payload: Row }>
}) {
  if (!active || !payload?.length) return null
  const r = payload[0].payload
  const isActual = r.load !== undefined && r.median === undefined
  const value = isActual ? r.load : r.median
  return (
    <div className="bg-popover text-popover-foreground rounded-md border p-3 font-mono text-xs shadow-[0_12px_32px_-12px_rgb(0_0_0/0.28)] ring-1 ring-foreground/5 backdrop-blur">
      <div className="flex items-baseline gap-2">
        <span className="text-foreground text-lg font-semibold tracking-tight tabular-nums">
          {value?.toFixed(1) ?? "—"} GW
        </span>
        <span className="text-muted-foreground text-[10px] uppercase tracking-wide">
          {isActual ? "actual" : "forecast"}
        </span>
      </div>
      {!isActual && r.lo !== undefined && r.hi !== undefined ? (
        <div className="text-muted-foreground mt-0.5 tabular-nums">
          80% range: {r.lo.toFixed(1)}–{r.hi.toFixed(1)} GW
        </div>
      ) : null}
      <div className="text-muted-foreground mt-1.5 border-t pt-1.5 tabular-nums">
        {fmtTooltipTs(r.ts_ms)} UTC
      </div>
    </div>
  )
}

// Tick the clock once a minute so the "now" indicator slides across the
// chart without a full page refresh.
function useNowTick(periodMs = 60_000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), periodMs)
    return () => window.clearInterval(id)
  }, [periodMs])
  return now
}

const actualsFetcher = async (url: string): Promise<ActualsResponse> => {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`actuals ${r.status}`)
  return r.json()
}

export function ForecastChart({
  forecast,
  ba,
}: {
  forecast: ForecastResponse
  ba: BaCode
}) {
  // Historical actuals run into the forecast start. On fetch error the
  // chart just renders forecast-only — better than blocking the whole
  // component on an extra network hop.
  const { data: actuals } = useSWR<ActualsResponse>(
    `/api/actuals?ba=${ba}&hours=${ACTUALS_HOURS}`,
    actualsFetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 300_000,
      keepPreviousData: true,
    },
  )

  const nowMs = useNowTick()

  // All heavy derivations live in useMemo so the 60-second now-tick just
  // moves the ReferenceLine — it doesn't rebuild the 200+ row dataset,
  // re-scan for peak, or recompute night bands.
  const { rows, forecastStartMs, lastActualMs } = useMemo(
    () => buildRows(forecast, actuals, ba),
    [forecast, actuals, ba],
  )
  const tempRows = useMemo(() => buildTempRows(forecast), [forecast])
  const peakRow = useMemo<Row | null>(() => {
    let best: Row | null = null
    for (const r of rows) {
      if (r.median === undefined) continue
      if (!best || (best.median !== undefined && r.median > best.median)) {
        best = r
      }
    }
    return best
  }, [rows])
  const bands = useMemo(() => nightBands(rows), [rows])
  const xDomain = useMemo<[number, number]>(
    () =>
      rows.length > 0
        ? [rows[0].ts_ms, rows[rows.length - 1].ts_ms]
        : [0, 1],
    [rows],
  )

  if (rows.length === 0) return null

  // Only surface the "now" marker when it falls inside the rendered
  // window — otherwise it'd be stuck against the axis edge.
  const nowInRange = nowMs >= xDomain[0] && nowMs <= xDomain[1]

  return (
    <div className="space-y-2">
      <ChartContainer config={chartConfig} className="h-[360px] w-full">
        <ComposedChart
          data={rows}
          syncId="surge-forecast"
          margin={{ top: 20, right: 16, left: 4, bottom: 4 }}
        >
          <defs>
            <linearGradient id="forecastFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="var(--color-median)" stopOpacity={0.40} />
              <stop offset="60%"  stopColor="var(--color-median)" stopOpacity={0.15} />
              <stop offset="100%" stopColor="var(--color-median)" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="bandGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="var(--color-band)" stopOpacity={0.22} />
              <stop offset="100%" stopColor="var(--color-band)" stopOpacity={0.04} />
            </linearGradient>
            <linearGradient id="actualFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="var(--color-load)" stopOpacity={0.28} />
              <stop offset="100%" stopColor="var(--color-load)" stopOpacity={0.02} />
            </linearGradient>
          </defs>

          {bands.map((b, i) => (
            <ReferenceArea
              key={`night-${i}`}
              x1={b.x1}
              x2={b.x2}
              fill="var(--foreground)"
              fillOpacity={0.03}
              stroke="none"
              ifOverflow="extendDomain"
            />
          ))}

          <CartesianGrid strokeDasharray="2 4" stroke="var(--border)" opacity={0.4} vertical={false} />

          <XAxis
            dataKey="ts_ms"
            type="number"
            scale="time"
            domain={xDomain}
            axisLine={false}
            tickLine={false}
            tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
            tickMargin={10}
            tickFormatter={fmtTick}
            minTickGap={40}
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

          <Tooltip
            cursor={{ stroke: "var(--border)", strokeDasharray: "3 3" }}
            content={<CustomTooltip />}
          />

          {/* 80% uncertainty band — invisible floor + visible delta above.
              Only renders where bandHeight is defined (forecast rows). */}
          <Area
            yAxisId="load"
            type="linear"
            dataKey="lo"
            stackId="band"
            stroke="none"
            fill="transparent"
            isAnimationActive={false}
            connectNulls={false}
          />
          <Area
            yAxisId="load"
            type="linear"
            dataKey="bandHeight"
            stackId="band"
            stroke="none"
            fill="url(#bandGradient)"
            isAnimationActive={false}
            connectNulls={false}
          />

          {/* Historical actuals — solid line + soft gradient fill. */}
          <Area
            yAxisId="load"
            type="linear"
            dataKey="load"
            stroke="var(--color-load)"
            strokeWidth={2.2}
            fill="url(#actualFill)"
            dot={false}
            activeDot={{ r: 4, strokeWidth: 0, fill: "var(--color-load)" }}
            isAnimationActive={false}
            connectNulls={false}
          />

          {/* Forecast median — dashed line, no fill (the band renders the
              uncertainty; doubling up with a hero fill would be noisy). */}
          <Line
            yAxisId="load"
            type="linear"
            dataKey="median"
            stroke="var(--color-median)"
            strokeWidth={2.2}
            strokeDasharray="5 4"
            dot={false}
            activeDot={{ r: 4, strokeWidth: 0, fill: "var(--color-median)" }}
            isAnimationActive={false}
            connectNulls={false}
          />

          {/* Forecast-start divider. The canonical "actuals end / forecast
              begins" visual break the skill calls for. */}
          {lastActualMs !== null ? (
            <ReferenceLine
              x={forecastStartMs}
              stroke="var(--foreground)"
              strokeOpacity={0.45}
              strokeDasharray="4 4"
              strokeWidth={1}
              label={{
                value: "forecast",
                position: "insideTopRight",
                fill: "var(--muted-foreground)",
                fontSize: 10,
                fontWeight: 500,
                offset: 6,
              }}
            />
          ) : null}

          {/* Peak marker on the forecast side. */}
          {peakRow && peakRow.median !== undefined ? (
            <ReferenceDot
              yAxisId="load"
              x={peakRow.ts_ms}
              y={peakRow.median}
              r={4}
              fill="var(--color-median)"
              stroke="var(--background)"
              strokeWidth={2}
              label={{
                value: `${peakRow.median.toFixed(1)} GW`,
                position: "top",
                fill: "var(--foreground)",
                fontSize: 11,
                fontWeight: 600,
                offset: 10,
                style: { fontVariantNumeric: "tabular-nums" },
              }}
            />
          ) : null}

          {/* Live "now" indicator — only drawn when within the window. */}
          {nowInRange ? (
            <ReferenceLine
              x={nowMs}
              stroke="var(--foreground)"
              strokeOpacity={0.25}
              strokeDasharray="2 4"
              strokeWidth={1}
              label={{
                value: "now",
                position: "insideTop",
                fill: "var(--muted-foreground)",
                fontSize: 10,
                fontWeight: 500,
                offset: 6,
              }}
            />
          ) : null}
        </ComposedChart>
      </ChartContainer>

      {/* Temperature small multiple — aligned x-axis via syncId so the
          tooltip crosshair moves in lockstep. Sits below the load panel
          rather than sharing a right-side y-axis (dual axes are bad). */}
      {tempRows.length > 0 ? (
        <ChartContainer config={chartConfig} className="h-[90px] w-full">
          <ComposedChart
            data={tempRows}
            syncId="surge-forecast"
            margin={{ top: 4, right: 16, left: 4, bottom: 4 }}
          >
            <CartesianGrid strokeDasharray="2 4" stroke="var(--border)" opacity={0.3} vertical={false} />
            <XAxis
              dataKey="ts_ms"
              type="number"
              scale="time"
              domain={xDomain}
              axisLine={false}
              tickLine={false}
              tick={false}
              height={2}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              width={44}
              tick={{ fill: "var(--color-temp)", fontSize: 10, opacity: 0.8 }}
              tickFormatter={(v: number) => `${v.toFixed(0)}°`}
              label={{
                value: "°C", position: "insideLeft", offset: 8,
                style: { fill: "var(--muted-foreground)", fontSize: 10, textAnchor: "start" },
              }}
            />
            <Line
              type="linear"
              dataKey="tempC"
              stroke="var(--color-temp)"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ChartContainer>
      ) : null}
    </div>
  )
}
