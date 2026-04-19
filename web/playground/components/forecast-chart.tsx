"use client"

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts"

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import type { ForecastResponse } from "@/lib/types"

const chartConfig = {
  median: {
    label: "Forecast",
    color: "var(--chart-1)",
  },
  band: {
    label: "80% probability",
    color: "var(--chart-1)",
  },
  lo: { label: "10th percentile", color: "var(--chart-3)" },
  hi: { label: "90th percentile", color: "var(--chart-4)" },
} satisfies ChartConfig

type Row = {
  ts: string
  hour: string
  median: number
  lo: number
  hi: number
  bandHeight: number  // hi − lo, stacked on top of lo
}

function buildRows(fc: ForecastResponse): Row[] {
  return fc.points.map((p) => {
    const d = new Date(p.ts_utc)
    return {
      ts: d.toISOString(),
      hour: d.toLocaleString("en-US", {
        weekday: "short",
        hour: "numeric",
        hour12: true,
        timeZone: "UTC",
      }),
      median: p.median_mw / 1000,
      lo: p.p10_mw / 1000,
      hi: p.p90_mw / 1000,
      bandHeight: (p.p90_mw - p.p10_mw) / 1000,
    }
  })
}

export function ForecastChart({ forecast }: { forecast: ForecastResponse }) {
  const rows = buildRows(forecast)
  const tickEvery = Math.max(1, Math.floor(rows.length / 8))

  return (
    <ChartContainer config={chartConfig} className="h-[360px] w-full">
      <ComposedChart data={rows} margin={{ top: 12, right: 16, left: 4, bottom: 4 }}>
        <defs>
          <linearGradient id="forecastBand" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--color-band)" stopOpacity={0.32} />
            <stop offset="100%" stopColor="var(--color-band)" stopOpacity={0.08} />
          </linearGradient>
        </defs>

        <CartesianGrid strokeDasharray="2 4" stroke="var(--border)" opacity={0.5} />
        <XAxis
          dataKey="hour"
          axisLine={false}
          tickLine={false}
          interval={tickEvery - 1}
          tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
          tickMargin={8}
        />
        <YAxis
          axisLine={false}
          tickLine={false}
          width={44}
          tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
          tickFormatter={(v: number) => `${v.toFixed(0)}`}
          label={{
            value: "GW",
            position: "insideLeft",
            offset: 8,
            style: {
              fill: "var(--muted-foreground)",
              fontSize: 11,
              textAnchor: "start",
            },
          }}
        />

        <ChartTooltip
          cursor={{ stroke: "var(--border)", strokeDasharray: "3 3" }}
          content={
            <ChartTooltipContent
              className="font-mono text-xs tabular-nums"
              hideIndicator={false}
              formatter={(value, name) => {
                const n = typeof name === "string" ? name : String(name)
                if (n === "bandHeight") return null
                const v = typeof value === "number" ? value.toFixed(2) : String(value)
                const label =
                  chartConfig[n as keyof typeof chartConfig]?.label ?? n
                return (
                  <div className="flex w-full items-center justify-between gap-4">
                    <span className="text-muted-foreground">{label}</span>
                    <span className="font-medium">{v} GW</span>
                  </div>
                )
              }}
            />
          }
        />

        {/* 80% band — stacked area: invisible floor at `lo`, visible delta at `bandHeight`.
            Hidden from tooltip via formatter returning null. */}
        <Area
          type="monotone"
          dataKey="lo"
          stackId="band"
          stroke="none"
          fill="transparent"
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey="bandHeight"
          stackId="band"
          stroke="none"
          fill="url(#forecastBand)"
          isAnimationActive={false}
        />

        <Line
          type="monotone"
          dataKey="median"
          stroke="var(--color-median)"
          strokeWidth={2.2}
          dot={false}
          activeDot={{ r: 4, strokeWidth: 0, fill: "var(--color-median)" }}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ChartContainer>
  )
}
