"use client"

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import type { ForecastResponse } from "@/lib/types"

type Row = {
  t: string
  hour: string
  median: number
  lo: number
  hi: number
  bandHi: number
}

function buildRows(fc: ForecastResponse): Row[] {
  return fc.points.map((p) => {
    const d = new Date(p.ts_utc)
    return {
      t: d.toISOString(),
      hour: d.toLocaleString("en-US", {
        weekday: "short",
        hour: "numeric",
        timeZone: "UTC",
      }),
      median: p.median_mw / 1000,
      lo: p.p10_mw / 1000,
      hi: p.p90_mw / 1000,
      bandHi: (p.p90_mw - p.p10_mw) / 1000,
    }
  })
}

export function ForecastChart({ forecast }: { forecast: ForecastResponse }) {
  const rows = buildRows(forecast)
  return (
    <div className="h-[360px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="band" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#60a5fa" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#60a5fa" stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis
            dataKey="hour"
            tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
            interval={Math.max(0, Math.floor(rows.length / 8) - 1)}
          />
          <YAxis
            tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
            label={{
              value: "GW",
              position: "insideLeft",
              style: { fill: "hsl(var(--muted-foreground))", fontSize: 11 },
            }}
            width={50}
          />
          <Tooltip
            contentStyle={{
              background: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value, name) => {
              const n = typeof name === "string" ? name : String(name)
              const label =
                n === "median" ? "Median"
                  : n === "lo" ? "10th %ile"
                  : n === "hi" ? "90th %ile"
                  : n
              const v = typeof value === "number" ? value.toFixed(2) : String(value)
              return [`${v} GW`, label]
            }}
          />
          <Area type="monotone" dataKey="hi" stackId="1" stroke="none" fill="url(#band)" isAnimationActive={false} />
          <Area type="monotone" dataKey="lo" stackId="1" stroke="none" fill="hsl(var(--background))" isAnimationActive={false} />
          <Line type="monotone" dataKey="median" stroke="#60a5fa" strokeWidth={2} dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
