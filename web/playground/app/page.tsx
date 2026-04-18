"use client"

import { useCallback, useEffect, useState } from "react"

import { ForecastChart } from "@/components/forecast-chart"
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

export default function Page() {
  const [ba, setBa] = useState<string>("PJM")
  const [horizon, setHorizon] = useState<number>(24)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)

  const fetchForecast = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const r = await fetch(`/api/forecast?ba=${ba}&horizon=${horizon}`)
      if (!r.ok) {
        const body = await r.json().catch(() => ({}))
        throw new Error(body.detail ?? body.error ?? `HTTP ${r.status}`)
      }
      setForecast(await r.json())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [ba, horizon])

  useEffect(() => {
    fetchForecast()
  }, [fetchForecast])

  const peak = forecast ? Math.max(...forecast.points.map((p) => p.median_mw)) : null
  const peakPt = forecast?.points.find((p) => p.median_mw === peak)
  const mean = forecast
    ? forecast.points.reduce((s, p) => s + p.median_mw, 0) / forecast.points.length
    : null

  return (
    <div className="min-h-svh bg-background p-6 md:p-10">
      <div className="mx-auto max-w-5xl space-y-6">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold tracking-tight">Surge playground</h1>
          <p className="text-muted-foreground max-w-3xl">
            Open probabilistic day-ahead load forecasts for 7 US balancing
            authorities. Pick a grid, pick a horizon, see the forecast. Model:
            Chronos-2 fine-tuned on 7 years of public data. Test MASE 0.45,
            ~2% MAPE — matching what utilities pay tens of thousands per year
            for.
          </p>
        </header>

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
                <Select value={ba} onValueChange={setBa}>
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
                <label className="text-muted-foreground text-xs font-medium uppercase">
                  Forecast horizon — {horizon} hours ({(horizon / 24).toFixed(1)} days)
                </label>
                <Slider
                  min={1}
                  max={168}
                  step={1}
                  value={[horizon]}
                  onValueChange={(v) => setHorizon(v[0])}
                />
              </div>
            </div>

            <div className="flex items-center justify-between">
              <p className="text-muted-foreground text-xs">
                {forecast && (
                  <>
                    Forecast starts{" "}
                    <span className="font-mono">
                      {new Date(forecast.points[0]?.ts_utc).toISOString().slice(0, 16)}Z
                    </span>{" "}
                    · {horizon} hourly steps · model{" "}
                    <span className="font-mono">{forecast.model}</span>
                  </>
                )}
                {error && <span className="text-destructive">error: {error}</span>}
              </p>
              <Button
                onClick={fetchForecast}
                disabled={loading}
                size="sm"
                variant="outline"
              >
                {loading ? "loading…" : "Refresh"}
              </Button>
            </div>

            {forecast && <ForecastChart forecast={forecast} />}

            {forecast && (
              <div className="text-muted-foreground grid grid-cols-2 gap-4 border-t pt-4 text-sm md:grid-cols-3">
                <div>
                  <div className="text-xs uppercase">Peak</div>
                  <div className="font-mono text-base text-foreground">
                    {peak ? (peak / 1000).toFixed(1) : "—"} GW
                  </div>
                  {peakPt && (
                    <div className="text-xs">
                      at {new Date(peakPt.ts_utc).toISOString().slice(11, 16)}Z
                    </div>
                  )}
                </div>
                <div>
                  <div className="text-xs uppercase">Mean</div>
                  <div className="font-mono text-base text-foreground">
                    {mean ? (mean / 1000).toFixed(1) : "—"} GW
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase">PI width</div>
                  <div className="font-mono text-base text-foreground">
                    {forecast.points[0]
                      ? (
                          ((forecast.points[0].p90_mw - forecast.points[0].p10_mw) /
                            forecast.points[0].median_mw) *
                          100
                        ).toFixed(1)
                      : "—"}
                    %
                  </div>
                  <div className="text-xs">at step 1</div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <footer className="text-muted-foreground text-xs">
          <p>
            Research and reference use only. Not for trading, regulated bidding,
            or bankability-graded decisions.{" "}
            <a
              href="https://github.com/tylergibbs1/surge"
              className="underline"
              target="_blank"
              rel="noreferrer"
            >
              Code on GitHub
            </a>
            .
          </p>
        </footer>
      </div>
    </div>
  )
}
