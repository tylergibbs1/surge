// Grid view — 53 BA cards with filter sidebar, powered by the daily
// forecasts/all.json bake. Rendered server-side so the first paint ships
// with all data; no loading spinner on cold visits.

import { headers } from "next/headers"
import Link from "next/link"

import { BaGrid } from "@/components/ba-grid"
import { Glossary } from "@/components/glossary"

type ForecastPoint = {
  ts_utc: string
  median_mw: number
  p10_mw: number
  p90_mw: number
}
type Forecast = {
  ba: string
  model: string
  as_of_utc: string
  context_start_utc: string
  context_end_utc: string
  horizon: number
  units: string
  points: ForecastPoint[]
}
type AllPayload = {
  baked_at: string
  horizon: number
  forecasts: Forecast[]
}

// Route-segment config: opt into ISR so the first request after a bake
// invalidates; subsequent requests hit the Next.js data cache for free.
export const revalidate = 300

async function fetchAll(): Promise<AllPayload | null> {
  const h = await headers()
  const host = h.get("host") ?? "localhost:3000"
  // Protocol: vercel always sets x-forwarded-proto; local dev is http.
  const proto = h.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https")
  try {
    const r = await fetch(`${proto}://${host}/api/forecast-all`, {
      next: { revalidate: 300 },
    })
    if (!r.ok) return null
    return (await r.json()) as AllPayload
  } catch {
    return null
  }
}

export default async function GridPage() {
  const data = await fetchAll()

  return (
    <div className="bg-background min-h-svh p-6 md:p-10">
      <div className="mx-auto max-w-7xl space-y-6">
        <header className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-2">
            <h1 className="text-3xl font-semibold tracking-tight">
              Surge grid
            </h1>
            <p className="max-w-2xl text-muted-foreground">
              Day-ahead load forecasts for every EIA-930 demand balancing
              authority — ranked by utilisation, filterable by interconnection
              and size. Click a card to open its map-and-chart detail.
            </p>
          </div>
          <nav className="flex overflow-hidden rounded-full bg-foreground/5 p-1 text-xs font-medium ring-1 ring-foreground/10">
            <Link
              href="/"
              className="rounded-full px-3 py-1.5 text-muted-foreground transition hover:text-foreground"
            >
              Map
            </Link>
            <Link
              href="/grid"
              className="rounded-full bg-background px-3 py-1.5 shadow-sm"
              aria-current="page"
            >
              Grid
            </Link>
          </nav>
        </header>

        {data ? (
          <BaGrid forecasts={data.forecasts} bakedAt={data.baked_at} />
        ) : (
          <div className="rounded-xl border border-dashed border-foreground/15 p-10 text-center">
            <p className="text-sm text-muted-foreground">
              Baked forecasts aren&apos;t available yet. The daily bake at
              06:15 UTC populates this view — until it runs, or if the Blob
              store isn&apos;t linked, the grid stays empty.
            </p>
            <Link
              href="/"
              className="mt-4 inline-block text-sm font-medium underline-offset-4 hover:underline"
            >
              → use the live map instead
            </Link>
          </div>
        )}

        <Glossary />

        <footer className="space-y-2 text-xs text-muted-foreground">
          <p className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <a
              href="https://github.com/tylergibbs1/surge"
              className="underline-offset-4 hover:underline"
              target="_blank"
              rel="noreferrer"
            >
              github.com/tylergibbs1/surge
            </a>
            <span aria-hidden="true" className="text-foreground/20">·</span>
            <a
              href="https://huggingface.co/Tylerbry1/surge-fm-v3"
              className="underline-offset-4 hover:underline"
              target="_blank"
              rel="noreferrer"
            >
              huggingface.co/Tylerbry1/surge-fm-v3
            </a>
          </p>
          <p>
            Research and reference use only. Not for trading, regulated
            bidding, or bankability-graded decisions.
          </p>
        </footer>
      </div>
    </div>
  )
}
