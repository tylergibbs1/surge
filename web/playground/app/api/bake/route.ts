// Daily bake: runs inference for every forecastable BA and writes the
// result to Vercel Blob so the /api/forecast route can serve static JSON
// at CDN latency. Triggered by a GitHub Actions cron (see
// `.github/workflows/bake.yml`) with a shared Bearer token.
//
// Why this lives in Next.js (and not Modal): @vercel/blob's SDK is
// first-party for this runtime. Keeping the Blob integration here means
// we avoid hand-rolling Vercel's REST API in Python and everything
// auth-related stays on one side of the wire.

import { put } from "@vercel/blob"
import { NextRequest } from "next/server"

import { BAS, type BaCode } from "@/lib/us-grid-geo"

const MODAL_URL =
  process.env.SURGE_API_URL ??
  "https://tylergibbs1--surge-api-fastapi-app.modal.run"

const BAKE_SECRET = process.env.BAKE_SECRET
const BLOB_TOKEN = process.env.BLOB_READ_WRITE_TOKEN

// Max horizon we serve. Clients slice this down at read time. 168h = 7d is
// the API's upper bound.
const BAKE_HORIZON = 168

// How many BAs to fetch from Modal in parallel. Modal is configured with
// max_containers=20 and scaledown_window=600, so ~10 concurrent is a sweet
// spot: fast enough to finish in ~30s, gentle enough not to cold-start
// 20 containers simultaneously.
const CONCURRENCY = 10

// Run on the Node runtime (not edge) — @vercel/blob's put() needs Node.
// Default fluid-compute timeout is 300s, plenty for 53 sequential-ish
// Modal calls even with cold starts.
export const runtime = "nodejs"
export const maxDuration = 300

type ForecastResponse = {
  ba: string
  model: string
  as_of_utc: string
  context_start_utc: string
  context_end_utc: string
  horizon: number
  units: string
  points: Array<{
    ts_utc: string
    median_mw: number
    p10_mw: number
    p90_mw: number
    temp_c?: number | null
  }>
}

async function fetchOneForecast(ba: BaCode): Promise<ForecastResponse> {
  // Ask Modal for the max horizon so one bake covers every client request.
  const r = await fetch(
    `${MODAL_URL}/forecast/${encodeURIComponent(ba)}?horizon=${BAKE_HORIZON}`,
    { cache: "no-store" },
  )
  if (!r.ok) throw new Error(`${ba}: upstream ${r.status}`)
  return (await r.json()) as ForecastResponse
}

async function withConcurrency<T>(
  items: readonly T[],
  limit: number,
  work: (item: T) => Promise<unknown>,
): Promise<void> {
  // Hand-rolled p-limit: avoids pulling in a dep for a 10-line helper.
  const queue = items.slice()
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (queue.length) {
      const next = queue.shift()
      if (next !== undefined) await work(next)
    }
  })
  await Promise.all(workers)
}

export async function POST(req: NextRequest): Promise<Response> {
  if (!BAKE_SECRET) {
    return Response.json({ error: "BAKE_SECRET not set on server" }, { status: 500 })
  }
  if (!BLOB_TOKEN) {
    return Response.json(
      { error: "BLOB_READ_WRITE_TOKEN not set on server" },
      { status: 500 },
    )
  }
  // Constant-time-ish compare: Bearer token auth. Anyone who can POST here
  // can trigger a bake, which hits Modal 53× and burns a small amount of
  // compute — not free, so gate it.
  const auth = req.headers.get("authorization") ?? ""
  const provided = auth.startsWith("Bearer ") ? auth.slice(7) : ""
  if (provided !== BAKE_SECRET) {
    return Response.json({ error: "unauthorized" }, { status: 401 })
  }

  const started = Date.now()
  const results: Array<{ ba: string; ok: boolean; url?: string; error?: string }> = []
  const manifest: Record<string, { url: string; as_of_utc: string }> = {}

  await withConcurrency(BAS, CONCURRENCY, async (ba) => {
    try {
      const forecast = await fetchOneForecast(ba)
      const blob = await put(
        `forecasts/${ba}.json`,
        JSON.stringify(forecast),
        {
          // The surge-blob store is configured private, so blobs are served
          // via signed URLs only. The /api/forecast reader proxies the fetch
          // server-side with the read-write token — browsers never see the URL.
          access: "private",
          token: BLOB_TOKEN,
          contentType: "application/json",
          // Stable paths — overwrite yesterday's forecast rather than
          // accumulating files. addRandomSuffix would give each bake a
          // unique URL (wrong for our "latest-wins" semantics).
          addRandomSuffix: false,
          allowOverwrite: true,
          // Edge cache for 30 min; the underlying data only turns over
          // once/day, and stale-while-revalidate is handled by the API
          // route reading this.
          cacheControlMaxAge: 1800,
        },
      )
      manifest[ba] = { url: blob.url, as_of_utc: forecast.as_of_utc }
      results.push({ ba, ok: true, url: blob.url })
    } catch (e) {
      results.push({ ba, ok: false, error: String(e) })
    }
  })

  // Single consolidated manifest so readers can discover the per-BA URLs
  // without maintaining a client-side list. One more blob write, trivial.
  let manifestUrl: string | undefined
  try {
    const m = await put(
      "forecasts/manifest.json",
      JSON.stringify({
        baked_at: new Date().toISOString(),
        horizon: BAKE_HORIZON,
        entries: manifest,
      }),
      {
        access: "private",
        token: BLOB_TOKEN,
        contentType: "application/json",
        addRandomSuffix: false,
        allowOverwrite: true,
        cacheControlMaxAge: 1800,
      },
    )
    manifestUrl = m.url
  } catch (e) {
    results.push({ ba: "__manifest__", ok: false, error: String(e) })
  }

  const ok = results.filter((r) => r.ok).length
  const fail = results.length - ok
  return Response.json(
    {
      elapsed_ms: Date.now() - started,
      ok,
      fail,
      manifest_url: manifestUrl,
      results,
    },
    { status: fail === 0 ? 200 : 207 }, // 207 Multi-Status on partial failure
  )
}
