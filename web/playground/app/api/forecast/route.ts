// Day-ahead forecast API. Two-tier serving:
//
//   1. Baked JSON from Vercel Blob — daily bake writes horizon=168 to
//      forecasts/{BA}.json via /api/bake. When a cache hit lands we slice
//      to the requested horizon and return in ~20 ms from the Vercel edge.
//
//   2. Live inference fallback — Modal (default) or RunPod, used when the
//      baked payload is missing, older than ~25 h, or the client passed
//      ?force=1 (e.g. for a custom horizon beyond what was baked).
//
// Mode is transparent to the client; the `x-surge-backend` response header
// reports which path was used.

import { head } from "@vercel/blob"
import { NextRequest } from "next/server"

import { BAS, type BaCode } from "@/lib/us-grid-geo"

// The bake/read side talks to Vercel Blob via the @vercel/blob SDK; the
// read-write token is auto-injected when the blob store is linked to the
// project. On local dev BLOB_READ_WRITE_TOKEN is undefined — we short-
// circuit the blob path and go straight to live inference.
const BLOB_TOKEN = process.env.BLOB_READ_WRITE_TOKEN

// The Blob route needs the Node runtime (the SDK isn't edge-safe).
export const runtime = "nodejs"

const API =
  process.env.SURGE_API_URL ??
  "https://tylergibbs1--surge-api-fastapi-app.modal.run"
const RUNPOD_KEY = process.env.RUNPOD_API_KEY

// Max age of a baked payload we'll serve. Bake runs once a day (~06:15 UTC);
// 25 h gives headroom for delayed or retried cron runs without serving
// genuinely stale forecasts on multi-day outages.
const BAKED_MAX_AGE_MS = 25 * 60 * 60 * 1000

// Host allow-list: prevents bearer-token exfil or SSRF if SURGE_API_URL
// is misconfigured. Vercel Blob's public CDN is whitelisted separately.
const ALLOWED_UPSTREAM_HOSTS = new Set([
  "tylergibbs1--surge-api-fastapi-app.modal.run",
  "api.runpod.ai",
  "127.0.0.1",
  "localhost",
])
;(() => {
  try {
    const u = new URL(API)
    if (u.protocol !== "https:" && u.protocol !== "http:") {
      throw new Error(`rejected protocol ${u.protocol}`)
    }
    if (!ALLOWED_UPSTREAM_HOSTS.has(u.hostname)) {
      throw new Error(`upstream host ${u.hostname} not in allow-list`)
    }
  } catch (e) {
    throw new Error(`invalid SURGE_API_URL=${API}: ${String(e)}`)
  }
})()

const CACHE_HEADERS = {
  "cache-control":
    "public, max-age=60, s-maxage=300, stale-while-revalidate=600",
} as const

const BA_SET = new Set<string>(BAS as readonly string[])

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

function isRunPod(): boolean {
  return API.startsWith("https://api.runpod.ai")
}

// ─── Baked path (fast) ─────────────────────────────────────────────────

type BakedResult =
  | { ok: true; payload: ForecastResponse }
  | { ok: false; reason: string }

async function fetchBaked(ba: BaCode, horizon: number): Promise<BakedResult> {
  if (!BLOB_TOKEN) return { ok: false, reason: "no-token" }

  // Private-store lookup: head() hits the Vercel Blob API and returns a
  // signed URL for the current version of the blob. Then we fetch that URL
  // server-side — browsers never see the token. One extra RTT vs a
  // public-CDN read, but still well under live-inference latency.
  let blobUrl: string
  try {
    const meta = await head(`forecasts/${ba}.json`, { token: BLOB_TOKEN })
    blobUrl = meta.url
  } catch (e) {
    return { ok: false, reason: `head-err:${String(e).slice(0, 40)}` }
  }

  const r = await fetch(blobUrl, { next: { revalidate: 60 } })
  if (!r.ok) return { ok: false, reason: `fetch-${r.status}` }

  const payload = (await r.json()) as ForecastResponse

  // Stale check: don't silently serve forecasts from last week if the bake
  // cron has been broken.
  const ageMs = Date.now() - Date.parse(payload.as_of_utc)
  if (!Number.isFinite(ageMs)) return { ok: false, reason: "bad-as_of" }
  if (ageMs > BAKED_MAX_AGE_MS) {
    return { ok: false, reason: `stale-${Math.round(ageMs / 3600_000)}h` }
  }

  // Baked payload is at horizon=168. Slice down to what the client asked
  // for; never up (if someone asks for 169 we have to fall back to live).
  if (payload.points.length < horizon) {
    return { ok: false, reason: `short-${payload.points.length}` }
  }
  return {
    ok: true,
    payload: { ...payload, horizon, points: payload.points.slice(0, horizon) },
  }
}

// ─── Live inference path (fallback) ────────────────────────────────────

async function fetchLiveModal(ba: BaCode, horizon: number): Promise<Response> {
  return fetch(
    `${API}/forecast/${encodeURIComponent(ba)}?horizon=${horizon}`,
    { cache: "no-store" },
  )
}

async function fetchLiveRunPod(ba: BaCode, horizon: number): Promise<Response> {
  if (!RUNPOD_KEY) {
    return Response.json(
      { error: "RUNPOD_API_KEY not set on server" },
      { status: 500 },
    )
  }
  const r = await fetch(`${API}/runsync`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${RUNPOD_KEY}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({ input: { ba, horizon } }),
    cache: "no-store",
  })
  const payload = await r.json()
  if (payload.status === "COMPLETED" && payload.output) {
    return Response.json(payload.output, { status: 200 })
  }
  if (payload.output?.error) {
    return Response.json({ detail: payload.output.error }, { status: 400 })
  }
  return Response.json(payload, { status: 502 })
}

function errorResponse(body: object, status: number): Response {
  return Response.json(body, { status })
}

// ─── Handler ───────────────────────────────────────────────────────────

export async function GET(req: NextRequest): Promise<Response> {
  const baRaw = req.nextUrl.searchParams.get("ba")
  const ba = baRaw ? baRaw.toUpperCase() : null
  const horizonRaw = req.nextUrl.searchParams.get("horizon") ?? "24"
  const horizon = Number.parseInt(horizonRaw, 10)
  const force = req.nextUrl.searchParams.get("force") === "1"

  if (!ba || !BA_SET.has(ba)) {
    return errorResponse(
      { error: `invalid ba; must be one of ${[...BA_SET].join(",")}` },
      400,
    )
  }
  if (!Number.isInteger(horizon) || horizon < 1 || horizon > 168) {
    return errorResponse(
      { error: "horizon must be an integer in 1..168" },
      422,
    )
  }

  // Fast path: baked JSON from Vercel Blob. Skipped when force=1 or the
  // read-write token isn't configured.
  let blobReason = "skipped"
  if (!force) {
    try {
      const baked = await fetchBaked(ba as BaCode, horizon)
      if (baked.ok) {
        return new Response(JSON.stringify(baked.payload), {
          status: 200,
          headers: {
            ...CACHE_HEADERS,
            "content-type": "application/json",
            "x-surge-backend": "blob",
          },
        })
      }
      blobReason = baked.reason
    } catch (e) {
      // Don't let a blob hiccup take the whole endpoint down; fall through
      // to live inference.
      blobReason = `exc:${String(e).slice(0, 40)}`
    }
  }

  // Slow path: call Modal or RunPod.
  const t0 = Date.now()
  let upstream: Response
  try {
    upstream = isRunPod()
      ? await fetchLiveRunPod(ba as BaCode, horizon)
      : await fetchLiveModal(ba as BaCode, horizon)
  } catch {
    return errorResponse({ error: "upstream unreachable" }, 502)
  }
  const elapsed = Date.now() - t0

  if (!upstream.ok) {
    return errorResponse({ error: "upstream error" }, upstream.status)
  }

  const ct = upstream.headers.get("content-type") ?? ""
  if (!ct.includes("application/json")) {
    return errorResponse({ error: "upstream returned non-json" }, 502)
  }

  const text = await upstream.text()
  try {
    JSON.parse(text)
  } catch {
    return errorResponse({ error: "upstream returned invalid json" }, 502)
  }

  return new Response(text, {
    status: 200,
    headers: {
      ...CACHE_HEADERS,
      "content-type": "application/json",
      "x-upstream-latency-ms": String(elapsed),
      "x-surge-backend": isRunPod() ? "runpod" : "modal",
      "x-blob-fallback-reason": blobReason,
    },
  })
}
