// Auto-selects between local FastAPI and RunPod serverless based on env:
//   - SURGE_API_URL starts with "https://api.runpod.ai" → POST /runsync with bearer
//   - anything else (default Modal) → GET /forecast/{ba}
// Zero code change to switch — set vars in .env.local.

import { NextRequest } from "next/server"

// Default to the Modal-hosted FastAPI so users cloning the repo get a
// working demo without running anything locally. Override with
// SURGE_API_URL=http://127.0.0.1:8000 for local dev.
const API =
  process.env.SURGE_API_URL ??
  "https://tylergibbs1--surge-api-fastapi-app.modal.run"
const RUNPOD_KEY = process.env.RUNPOD_API_KEY

// Next.js 16 sets `Cache-Control: private, no-cache, no-store, max-age=0,
// must-revalidate` on dynamic route handlers by default, which disables
// both browser and Vercel-edge caching. Project-level headers in
// vercel.json DO NOT override this — we have to attach the header to the
// Response we return. See:
// https://github.com/vercel/next.js/blob/v16.2.2/docs/01-app/02-guides/cdn-caching.mdx
//
// - max-age=60: browser keeps the JSON for 60 s
// - s-maxage=300: Vercel's edge POPs hold it for 5 min
// - stale-while-revalidate=600: on a miss past 300 s, serve stale
//   immediately and refresh in the background. Keeps tail latency low.
const CACHE_HEADERS = {
  "cache-control":
    "public, max-age=60, s-maxage=300, stale-while-revalidate=600",
} as const

function isRunPod(): boolean {
  return API.startsWith("https://api.runpod.ai")
}

async function fetchLocal(ba: string, horizon: number): Promise<Response> {
  return fetch(`${API}/forecast/${ba}?horizon=${horizon}`, { cache: "no-store" })
}

async function fetchRunPod(ba: string, horizon: number): Promise<Response> {
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
  // Errors don't carry the cache headers — we don't want a 422 to be
  // served from edge for 5 minutes.
  return Response.json(body, { status })
}

export async function GET(req: NextRequest): Promise<Response> {
  const ba = req.nextUrl.searchParams.get("ba")?.toUpperCase()
  const horizon = Number(req.nextUrl.searchParams.get("horizon") ?? "24")

  if (!ba) return errorResponse({ error: "missing ba" }, 400)
  if (horizon < 1 || horizon > 168) {
    return errorResponse({ error: "horizon out of range (1..168)" }, 422)
  }

  const t0 = Date.now()
  const upstream = isRunPod()
    ? await fetchRunPod(ba, horizon)
    : await fetchLocal(ba, horizon)
  const elapsed = Date.now() - t0

  if (!upstream.ok) {
    // Pass upstream error through without our cache headers.
    const text = await upstream.text()
    return new Response(text, {
      status: upstream.status,
      headers: { "content-type": "application/json" },
    })
  }

  if (upstream.headers.get("content-type")?.includes("application/json")) {
    const body = await upstream.text()
    return new Response(body, {
      status: 200,
      headers: {
        ...CACHE_HEADERS,
        "content-type": "application/json",
        "x-upstream-latency-ms": String(elapsed),
        "x-upstream-backend": isRunPod() ? "runpod" : "modal",
      },
    })
  }
  return upstream
}
