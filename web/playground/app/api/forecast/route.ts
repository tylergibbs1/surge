// Auto-selects between local FastAPI and RunPod serverless based on env:
//   - SURGE_API_URL starts with "https://api.runpod.ai" → POST /runsync with bearer
//   - anything else (default Modal) → GET /forecast/{ba}
// Zero code change to switch — set vars in .env.local.

import { NextRequest } from "next/server"

import { BAS, type BaCode } from "@/lib/us-grid-geo"

// Default to the Modal-hosted FastAPI so users cloning the repo get a
// working demo without running anything locally. Override with
// SURGE_API_URL=http://127.0.0.1:8000 for local dev.
const API =
  process.env.SURGE_API_URL ??
  "https://tylergibbs1--surge-api-fastapi-app.modal.run"
const RUNPOD_KEY = process.env.RUNPOD_API_KEY

// Host allow-list: prevents bearer-token exfil or SSRF to cloud metadata
// endpoints if SURGE_API_URL is ever misconfigured. localhost is allowed
// for dev; prod must be https to one of the known providers.
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
    // Fail the module import — better than silently routing creds to an
    // arbitrary attacker-controlled URL at runtime.
    throw new Error(`invalid SURGE_API_URL=${API}: ${String(e)}`)
  }
})()

// Next.js 16 sets `Cache-Control: private, no-cache, no-store` on dynamic
// route handlers by default. Override on successful responses so Vercel's
// edge + browsers cache for 5 min / 60 s respectively.
const CACHE_HEADERS = {
  "cache-control":
    "public, max-age=60, s-maxage=300, stale-while-revalidate=600",
} as const

const BA_SET = new Set<string>(BAS as readonly string[])

function isRunPod(): boolean {
  return API.startsWith("https://api.runpod.ai")
}

async function fetchLocal(ba: BaCode, horizon: number): Promise<Response> {
  // `ba` is already validated against the allow-list, so no escaping
  // concerns. `horizon` is a bounded integer. Using encodeURIComponent
  // anyway as a defence-in-depth belt-and-suspenders.
  return fetch(
    `${API}/forecast/${encodeURIComponent(ba)}?horizon=${horizon}`,
    { cache: "no-store" },
  )
}

async function fetchRunPod(ba: BaCode, horizon: number): Promise<Response> {
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
  // Errors are deliberately not cached — we don't want 422 / 502 pinned
  // at the edge for 5 minutes.
  return Response.json(body, { status })
}

export async function GET(req: NextRequest): Promise<Response> {
  const baRaw = req.nextUrl.searchParams.get("ba")
  const ba = baRaw ? baRaw.toUpperCase() : null
  const horizonRaw = req.nextUrl.searchParams.get("horizon") ?? "24"
  const horizon = Number.parseInt(horizonRaw, 10)

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

  const t0 = Date.now()
  let upstream: Response
  try {
    upstream = isRunPod()
      ? await fetchRunPod(ba as BaCode, horizon)
      : await fetchLocal(ba as BaCode, horizon)
  } catch {
    return errorResponse({ error: "upstream unreachable" }, 502)
  }
  const elapsed = Date.now() - t0

  if (!upstream.ok) {
    // Swallow upstream error bodies — they may contain stack traces. The
    // client only learns the HTTP status.
    return errorResponse({ error: "upstream error" }, upstream.status)
  }

  const ct = upstream.headers.get("content-type") ?? ""
  if (!ct.includes("application/json")) {
    return errorResponse({ error: "upstream returned non-json" }, 502)
  }

  // Validate the body is JSON before forwarding — if upstream got 200'd
  // but returned garbage (misconfigured proxy, etc.), don't pass it on.
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
      "x-upstream-backend": isRunPod() ? "runpod" : "modal",
    },
  })
}
