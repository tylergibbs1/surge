// Auto-selects between local FastAPI and RunPod serverless based on env:
//   - SURGE_API_URL starts with "https://api.runpod.ai" → POST /runsync with bearer
//   - anything else (default http://127.0.0.1:8000) → GET /forecast/{ba}
// Zero code change to switch — set vars in .env.local.

import { NextRequest } from "next/server"

// Default to the Modal-hosted FastAPI so users cloning the repo get a
// working demo without running anything locally. Override with
// SURGE_API_URL=http://127.0.0.1:8000 for local dev.
const API =
  process.env.SURGE_API_URL ??
  "https://tylergibbs1--surge-api-fastapi-app.modal.run"
const RUNPOD_KEY = process.env.RUNPOD_API_KEY

function isRunPod(): boolean {
  return API.startsWith("https://api.runpod.ai")
}

async function fetchLocal(ba: string, horizon: number) {
  return fetch(`${API}/forecast/${ba}?horizon=${horizon}`, { cache: "no-store" })
}

async function fetchRunPod(ba: string, horizon: number) {
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
  // Unwrap RunPod's {status, output} envelope to match local shape.
  const payload = await r.json()
  if (payload.status === "COMPLETED" && payload.output) {
    return Response.json(payload.output, { status: 200 })
  }
  if (payload.output?.error) {
    return Response.json({ detail: payload.output.error }, { status: 400 })
  }
  return Response.json(payload, { status: 502 })
}

export async function GET(req: NextRequest) {
  const ba = req.nextUrl.searchParams.get("ba")?.toUpperCase()
  const horizon = Number(req.nextUrl.searchParams.get("horizon") ?? "24")

  if (!ba) {
    return Response.json({ error: "missing ba" }, { status: 400 })
  }
  if (horizon < 1 || horizon > 168) {
    return Response.json({ error: "horizon out of range (1..168)" }, { status: 422 })
  }

  const t0 = Date.now()
  const upstream = isRunPod()
    ? await fetchRunPod(ba, horizon)
    : await fetchLocal(ba, horizon)
  const elapsed = Date.now() - t0

  if (upstream.headers.get("content-type")?.includes("application/json")) {
    const body = await upstream.text()
    return new Response(body, {
      status: upstream.status,
      headers: {
        "content-type": "application/json",
        "x-upstream-latency-ms": String(elapsed),
        "x-upstream-backend": isRunPod() ? "runpod" : "local",
      },
    })
  }
  return upstream
}
