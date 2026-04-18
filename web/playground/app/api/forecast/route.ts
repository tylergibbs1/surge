// Proxies to the Surge forecast API. In dev this points at a local FastAPI
// (http://127.0.0.1:8000). In prod set SURGE_API_URL to the RunPod endpoint.

import { NextRequest } from "next/server"

const API = process.env.SURGE_API_URL ?? "http://127.0.0.1:8000"

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
  const upstream = await fetch(`${API}/forecast/${ba}?horizon=${horizon}`, {
    cache: "no-store",
  })
  const text = await upstream.text()
  const elapsed = Date.now() - t0

  return new Response(text, {
    status: upstream.status,
    headers: {
      "content-type": upstream.headers.get("content-type") ?? "application/json",
      "x-upstream-latency-ms": String(elapsed),
    },
  })
}
