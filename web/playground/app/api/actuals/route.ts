// Per-BA realized load proxy. Used by the forecast chart to draw the
// historical context line that runs into the dashed forecast — without
// actuals behind the forecast-start break, the break has no meaning.
//
// Proxies to FastAPI /actuals/{ba}. Read-only; no blob cache (the scan
// upstream is cheap enough to skip the bake complexity of /forecast).

import { NextRequest } from "next/server"

import { BAS } from "@/lib/us-grid-geo"

const API =
  process.env.SURGE_API_URL ??
  "https://tylergibbs1--surge-api-fastapi-app.modal.run"

const ALLOWED_UPSTREAM_HOSTS = new Set([
  "tylergibbs1--surge-api-fastapi-app.modal.run",
  "api.runpod.ai",
  "127.0.0.1",
  "localhost",
])
;(() => {
  const u = new URL(API)
  if (!ALLOWED_UPSTREAM_HOSTS.has(u.hostname)) {
    throw new Error(`upstream host ${u.hostname} not in allow-list`)
  }
})()

const BA_SET = new Set<string>(BAS as readonly string[])

// 5-minute edge cache — matches the upstream /actuals Cache-Control. The
// store is only refreshed when the hourly ingest cron lands, so a
// five-minute stale window is indistinguishable from the source.
const CACHE_HEADERS = {
  "cache-control":
    "public, max-age=60, s-maxage=300, stale-while-revalidate=600",
} as const

export async function GET(req: NextRequest): Promise<Response> {
  const baRaw = req.nextUrl.searchParams.get("ba")
  const ba = baRaw ? baRaw.toUpperCase() : null
  const hoursRaw = req.nextUrl.searchParams.get("hours") ?? "48"
  const hours = Number.parseInt(hoursRaw, 10)

  if (!ba || !BA_SET.has(ba)) {
    return Response.json(
      { error: `invalid ba; must be one of ${[...BA_SET].join(",")}` },
      { status: 400 },
    )
  }
  if (!Number.isInteger(hours) || hours < 1 || hours > 720) {
    return Response.json(
      { error: "hours must be an integer in 1..720" },
      { status: 422 },
    )
  }

  let upstream: Response
  try {
    upstream = await fetch(
      `${API}/actuals/${encodeURIComponent(ba)}?hours=${hours}`,
      { next: { revalidate: 300 } },
    )
  } catch {
    return Response.json({ error: "upstream unreachable" }, { status: 502 })
  }

  if (!upstream.ok) {
    return Response.json(
      { error: "upstream error", status: upstream.status },
      { status: upstream.status === 503 ? 503 : 502 },
    )
  }

  const text = await upstream.text()
  return new Response(text, {
    status: 200,
    headers: {
      ...CACHE_HEADERS,
      "content-type": "application/json",
    },
  })
}
