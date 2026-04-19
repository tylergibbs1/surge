// Proxy to the FastAPI /current-load endpoint on Modal. Keeps the browser
// talking only to the Vercel origin (preserves the CORS/allow-list model
// in the forecast route) and lets us add edge caching so the Liveline
// hero widget polling at 60 s doesn't turn into 60 Modal wake-ups an hour.

import { NextRequest } from "next/server"

const API =
  process.env.SURGE_API_URL ??
  "https://tylergibbs1--surge-api-fastapi-app.modal.run"

// Cache on the Vercel edge for 50 s — shorter than the 60 s poll so the
// widget feels responsive to new hourly data, longer than the time
// between overlapping concurrent requests so we're not hammering Modal.
const CACHE_HEADERS = {
  "cache-control":
    "public, max-age=30, s-maxage=50, stale-while-revalidate=300",
} as const

export async function GET(req: NextRequest): Promise<Response> {
  const hoursRaw = req.nextUrl.searchParams.get("hours") ?? "24"
  const hours = Number.parseInt(hoursRaw, 10)
  if (!Number.isInteger(hours) || hours < 1 || hours > 168) {
    return Response.json(
      { error: "hours must be in 1..168" },
      { status: 422 },
    )
  }

  try {
    const r = await fetch(`${API}/current-load?hours=${hours}`, {
      // Let Next.js's data cache do its thing — revalidate at the same
      // interval the edge cache above holds.
      next: { revalidate: 50 },
    })
    if (!r.ok) {
      return Response.json(
        { error: "upstream error", status: r.status },
        { status: r.status === 503 ? 503 : 502 },
      )
    }
    const text = await r.text()
    return new Response(text, {
      status: 200,
      headers: {
        ...CACHE_HEADERS,
        "content-type": "application/json",
      },
    })
  } catch {
    return Response.json({ error: "upstream unreachable" }, { status: 502 })
  }
}
