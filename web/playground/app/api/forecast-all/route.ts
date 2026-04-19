// Bulk read of baked forecasts. Returns the payload written by /api/bake's
// forecasts/all.json — every demand BA at horizon=168 in one response.
// Feeds the /grid page.
//
// Cache strategy: the underlying blob only turns over once a day. We set
// edge-cache to 30 min (matches the bake's cacheControlMaxAge), so the
// hot path after the first miss is a pure Vercel CDN hit, no blob round
// trip.

import { head } from "@vercel/blob"

const BLOB_TOKEN = process.env.BLOB_READ_WRITE_TOKEN

// @vercel/blob pulls in Node-only deps — can't run on the edge runtime.
export const runtime = "nodejs"

const CACHE_HEADERS = {
  "cache-control":
    "public, max-age=300, s-maxage=1800, stale-while-revalidate=3600",
} as const

export async function GET(): Promise<Response> {
  if (!BLOB_TOKEN) {
    return Response.json(
      { error: "BLOB_READ_WRITE_TOKEN not configured" },
      { status: 503 },
    )
  }

  let blobUrl: string
  try {
    const meta = await head("forecasts/all.json", { token: BLOB_TOKEN })
    blobUrl = meta.url
  } catch (e) {
    // 404: bake hasn't run yet. Tell the client explicitly so it can
    // render an empty state instead of spinning.
    return Response.json(
      { error: "no baked forecasts yet", detail: String(e).slice(0, 120) },
      { status: 404 },
    )
  }

  const r = await fetch(blobUrl, {
    headers: { authorization: `Bearer ${BLOB_TOKEN}` },
    next: { revalidate: 300 },
  })
  if (!r.ok) {
    return Response.json(
      { error: "blob read failed", status: r.status },
      { status: 502 },
    )
  }

  const text = await r.text()
  return new Response(text, {
    status: 200,
    headers: {
      ...CACHE_HEADERS,
      "content-type": "application/json",
      "x-surge-backend": "blob",
    },
  })
}
