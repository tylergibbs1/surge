// Live US-48 demand powering the "US demand · live" hero. Uses the
// public endpoint that the EIA Grid Monitor dashboard itself calls —
// one request for a pre-aggregated US48 total, no API key, no per-BA
// fan-out. Falls back to our own Modal `/current-load` aggregate on any
// dashboard API failure so the hero stays up even if EIA breaks the
// undocumented endpoint.

import { NextRequest } from "next/server"

// Matches EIA's timezone handling: the query-string `timezone=` param is
// advisory; the response always comes back with UTC timestamps.
const EIA_DASHBOARD_API =
  "https://www.eia.gov/electricity/930-api/region_data/series_data"

// Identify ourselves so EIA can contact us if they want a formal API
// partnership or have to rate-limit us. Public dashboard endpoint is
// undocumented; being a good citizen here buys us headroom.
const UA = "surge-playground/1.0 (+https://github.com/tylergibbs1/surge)"

const MODAL_FALLBACK =
  process.env.SURGE_API_URL ??
  "https://tylergibbs1--surge-api-fastapi-app.modal.run"

// Shorter maxage than before — we're now reading near-real-time from
// EIA directly, so stale-by-50s is the floor, not the ceiling.
const CACHE_HEADERS = {
  "cache-control":
    "public, max-age=30, s-maxage=50, stale-while-revalidate=300",
} as const

type Point = { ts_utc: string; total_mw: number; ba_count: number }

type EiaSeries = {
  data?: Array<{
    VALUES?: {
      DATES?: string[]
      DATA?: Array<number | null>
    }
  }>
}

function pad2(n: number): string {
  return n < 10 ? `0${n}` : String(n)
}

// EIA's API takes dates as "MMDDYYYY HH:MM:SS" in UTC. `Date.toISOString()`
// gives us UTC components trivially.
function fmtEia(d: Date): string {
  return (
    pad2(d.getUTCMonth() + 1) +
    pad2(d.getUTCDate()) +
    d.getUTCFullYear() +
    " " +
    pad2(d.getUTCHours()) +
    ":00:00"
  )
}

// The API returns dates as "MM/DD/YYYY HH:MM:SS" in UTC.
function parseEiaDate(s: string): string {
  const [datePart, timePart] = s.split(" ")
  const [mm, dd, yyyy] = datePart.split("/")
  return `${yyyy}-${mm}-${dd}T${timePart}Z`
}

async function fetchEiaUs48(hours: number): Promise<Point[]> {
  // Ask for a slightly wider window than requested so EIA's own
  // publication lag (they trail real time by a few hours for some
  // hours) still yields `hours` good points after filtering.
  const now = new Date()
  const end = new Date(now)
  end.setUTCMinutes(0, 0, 0)
  end.setUTCHours(end.getUTCHours() + 1) // inclusive of the current hour
  const start = new Date(end.getTime() - (hours + 12) * 3_600_000)

  const qs = new URLSearchParams({
    "respondent[0]": "US48",
    "type[0]": "D",
    frequency: "hourly",
    start: fmtEia(start),
    end: fmtEia(end),
    timezone: "Eastern",
    limit: "10000",
    offset: "0",
  })

  const r = await fetch(`${EIA_DASHBOARD_API}?${qs}`, {
    headers: {
      "User-Agent": UA,
      Referer:
        "https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48",
      Accept: "application/json",
    },
    // Vercel data cache: keep one response per clock-minute so a burst
    // of hero polls collapses to at most one upstream hit.
    next: { revalidate: 50 },
  })
  if (!r.ok) throw new Error(`eia dashboard ${r.status}`)
  const body = (await r.json()) as EiaSeries[]

  const series = body?.[0]?.data?.[0]?.VALUES
  const dates = series?.DATES ?? []
  const values = series?.DATA ?? []
  if (dates.length === 0 || dates.length !== values.length) {
    throw new Error("eia dashboard: empty or malformed payload")
  }

  const out: Point[] = []
  for (let i = 0; i < dates.length; i++) {
    const v = values[i]
    if (v == null) continue
    out.push({
      ts_utc: parseEiaDate(dates[i]),
      total_mw: v,
      // ba_count is an artifact of our aggregate path. EIA publishes a
      // single US48 number; 0 here signals "pre-aggregated, no coverage
      // gate was applied."
      ba_count: 0,
    })
  }
  return out.slice(-hours)
}

async function fetchModalFallback(hours: number): Promise<Point[]> {
  const r = await fetch(`${MODAL_FALLBACK}/current-load?hours=${hours}`, {
    next: { revalidate: 50 },
  })
  if (!r.ok) throw new Error(`modal fallback ${r.status}`)
  const body = await r.json()
  return body.points as Point[]
}

export async function GET(req: NextRequest): Promise<Response> {
  const hoursRaw = req.nextUrl.searchParams.get("hours") ?? "24"
  const hours = Number.parseInt(hoursRaw, 10)
  if (!Number.isInteger(hours) || hours < 1 || hours > 168) {
    return Response.json(
      { error: "hours must be in 1..168" },
      { status: 422 },
    )
  }

  let points: Point[]
  let source = "eia-dashboard"
  try {
    points = await fetchEiaUs48(hours)
    if (points.length === 0) throw new Error("eia dashboard returned no points")
  } catch (e) {
    console.warn("[live-load] eia dashboard failed:", e)
    try {
      points = await fetchModalFallback(hours)
      source = "modal-fallback"
    } catch (e2) {
      console.error("[live-load] modal fallback failed:", e2)
      return Response.json({ error: "upstreams unreachable" }, { status: 502 })
    }
  }

  if (points.length === 0) {
    return Response.json({ error: "no recent load data" }, { status: 503 })
  }

  const latest = points[points.length - 1]
  const payload = {
    as_of_utc: new Date().toISOString(),
    latest_ts_utc: latest.ts_utc,
    latest_total_mw: latest.total_mw,
    hours: points.length,
    points,
  }
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: {
      ...CACHE_HEADERS,
      "content-type": "application/json",
      "x-surge-source": source,
    },
  })
}
