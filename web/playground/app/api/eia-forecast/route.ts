// Day-ahead demand forecast for one BA — sourced from EIA's Grid
// Monitor dashboard API (type=DF, "Day-ahead demand forecast"). These
// are the forecasts the BA operators themselves submit to EIA every
// morning. Used as a reference line on surge's forecast chart so a
// reader can compare surge-fm-v3 against operator consensus.
//
// Not all 53 BAs publish DF — smaller non-RTO utilities often don't.
// The endpoint returns an empty `points` array when a BA is absent; the
// frontend then simply omits the reference line for that BA.

import { NextRequest } from "next/server"

import { BAS } from "@/lib/us-grid-geo"

const EIA_DASHBOARD_API =
  "https://www.eia.gov/electricity/930-api/region_data/series_data"

const UA = "surge-playground/1.0 (+https://github.com/tylergibbs1/surge)"

// 5-min edge cache: EIA's DF only updates when a BA submits a new day.
const CACHE_HEADERS = {
  "cache-control":
    "public, max-age=60, s-maxage=300, stale-while-revalidate=600",
} as const

const BA_SET = new Set<string>(BAS as readonly string[])

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

function parseEiaDate(s: string): string {
  const [datePart, timePart] = s.split(" ")
  const [mm, dd, yyyy] = datePart.split("/")
  return `${yyyy}-${mm}-${dd}T${timePart}Z`
}

export async function GET(req: NextRequest): Promise<Response> {
  const baRaw = req.nextUrl.searchParams.get("ba")
  const ba = baRaw ? baRaw.toUpperCase() : null
  const hoursRaw = req.nextUrl.searchParams.get("hours") ?? "168"
  const hours = Number.parseInt(hoursRaw, 10)

  if (!ba || !BA_SET.has(ba)) {
    return Response.json(
      { error: `invalid ba; must be one of ${[...BA_SET].join(",")}` },
      { status: 400 },
    )
  }
  if (!Number.isInteger(hours) || hours < 1 || hours > 720) {
    return Response.json(
      { error: "hours must be in 1..720" },
      { status: 422 },
    )
  }

  // Forward-looking window: the "day-ahead" series is a forecast, so
  // we want `now` → `now + hours`. A 12-hour leftward pad covers EIA's
  // occasional lag in republishing the current hour.
  const now = new Date()
  now.setUTCMinutes(0, 0, 0)
  const start = new Date(now.getTime() - 12 * 3_600_000)
  const end = new Date(now.getTime() + hours * 3_600_000)

  const qs = new URLSearchParams({
    "respondent[0]": ba,
    "type[0]": "DF",
    frequency: "hourly",
    start: fmtEia(start),
    end: fmtEia(end),
    timezone: "Eastern",
    limit: "10000",
    offset: "0",
  })

  let upstream: Response
  try {
    upstream = await fetch(`${EIA_DASHBOARD_API}?${qs}`, {
      headers: {
        "User-Agent": UA,
        Referer:
          "https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48",
        Accept: "application/json",
      },
      next: { revalidate: 300 },
    })
  } catch {
    return Response.json({ error: "upstream unreachable" }, { status: 502 })
  }

  if (!upstream.ok) {
    return Response.json(
      { error: "upstream error", status: upstream.status },
      { status: 502 },
    )
  }

  const body = (await upstream.json()) as EiaSeries[]
  const series = body?.[0]?.data?.[0]?.VALUES
  const dates = series?.DATES ?? []
  const values = series?.DATA ?? []

  const points: Array<{ ts_utc: string; df_mw: number }> = []
  for (let i = 0; i < Math.min(dates.length, values.length); i++) {
    const v = values[i]
    if (v == null) continue
    points.push({ ts_utc: parseEiaDate(dates[i]), df_mw: v })
  }

  const payload = {
    ba,
    source: "eia-930",
    type: "DF",
    hours: points.length,
    points,
  }
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: {
      ...CACHE_HEADERS,
      "content-type": "application/json",
    },
  })
}
