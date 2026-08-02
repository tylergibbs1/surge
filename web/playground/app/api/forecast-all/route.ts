// Bulk read of the latest complete seven-RTO immutable forecast run.
// The publisher advances forecasts/v2/current.json only after every region
// validates and the immutable run manifest is durable.

import { head } from "@vercel/blob"
import { NextRequest } from "next/server"

import { RTO_CODES } from "@/lib/v2-contracts"

const BLOB_TOKEN = process.env.BLOB_READ_WRITE_TOKEN
const MAX_ARTIFACT_AGE_MS = 36 * 60 * 60 * 1000

export const runtime = "nodejs"

const CACHE_HEADERS = {
  "cache-control":
    "public, max-age=300, s-maxage=1800, stale-while-revalidate=3600",
} as const

type CurrentPointer = {
  schema_version: "2.0"
  run_id: string
  published_at_utc: string
  all_url: string
}

type AllPayload = {
  schema_version: "2.0"
  run_id: string
  baked_at: string
  horizon: number
  forecasts: Array<{ ba: string; points: unknown[] }>
}

async function fetchPrivateJson(
  url: string,
  options: { freshKey?: string } = {},
): Promise<unknown> {
  const parsed = new URL(url)
  if (
    parsed.protocol !== "https:" ||
    !parsed.hostname.endsWith(".blob.vercel-storage.com")
  ) {
    throw new Error("unexpected blob host")
  }
  if (options.freshKey) parsed.searchParams.set("surge_release", options.freshKey)
  const response = await fetch(
    parsed,
    options.freshKey
      ? {
          headers: { authorization: `Bearer ${BLOB_TOKEN}` },
          cache: "no-store",
        }
      : {
          headers: { authorization: `Bearer ${BLOB_TOKEN}` },
          next: { revalidate: 300 },
        },
  )
  if (!response.ok) throw new Error(`blob read ${response.status}`)
  return response.json()
}

function validatePointer(value: unknown): CurrentPointer {
  if (!value || typeof value !== "object") throw new Error("invalid current pointer")
  const pointer = value as Partial<CurrentPointer>
  if (
    pointer.schema_version !== "2.0" ||
    !pointer.run_id ||
    !pointer.all_url ||
    !pointer.published_at_utc
  ) {
    throw new Error("invalid current pointer")
  }
  const age = Date.now() - Date.parse(pointer.published_at_utc)
  if (!Number.isFinite(age) || age > MAX_ARTIFACT_AGE_MS) {
    throw new Error("current pointer is stale")
  }
  return pointer as CurrentPointer
}

function validatePayload(value: unknown, pointer: CurrentPointer): AllPayload {
  if (!value || typeof value !== "object") throw new Error("invalid forecast payload")
  const payload = value as Partial<AllPayload>
  if (
    payload.schema_version !== "2.0" ||
    payload.run_id !== pointer.run_id ||
    payload.horizon !== 168 ||
    !Array.isArray(payload.forecasts) ||
    payload.forecasts.length !== RTO_CODES.length
  ) {
    throw new Error("invalid forecast payload")
  }
  const codes = new Set(payload.forecasts.map((forecast) => forecast.ba))
  if (
    RTO_CODES.some((code) => !codes.has(code)) ||
    payload.forecasts.some((forecast) => forecast.points.length !== 168)
  ) {
    throw new Error("forecast payload is incomplete")
  }
  return payload as AllPayload
}

export async function GET(request: NextRequest): Promise<Response> {
  if (!BLOB_TOKEN) {
    return Response.json(
      { error: "BLOB_READ_WRITE_TOKEN not configured" },
      { status: 503 },
    )
  }

  try {
    const expectedRunId = request.nextUrl.searchParams.get("run_id")
    const metadata = await head("forecasts/v2/current.json", {
      token: BLOB_TOKEN,
    })
    const pointer = validatePointer(
      await fetchPrivateJson(metadata.url, {
        // Release verification supplies the expected immutable run ID. This
        // bypasses both Next's data cache and any stable-URL Blob cache.
        freshKey: expectedRunId ?? undefined,
      }),
    )
    if (expectedRunId && pointer.run_id !== expectedRunId) {
      throw new Error("current pointer does not match the requested release")
    }
    const payload = validatePayload(await fetchPrivateJson(pointer.all_url), pointer)
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: {
        ...CACHE_HEADERS,
        "content-type": "application/json",
        "x-surge-backend": "immutable-ledger",
        "x-surge-run-id": pointer.run_id,
      },
    })
  } catch (error) {
    return Response.json(
      { error: "no complete current forecast run", detail: String(error) },
      { status: 503 },
    )
  }
}
