// Authenticated daily publisher for the seven-RTO v0.2 forecast ledger.
//
// One serialized Modal publisher commits all seven forecasts and the run
// marker durably before this route mirrors that immutable run into Blob.

import { head, put } from "@vercel/blob"
import { NextRequest } from "next/server"

import { RTO_CODES, type RtoCode } from "@/lib/v2-contracts"

const PUBLISHER_URL = process.env.SURGE_PUBLISHER_URL

const BAKE_SECRET = process.env.BAKE_SECRET
const BLOB_TOKEN = process.env.BLOB_READ_WRITE_TOKEN
const LEDGER_KEY = process.env.SURGE_LEDGER_KEY
const ALLOWED_API_HOSTS = new Set(
  (
    process.env.SURGE_ALLOWED_API_HOSTS ??
    "tylergibbs1--surge-api-v02-ledger-publisher-app.modal.run,127.0.0.1,localhost"
  )
    .split(",")
    .map((host) => host.trim().toLowerCase())
    .filter(Boolean),
)

const BAKE_HORIZON = 168
const MAX_DATA_LAG_MS = 12 * 60 * 60 * 1000
const MAX_ISSUANCE_AGE_MS = 90 * 60 * 1000
const MAX_CLOCK_SKEW_MS = 5 * 60 * 1000
const HOUR_MS = 60 * 60 * 1000

export const runtime = "nodejs"
export const maxDuration = 300

type ForecastPoint = {
  ts_utc: string
  mean_mw: number
  median_mw: number
  p10_mw: number
  p90_mw: number
  temp_c?: number | null
}

type ForecastResponse = {
  schema_version: "2.0"
  issuance_id: string
  run_id: string
  ba: string
  model: string
  model_revision: string
  model_artifact_sha256?: string | null
  as_of_utc: string
  generated_at_utc: string
  issued_at_utc: string
  data_cutoff_utc: string
  feature_cutoff_utc: string
  context_start_utc: string
  context_end_utc: string
  horizon: number
  units: string
  feature_spec_version: string
  feature_spec_sha256: string
  feature_snapshot_sha256: string
  availability_mode: string
  point_estimate_kind: "median" | "mean"
  mase_scale_24: number
  code_revision: string
  committed: boolean
  run_published: boolean
  published_at_utc: string | null
  warnings: string[]
  points: ForecastPoint[]
}

type LedgerBakeResponse = {
  schema_version: "2.0"
  run: {
    run_id: string
    scheduled_for_utc: string
    published_at_utc: string
    required_bas: string[]
    issuance_ids: Record<string, string>
  }
  forecasts: ForecastResponse[]
  committed_regions: number
  run_published: true
}

type BakeResult = {
  ba: string
  ok: boolean
  issuance_id?: string
  url?: string
  error?: string
}

type RunProvenance = {
  model: string
  model_revision: string
  model_artifact_sha256: string | null
  code_revision: string
  feature_spec_version: string
  feature_spec_sha256: string
  availability_mode: string
  point_estimate_kind: "median" | "mean"
  units: string
  horizon: number
}

function parseUtc(value: string, label: string): number {
  const timestamp = Date.parse(value)
  if (!Number.isFinite(timestamp) || !value.endsWith("Z")) {
    throw new Error(`${label}: expected an ISO-8601 UTC timestamp`)
  }
  return timestamp
}

function validateForecast(
  value: ForecastResponse,
  expectedBa: RtoCode,
  scheduledForUtc: string,
  checkedAtMs: number,
): ForecastResponse {
  if (value.schema_version !== "2.0") throw new Error(`${expectedBa}: wrong schema`)
  if (value.ba !== expectedBa) throw new Error(`${expectedBa}: BA mismatch`)
  if (!value.issuance_id || !value.run_id) throw new Error(`${expectedBa}: missing IDs`)
  if (!value.committed || !value.run_published || !value.published_at_utc) {
    throw new Error(`${expectedBa}: issuance is not durably published`)
  }
  if (
    !value.model ||
    !value.model_revision ||
    !value.code_revision ||
    value.code_revision === "unknown"
  ) {
    throw new Error(`${expectedBa}: missing model/code provenance`)
  }
  if (!value.feature_spec_version || !value.feature_spec_sha256) {
    throw new Error(`${expectedBa}: missing feature specification provenance`)
  }
  if (!value.feature_snapshot_sha256) {
    throw new Error(`${expectedBa}: missing feature snapshot digest`)
  }
  if (
    value.feature_spec_version !== "load-v2-core" ||
    value.availability_mode !== "exact_vintage" ||
    value.point_estimate_kind !== "median" ||
    value.units !== "MW"
  ) {
    throw new Error(`${expectedBa}: forecast does not satisfy the v0.2 live contract`)
  }
  if (value.horizon !== BAKE_HORIZON || value.points.length !== BAKE_HORIZON) {
    throw new Error(`${expectedBa}: expected ${BAKE_HORIZON} points`)
  }

  const issuedAt = parseUtc(value.issued_at_utc, `${expectedBa} issued_at_utc`)
  const scheduledFor = parseUtc(scheduledForUtc, "scheduled_for_utc")
  if (issuedAt > checkedAtMs + MAX_CLOCK_SKEW_MS) {
    throw new Error(`${expectedBa}: issuance is unexpectedly in the future`)
  }
  if (checkedAtMs - issuedAt > MAX_ISSUANCE_AGE_MS) {
    throw new Error(`${expectedBa}: issuance is too old for publication`)
  }
  if (
    scheduledFor > issuedAt + MAX_CLOCK_SKEW_MS ||
    issuedAt - scheduledFor > MAX_ISSUANCE_AGE_MS
  ) {
    throw new Error(`${expectedBa}: issuance is not aligned to the requested slot`)
  }
  const dataCutoff = parseUtc(value.data_cutoff_utc, `${expectedBa} data_cutoff_utc`)
  const featureCutoff = parseUtc(
    value.feature_cutoff_utc,
    `${expectedBa} feature_cutoff_utc`,
  )
  if (dataCutoff > issuedAt || featureCutoff > issuedAt) {
    throw new Error(`${expectedBa}: cutoff is after issuance`)
  }
  if (issuedAt - dataCutoff > MAX_DATA_LAG_MS) {
    throw new Error(`${expectedBa}: source data is more than 12 hours old`)
  }

  let previous = 0
  value.points.forEach((point, index) => {
    const validAt = parseUtc(point.ts_utc, `${expectedBa} point ${index}`)
    if (validAt <= issuedAt) throw new Error(`${expectedBa}: point is not post-issuance`)
    if (index > 0 && validAt - previous !== HOUR_MS) {
      throw new Error(`${expectedBa}: points are not contiguous hourly values`)
    }
    previous = validAt
    const numbers = [point.mean_mw, point.p10_mw, point.median_mw, point.p90_mw]
    if (numbers.some((number) => !Number.isFinite(number) || number < 0)) {
      throw new Error(`${expectedBa}: non-finite or negative forecast value`)
    }
    if (!(point.p10_mw <= point.median_mw && point.median_mw <= point.p90_mw)) {
      throw new Error(`${expectedBa}: crossing quantiles`)
    }
  })
  return value
}

function runProvenance(value: ForecastResponse): RunProvenance {
  return {
    model: value.model,
    model_revision: value.model_revision,
    model_artifact_sha256: value.model_artifact_sha256 ?? null,
    code_revision: value.code_revision,
    feature_spec_version: value.feature_spec_version,
    feature_spec_sha256: value.feature_spec_sha256,
    availability_mode: value.availability_mode,
    point_estimate_kind: value.point_estimate_kind,
    units: value.units,
    horizon: value.horizon,
  }
}

function assertUniformRunProvenance(
  forecasts: readonly ForecastResponse[],
): RunProvenance {
  if (forecasts.length !== RTO_CODES.length) {
    throw new Error(`expected ${RTO_CODES.length} forecasts for publication`)
  }
  const provenance = runProvenance(forecasts[0])
  const expected = JSON.stringify(provenance)
  for (const forecast of forecasts.slice(1)) {
    if (JSON.stringify(runProvenance(forecast)) !== expected) {
      throw new Error(`${forecast.ba}: mixed run provenance`)
    }
  }
  return provenance
}

function scheduledSlot(now = new Date()): string {
  const slot = new Date(now)
  // The scheduled job still lands on 06:15 UTC. An early/manual invocation
  // uses the most recent hourly :15 slot instead of colliding with yesterday's
  // completed release and accidentally accepting a day-old retry.
  slot.setUTCMinutes(15, 0, 0)
  if (slot.getTime() > now.getTime()) slot.setUTCHours(slot.getUTCHours() - 1)
  return slot.toISOString()
}

function assertUpstream(): void {
  if (!PUBLISHER_URL) throw new Error("SURGE_PUBLISHER_URL is required")
  const url = new URL(PUBLISHER_URL)
  if (
    !ALLOWED_API_HOSTS.has(url.hostname.toLowerCase()) ||
    !["http:", "https:"].includes(url.protocol)
  ) {
    throw new Error(`SURGE_PUBLISHER_URL host is not allowed: ${url.hostname}`)
  }
  if (
    url.pathname !== "/" ||
    url.search ||
    url.hash ||
    url.username ||
    url.password
  ) {
    throw new Error(
      "SURGE_PUBLISHER_URL must be an origin with no path, query, or credentials",
    )
  }
}

async function fetchLedgerBatch(
  scheduledForUtc: string,
  checkedAtMs: number,
): Promise<ForecastResponse[]> {
  if (!PUBLISHER_URL) throw new Error("SURGE_PUBLISHER_URL is required")
  const query = new URLSearchParams({
    horizon: String(BAKE_HORIZON),
    scheduled_for_utc: scheduledForUtc,
  })
  const response = await fetch(
    `${PUBLISHER_URL}/ledger/runs/bake?${query}`,
    {
      method: "POST",
      headers: { "x-surge-ledger-key": LEDGER_KEY ?? "" },
      cache: "no-store",
    },
  )
  if (!response.ok) throw new Error(`publisher upstream ${response.status}`)
  if (response.headers.get("x-surge-volume-committed") !== "true") {
    throw new Error("publisher did not attest a durable Modal Volume commit")
  }
  const value = (await response.json()) as LedgerBakeResponse
  if (
    value.schema_version !== "2.0" ||
    value.run_published !== true ||
    value.committed_regions !== RTO_CODES.length ||
    !Array.isArray(value.forecasts) ||
    value.forecasts.length !== RTO_CODES.length ||
    value.run.scheduled_for_utc !== scheduledForUtc
  ) {
    throw new Error("publisher returned an incomplete run")
  }
  const runPublishedAt = parseUtc(
    value.run.published_at_utc,
    "run published_at_utc",
  )
  if (runPublishedAt > Date.now() + MAX_CLOCK_SKEW_MS) {
    throw new Error("publisher run timestamp is unexpectedly in the future")
  }
  const byBa = new Map(value.forecasts.map((forecast) => [forecast.ba, forecast]))
  const ordered = RTO_CODES.map((ba) => {
    const forecast = byBa.get(ba)
    if (!forecast) throw new Error(`${ba}: publisher omitted forecast`)
    const validated = validateForecast(forecast, ba, scheduledForUtc, checkedAtMs)
    if (
      validated.run_id !== value.run.run_id ||
      value.run.issuance_ids[ba] !== validated.issuance_id ||
      validated.published_at_utc !== value.run.published_at_utc
    ) {
      throw new Error(`${ba}: publisher run marker disagrees with issuance`)
    }
    return validated
  })
  if (
    value.run.required_bas.length !== RTO_CODES.length ||
    RTO_CODES.some((ba) => !value.run.required_bas.includes(ba))
  ) {
    throw new Error("publisher run marker has the wrong required BA set")
  }
  return ordered
}

async function putJson(
  pathname: string,
  value: unknown,
  options: { allowOverwrite: boolean },
): Promise<string> {
  if (!BLOB_TOKEN) throw new Error("BLOB_READ_WRITE_TOKEN not configured")
  const body = JSON.stringify(value)
  const blob = await put(pathname, body, {
    access: "private",
    token: BLOB_TOKEN,
    contentType: "application/json",
    addRandomSuffix: false,
    allowOverwrite: options.allowOverwrite,
    cacheControlMaxAge: 300,
  })
  return blob.url
}

async function putImmutableJson(pathname: string, value: unknown): Promise<string> {
  if (!BLOB_TOKEN) throw new Error("BLOB_READ_WRITE_TOKEN not configured")
  const expected = JSON.stringify(value)
  try {
    const existing = await head(pathname, { token: BLOB_TOKEN })
    const response = await fetch(existing.url, {
      headers: { authorization: `Bearer ${BLOB_TOKEN}` },
      cache: "no-store",
    })
    if (!response.ok || (await response.text()) !== expected) {
      throw new Error(`immutable blob conflict at ${pathname}`)
    }
    return existing.url
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("immutable blob conflict")) {
      throw error
    }
  }
  return putJson(pathname, value, { allowOverwrite: false })
}

async function assertCurrentPointer(
  expected: { run_id: string },
): Promise<void> {
  if (!BLOB_TOKEN) throw new Error("BLOB_READ_WRITE_TOKEN not configured")
  const metadata = await head("forecasts/v2/current.json", { token: BLOB_TOKEN })
  const freshUrl = new URL(metadata.url)
  // The stable pointer URL may still have the preceding version in a CDN
  // cache. A per-run query key plus no-store makes this a real release
  // assertion rather than a check of the prior bake.
  freshUrl.searchParams.set("surge_release", expected.run_id)
  const response = await fetch(freshUrl, {
    headers: { authorization: `Bearer ${BLOB_TOKEN}` },
    cache: "no-store",
  })
  if (!response.ok) throw new Error(`current pointer read ${response.status}`)
  const actual = (await response.json()) as { run_id?: string }
  if (actual.run_id !== expected.run_id) {
    throw new Error("current pointer read-after-write did not expose the published run")
  }
}

export async function POST(req: NextRequest): Promise<Response> {
  if (!BAKE_SECRET || !BLOB_TOKEN || !LEDGER_KEY) {
    return Response.json(
      { error: "BAKE_SECRET, BLOB_READ_WRITE_TOKEN, and SURGE_LEDGER_KEY are required" },
      { status: 500 },
    )
  }
  const auth = req.headers.get("authorization") ?? ""
  if (auth !== `Bearer ${BAKE_SECRET}`) {
    return Response.json({ error: "unauthorized" }, { status: 401 })
  }
  try {
    assertUpstream()
  } catch (error) {
    return Response.json({ error: String(error) }, { status: 500 })
  }

  const started = Date.now()
  const scheduledForUtc = scheduledSlot(new Date(started))
  let ordered: ForecastResponse[]
  try {
    ordered = await fetchLedgerBatch(scheduledForUtc, started)
  } catch (error) {
    return Response.json(
      {
        published: false,
        scheduled_for_utc: scheduledForUtc,
        expected: RTO_CODES.length,
        forecast_ok: 0,
        forecast_fail: RTO_CODES.length,
        ok: 0,
        fail: RTO_CODES.length,
        elapsed_ms: Date.now() - started,
        error: String(error),
        results: RTO_CODES.map((ba) => ({ ba, ok: false, error: String(error) })),
      },
      { status: 503 },
    )
  }

  const results: BakeResult[] = ordered.map((forecast) => ({
    ba: forecast.ba,
    ok: true,
    issuance_id: forecast.issuance_id,
  }))
  const forecastOk = ordered.length
  const runIds = new Set(ordered.map((forecast) => forecast.run_id))
  if (runIds.size !== 1) {
    return Response.json(
      { published: false, error: "forecast run IDs do not match", results },
      { status: 500 },
    )
  }
  const runId = ordered[0].run_id
  let provenance: RunProvenance
  try {
    provenance = assertUniformRunProvenance(ordered)
  } catch (error) {
    return Response.json(
      { published: false, run_id: runId, error: String(error), results },
      { status: 409 },
    )
  }
  const bakedAt = new Date().toISOString()
  const allPayload = {
    schema_version: "2.0",
    run_id: runId,
    baked_at: bakedAt,
    horizon: BAKE_HORIZON,
    provenance,
    forecasts: ordered,
  }

  let immutableAllUrl = ""
  let manifestUrl = ""
  let currentUrl = ""
  let pointerAdvanced = false
  try {
    const entries: Record<string, { issuance_id: string; url: string }> = {}
    for (const forecast of ordered) {
      const path = `forecasts/v2/runs/${runId}/${forecast.ba}.json`
      const url = await putImmutableJson(path, forecast)
      entries[forecast.ba] = { issuance_id: forecast.issuance_id, url }
    }
    immutableAllUrl = await putImmutableJson(
      `forecasts/v2/runs/${runId}/all.json`,
      allPayload,
    )
    const manifest = {
      schema_version: "2.0",
      run_id: runId,
      scheduled_for_utc: scheduledForUtc,
      published_at_utc: bakedAt,
      expected_regions: RTO_CODES.length,
      provenance,
      entries,
      all_url: immutableAllUrl,
    }
    manifestUrl = await putImmutableJson(
      `forecasts/v2/runs/${runId}/manifest.json`,
      manifest,
    )
    const currentPointer = {
      schema_version: "2.0",
      run_id: runId,
      published_at_utc: bakedAt,
      manifest_url: manifestUrl,
      all_url: immutableAllUrl,
    }
    currentUrl = await putJson(
      "forecasts/v2/current.json",
      currentPointer,
      { allowOverwrite: true },
    )
    pointerAdvanced = true
    await assertCurrentPointer(currentPointer)
  } catch (error) {
    return Response.json(
      {
        published: pointerAdvanced,
        run_id: runId,
        expected: RTO_CODES.length,
        forecast_ok: forecastOk,
        forecast_fail: 0,
        error: String(error),
        publication_boundary: pointerAdvanced ? "advanced-unverified" : "not-advanced",
        elapsed_ms: Date.now() - started,
        results,
      },
      { status: 500 },
    )
  }

  // Compatibility mirrors are explicitly downstream of the v2 publication
  // boundary. Their failure cannot roll back or contradict current.json.
  const mirrorErrors: string[] = []
  for (const forecast of ordered) {
    try {
      await putJson(`forecasts/${forecast.ba}.json`, forecast, {
        allowOverwrite: true,
      })
    } catch (error) {
      mirrorErrors.push(`${forecast.ba}: ${String(error)}`)
    }
  }
  let legacyAllUrl: string | null = null
  try {
    legacyAllUrl = await putJson("forecasts/all.json", allPayload, {
      allowOverwrite: true,
    })
  } catch (error) {
    mirrorErrors.push(`all: ${String(error)}`)
  }

  return Response.json(
    {
      published: true,
      run_id: runId,
      scheduled_for_utc: scheduledForUtc,
      expected: RTO_CODES.length,
      forecast_ok: forecastOk,
      forecast_fail: 0,
      ok: forecastOk,
      fail: 0,
      manifest_url: manifestUrl,
      all_url: immutableAllUrl,
      legacy_all_url: legacyAllUrl,
      current_url: currentUrl,
      read_after_write_verified: true,
      compatibility_mirror_errors: mirrorErrors,
      elapsed_ms: Date.now() - started,
      results,
    },
    { status: 200 },
  )
}
