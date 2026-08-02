import { head } from "@vercel/blob"

import {
  classifyArtifactFreshness,
  classifyForecastFreshness,
  hoursBetween,
} from "@/lib/forecast-display"
import { isRtoCode, RTOS, type RtoMetadata } from "@/lib/rto-metadata"
import {
  SCOREBOARD_SCHEMA_VERSION,
  type DataFreshnessState,
  type RtoCode,
  type ScoreboardRegion,
  type ScoreboardSnapshot,
  type ScoreboardTrendPoint,
} from "@/lib/v2-contracts"

const ARTIFACT_PATH = "forecasts/v2/current.json" as const
const HOUR_MS = 3_600_000
const NEXT_24_HOURS_MS = 24 * HOUR_MS

type CurrentPointer = {
  schema_version: "2.0"
  run_id: string
  published_at_utc: string
  all_url: string
}

type V2AllPayload = LegacyAllPayload & {
  schema_version: "2.0"
  run_id: string
  horizon: number
}

type LegacyForecastPoint = {
  ts_utc: string
  median_mw: number
  p10_mw: number
  p90_mw: number
}

type LegacyForecast = {
  ba: string
  as_of_utc: string
  points: LegacyForecastPoint[]
}

type LegacyAllPayload = {
  baked_at: string
  forecasts: LegacyForecast[]
}

function validateBlobUrl(value: string): string {
  const url = new URL(value)
  if (
    url.protocol !== "https:" ||
    !url.hostname.endsWith(".blob.vercel-storage.com")
  ) {
    throw new Error("current pointer references an unexpected blob host")
  }
  return url.toString()
}

function validateCurrentPointer(value: unknown): CurrentPointer {
  if (!isRecord(value)) throw new Error("invalid current pointer")
  if (
    value.schema_version !== "2.0" ||
    typeof value.run_id !== "string" ||
    value.run_id.length === 0 ||
    !isIsoDate(value.published_at_utc) ||
    typeof value.all_url !== "string"
  ) {
    throw new Error("invalid current pointer")
  }
  return {
    schema_version: "2.0",
    run_id: value.run_id,
    published_at_utc: value.published_at_utc,
    all_url: validateBlobUrl(value.all_url),
  }
}

function validateV2Payload(value: unknown, pointer: CurrentPointer): V2AllPayload {
  if (!isRecord(value)) throw new Error("invalid v2 forecast payload")
  const parsed = parsePayload(value)
  if (
    value.schema_version !== "2.0" ||
    value.run_id !== pointer.run_id ||
    value.horizon !== 168 ||
    parsed === null ||
    parsed.forecasts.length !== RTOS.length
  ) {
    throw new Error("incomplete v2 forecast payload")
  }
  const codes = new Set(parsed.forecasts.map((forecast) => forecast.ba))
  if (
    codes.size !== RTOS.length ||
    RTOS.some((region) => !codes.has(region.code)) ||
    parsed.forecasts.some((forecast) => forecast.points.length !== 168)
  ) {
    throw new Error("incomplete v2 forecast payload")
  }
  return {
    ...parsed,
    schema_version: "2.0",
    run_id: pointer.run_id,
    horizon: 168,
  }
}

async function fetchPrivateJson(url: string, token: string): Promise<unknown> {
  const response = await fetch(url, {
    headers: { authorization: `Bearer ${token}` },
    next: { revalidate: 300 },
  })
  if (!response.ok) throw new Error(`blob read ${response.status}`)
  return response.json()
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function isIsoDate(value: unknown): value is string {
  return typeof value === "string" && Number.isFinite(Date.parse(value))
}

function parsePoint(value: unknown): LegacyForecastPoint | null {
  if (!isRecord(value)) return null
  if (
    !isIsoDate(value.ts_utc) ||
    !isFiniteNumber(value.median_mw) ||
    !isFiniteNumber(value.p10_mw) ||
    !isFiniteNumber(value.p90_mw) ||
    value.p10_mw < 0 ||
    value.median_mw < 0 ||
    value.p90_mw < 0 ||
    value.p10_mw > value.median_mw ||
    value.median_mw > value.p90_mw
  ) {
    return null
  }
  return {
    ts_utc: value.ts_utc,
    median_mw: value.median_mw,
    p10_mw: value.p10_mw,
    p90_mw: value.p90_mw,
  }
}

function parseForecast(value: unknown): LegacyForecast | null {
  if (!isRecord(value) || typeof value.ba !== "string") return null
  if (!isIsoDate(value.as_of_utc) || !Array.isArray(value.points)) return null

  const pointByTimestamp = new Map<string, LegacyForecastPoint>()
  for (const candidate of value.points) {
    const point = parsePoint(candidate)
    if (point && !pointByTimestamp.has(point.ts_utc)) {
      pointByTimestamp.set(point.ts_utc, point)
    }
  }

  const points = [...pointByTimestamp.values()].sort(
    (a, b) => Date.parse(a.ts_utc) - Date.parse(b.ts_utc)
  )
  if (points.length === 0) return null

  return {
    ba: value.ba.toUpperCase(),
    as_of_utc: value.as_of_utc,
    points,
  }
}

function parsePayload(value: unknown): LegacyAllPayload | null {
  if (!isRecord(value) || !isIsoDate(value.baked_at)) return null
  if (!Array.isArray(value.forecasts)) return null

  const forecasts = value.forecasts
    .map(parseForecast)
    .filter((forecast): forecast is LegacyForecast => forecast !== null)

  return { baked_at: value.baked_at, forecasts }
}

function emptyRegion(metadata: RtoMetadata, warning: string): ScoreboardRegion {
  return {
    ...metadata,
    state: "unavailable",
    issueAtUtc: null,
    issueAgeHours: null,
    coverageStartsAtUtc: null,
    coverageEndsAtUtc: null,
    next24hPeak: null,
    next24hTrend: [],
    warnings: [warning],
  }
}

function hasCompleteHourlyWindow(points: LegacyForecastPoint[]): boolean {
  if (points.length !== 24) return false
  return points.every((point, index) => {
    if (index === 0) return true
    return (
      Date.parse(point.ts_utc) - Date.parse(points[index - 1].ts_utc) ===
      HOUR_MS
    )
  })
}

function asTrendPoint(point: LegacyForecastPoint): ScoreboardTrendPoint {
  return {
    validAtUtc: point.ts_utc,
    medianMw: point.median_mw,
    p10Mw: point.p10_mw,
    p90Mw: point.p90_mw,
  }
}

function applyArtifactState(
  forecastState: DataFreshnessState,
  artifactState: DataFreshnessState
): DataFreshnessState {
  if (forecastState === "unavailable") return "unavailable"
  if (artifactState === "stale") return "stale"
  if (artifactState === "delayed" && forecastState === "fresh") return "delayed"
  return forecastState
}

function deriveRegion(
  metadata: RtoMetadata,
  forecast: LegacyForecast | undefined,
  artifactState: DataFreshnessState,
  now: Date
): ScoreboardRegion {
  if (!forecast) {
    return emptyRegion(metadata, "No valid forecast was found in the snapshot.")
  }

  const coverageStartsAtUtc = forecast.points.at(0)?.ts_utc ?? null
  const coverageEndsAtUtc = forecast.points.at(-1)?.ts_utc ?? null
  const windowEnd = now.getTime() + NEXT_24_HOURS_MS
  const next24hPoints = forecast.points.filter((point) => {
    const timestamp = Date.parse(point.ts_utc)
    return timestamp > now.getTime() && timestamp <= windowEnd
  })
  const completeWindow = hasCompleteHourlyWindow(next24hPoints)

  let state = classifyForecastFreshness({
    issueAtUtc: forecast.as_of_utc,
    coverageEndsAtUtc,
    next24hPointCount: completeWindow
      ? next24hPoints.length
      : Math.min(23, next24hPoints.length),
    now,
  })
  state = applyArtifactState(state, artifactState)

  const warnings: string[] = []
  if (state === "stale") {
    warnings.push(
      "This forecast is not current and should not be used operationally."
    )
  } else if (state === "delayed") {
    warnings.push(
      "This forecast is delayed or has incomplete current coverage."
    )
  }
  if (next24hPoints.length > 0 && !completeWindow) {
    warnings.push("A complete 24-point hourly window is not available.")
  }

  const peakPoint = completeWindow
    ? next24hPoints.reduce((peak, point) =>
        point.median_mw > peak.median_mw ? point : peak
      )
    : null

  return {
    ...metadata,
    state,
    issueAtUtc: forecast.as_of_utc,
    issueAgeHours: hoursBetween(now, new Date(forecast.as_of_utc)),
    coverageStartsAtUtc,
    coverageEndsAtUtc,
    next24hPeak: peakPoint
      ? {
          medianMw: peakPoint.median_mw,
          p10Mw: peakPoint.p10_mw,
          p90Mw: peakPoint.p90_mw,
          validAtUtc: peakPoint.ts_utc,
          pointCount: next24hPoints.length,
        }
      : null,
    next24hTrend: completeWindow ? next24hPoints.map(asTrendPoint) : [],
    warnings,
  }
}

function overallState(regions: ScoreboardRegion[]): DataFreshnessState {
  const available = regions.filter((region) => region.state !== "unavailable")
  if (available.length === 0) return "unavailable"
  if (available.some((region) => region.state === "stale")) return "stale"
  if (
    available.some((region) => region.state === "delayed") ||
    available.length !== regions.length
  ) {
    return "delayed"
  }
  return "fresh"
}

function noticeForState(state: DataFreshnessState): string {
  switch (state) {
    case "fresh":
      return "All seven regional forecasts are current with complete next-24-hour coverage."
    case "delayed":
      return "At least one regional forecast is delayed, incomplete, or unavailable. Check row-level timestamps before use."
    case "stale":
      return "At least one regional forecast or the source snapshot is stale. Historical timestamps remain visible, but current outlook values are withheld when coverage is incomplete."
    case "unavailable":
      return "No valid regional forecast snapshot is available. The page is showing explicit unavailable states instead of substituting zeros."
  }
}

function unavailableSnapshot(now: Date, reason: string): ScoreboardSnapshot {
  const regions = RTOS.map((metadata) => emptyRegion(metadata, reason))
  return {
    schemaVersion: SCOREBOARD_SCHEMA_VERSION,
    generatedAtUtc: now.toISOString(),
    overallState: "unavailable",
    source: {
      kind: "vercel-blob-snapshot",
      artifactPath: ARTIFACT_PATH,
      bakedAtUtc: null,
      artifactAgeHours: null,
      state: "unavailable",
    },
    expectedRegions: RTOS.length,
    availableRegions: 0,
    currentRegions: 0,
    regions,
    notice: noticeForState("unavailable"),
  }
}

export function buildScoreboardSnapshot(
  input: unknown,
  now = new Date()
): ScoreboardSnapshot {
  const payload = parsePayload(input)
  if (!payload) {
    return unavailableSnapshot(
      now,
      "The forecast snapshot did not match the expected v1 payload."
    )
  }

  const artifactState = classifyArtifactFreshness(payload.baked_at, now)
  const forecasts = new Map<RtoCode, LegacyForecast>()
  for (const forecast of payload.forecasts) {
    if (!isRtoCode(forecast.ba)) continue
    const existing = forecasts.get(forecast.ba)
    if (
      !existing ||
      Date.parse(forecast.as_of_utc) > Date.parse(existing.as_of_utc)
    ) {
      forecasts.set(forecast.ba, forecast)
    }
  }

  const regions = RTOS.map((metadata) =>
    deriveRegion(metadata, forecasts.get(metadata.code), artifactState, now)
  )
  const state = overallState(regions)
  const bakedAt = new Date(payload.baked_at)

  return {
    schemaVersion: SCOREBOARD_SCHEMA_VERSION,
    generatedAtUtc: now.toISOString(),
    overallState: state,
    source: {
      kind: "vercel-blob-snapshot",
      artifactPath: ARTIFACT_PATH,
      bakedAtUtc: payload.baked_at,
      artifactAgeHours: hoursBetween(now, bakedAt),
      state: artifactState,
    },
    expectedRegions: RTOS.length,
    availableRegions: regions.filter((region) => region.issueAtUtc !== null)
      .length,
    currentRegions: regions.filter((region) => region.state === "fresh").length,
    regions,
    notice: noticeForState(state),
  }
}

export async function loadScoreboardSnapshot(): Promise<ScoreboardSnapshot> {
  const now = new Date()
  const token = process.env.BLOB_READ_WRITE_TOKEN
  if (!token) {
    return unavailableSnapshot(
      now,
      "Forecast snapshot storage is not configured in this environment."
    )
  }

  let pointerUrl: string
  try {
    const metadata = await head(ARTIFACT_PATH, { token })
    pointerUrl = metadata.url
  } catch {
    return unavailableSnapshot(
      now,
      "The current forecast pointer could not be located."
    )
  }

  try {
    const pointer = validateCurrentPointer(
      await fetchPrivateJson(pointerUrl, token)
    )
    const payload = validateV2Payload(
      await fetchPrivateJson(pointer.all_url, token),
      pointer
    )
    return buildScoreboardSnapshot(payload, now)
  } catch {
    return unavailableSnapshot(
      now,
      "No complete forecast run could be resolved through the current pointer."
    )
  }
}
