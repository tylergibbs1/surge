/**
 * Public, presentation-ready contracts for the v0.2 forecast scoreboard.
 *
 * The adapter in `lib/server/load-scoreboard.ts` resolves the immutable v2
 * run through its validated current pointer and leaves evidence that does not
 * exist yet as `null`.
 */

export const SCOREBOARD_SCHEMA_VERSION = "2.0" as const

export const RTO_CODES = [
  "PJM",
  "CISO",
  "ERCO",
  "MISO",
  "NYIS",
  "ISNE",
  "SWPP",
] as const

export type RtoCode = (typeof RTO_CODES)[number]

export type Interconnection = "Eastern" | "Western" | "Texas"

export type DataFreshnessState = "fresh" | "delayed" | "stale" | "unavailable"

export type ScoreboardTrendPoint = {
  validAtUtc: string
  medianMw: number
  p10Mw: number
  p90Mw: number
}

export type ScoreboardPeak = {
  medianMw: number
  p10Mw: number
  p90Mw: number
  validAtUtc: string
  pointCount: number
}

export type ScoreboardRegion = {
  code: RtoCode
  name: string
  shortName: string
  timezone: string
  interconnection: Interconnection
  detailHref: string
  state: DataFreshnessState
  issueAtUtc: string | null
  issueAgeHours: number | null
  coverageStartsAtUtc: string | null
  coverageEndsAtUtc: string | null
  next24hPeak: ScoreboardPeak | null
  next24hTrend: ScoreboardTrendPoint[]
  referencePeakMw: number
  warnings: string[]
}

export type ScoreboardSource = {
  kind: "vercel-blob-snapshot"
  artifactPath: "forecasts/v2/current.json"
  bakedAtUtc: string | null
  artifactAgeHours: number | null
  state: DataFreshnessState
}

export type ScoreboardSnapshot = {
  schemaVersion: typeof SCOREBOARD_SCHEMA_VERSION
  generatedAtUtc: string
  overallState: DataFreshnessState
  source: ScoreboardSource
  expectedRegions: number
  availableRegions: number
  currentRegions: number
  regions: ScoreboardRegion[]
  notice: string
}
