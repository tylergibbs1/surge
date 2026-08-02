import type { DataFreshnessState } from "@/lib/v2-contracts"

export const FRESHNESS_THRESHOLDS = {
  freshHours: 26,
  staleHours: 36,
  minimumCurrentWindowHours: 18,
  expectedNext24hPoints: 24,
} as const

const DATE_FORMATTERS = new Map<string, Intl.DateTimeFormat>()

function validDate(value: string | Date): Date | null {
  const date = value instanceof Date ? value : new Date(value)
  return Number.isFinite(date.getTime()) ? date : null
}

export function hoursBetween(later: Date, earlier: Date): number {
  return (later.getTime() - earlier.getTime()) / 3_600_000
}

export function classifyForecastFreshness({
  issueAtUtc,
  coverageEndsAtUtc,
  next24hPointCount,
  now,
}: {
  issueAtUtc: string | null
  coverageEndsAtUtc: string | null
  next24hPointCount: number
  now: Date
}): DataFreshnessState {
  const issueAt = issueAtUtc ? validDate(issueAtUtc) : null
  const coverageEnd = coverageEndsAtUtc ? validDate(coverageEndsAtUtc) : null

  if (!issueAt || !coverageEnd) return "unavailable"

  const ageHours = hoursBetween(now, issueAt)
  const remainingCoverageHours = hoursBetween(coverageEnd, now)

  if (
    ageHours < -1 ||
    ageHours > FRESHNESS_THRESHOLDS.staleHours ||
    next24hPointCount === 0 ||
    remainingCoverageHours < FRESHNESS_THRESHOLDS.minimumCurrentWindowHours
  ) {
    return "stale"
  }

  if (
    ageHours > FRESHNESS_THRESHOLDS.freshHours ||
    next24hPointCount < FRESHNESS_THRESHOLDS.expectedNext24hPoints
  ) {
    return "delayed"
  }

  return "fresh"
}

export function classifyArtifactFreshness(
  bakedAtUtc: string | null,
  now: Date
): DataFreshnessState {
  const bakedAt = bakedAtUtc ? validDate(bakedAtUtc) : null
  if (!bakedAt) return "unavailable"

  const ageHours = hoursBetween(now, bakedAt)
  if (ageHours < -1 || ageHours > FRESHNESS_THRESHOLDS.staleHours) {
    return "stale"
  }
  if (ageHours > FRESHNESS_THRESHOLDS.freshHours) return "delayed"
  return "fresh"
}

export function formatPower(megawatts: number | null): string {
  if (megawatts === null || !Number.isFinite(megawatts)) return "—"
  if (Math.abs(megawatts) >= 1_000) {
    return `${new Intl.NumberFormat("en-US", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    }).format(megawatts / 1_000)} GW`
  }
  return `${new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(megawatts)} MW`
}

export function formatLocalDateTime(
  value: string | null,
  timezone: string
): string {
  if (!value) return "—"
  const date = validDate(value)
  if (!date) return "—"

  let formatter = DATE_FORMATTERS.get(timezone)
  if (!formatter) {
    formatter = new Intl.DateTimeFormat("en-US", {
      timeZone: timezone,
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      timeZoneName: "short",
    })
    DATE_FORMATTERS.set(timezone, formatter)
  }
  return formatter.format(date)
}

export function formatUtcDateTime(value: string | null): string {
  if (!value) return "—"
  const date = validDate(value)
  if (!date) return "—"
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZoneName: "short",
  }).format(date)
}

export function formatAge(ageHours: number | null): string {
  if (ageHours === null || !Number.isFinite(ageHours)) return "Unknown age"
  if (ageHours < 0) return "Timestamp is in the future"
  if (ageHours < 1) return `${Math.max(1, Math.round(ageHours * 60))} min ago`
  if (ageHours < 48) return `${Math.round(ageHours)} hr ago`
  return `${Math.round(ageHours / 24)} days ago`
}

export const FRESHNESS_LABEL: Record<DataFreshnessState, string> = {
  fresh: "Current",
  delayed: "Delayed",
  stale: "Stale",
  unavailable: "Unavailable",
}

export const FRESHNESS_DESCRIPTION: Record<DataFreshnessState, string> = {
  fresh: "The snapshot is recent and contains a complete next-24-hour window.",
  delayed: "The snapshot is aging or its next-24-hour window is incomplete.",
  stale:
    "A regional forecast or the source snapshot is too old, or no longer covers a useful current window.",
  unavailable: "No valid forecast snapshot is available for this region.",
}
