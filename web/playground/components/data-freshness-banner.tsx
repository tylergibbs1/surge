import {
  FRESHNESS_DESCRIPTION,
  FRESHNESS_LABEL,
  formatAge,
  formatUtcDateTime,
} from "@/lib/forecast-display"
import type { ScoreboardSnapshot } from "@/lib/v2-contracts"

const STATE_STYLES = {
  fresh: {
    shell:
      "border-emerald-300/70 bg-emerald-50 text-emerald-950 dark:border-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-100",
    dot: "bg-emerald-600 dark:bg-emerald-400",
  },
  delayed: {
    shell:
      "border-amber-300/80 bg-amber-50 text-amber-950 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-100",
    dot: "bg-amber-600 dark:bg-amber-400",
  },
  stale: {
    shell:
      "border-orange-300/80 bg-orange-50 text-orange-950 dark:border-orange-700 dark:bg-orange-950/40 dark:text-orange-100",
    dot: "bg-orange-600 dark:bg-orange-400",
  },
  unavailable: {
    shell:
      "border-slate-300 bg-slate-100 text-slate-950 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100",
    dot: "bg-slate-500 dark:bg-slate-400",
  },
} as const

export function DataFreshnessBanner({
  snapshot,
}: {
  snapshot: ScoreboardSnapshot
}) {
  const style = STATE_STYLES[snapshot.overallState]
  const bakedAt = snapshot.source.bakedAtUtc

  return (
    <section
      aria-labelledby="freshness-heading"
      className={`border px-4 py-4 sm:px-5 ${style.shell}`}
    >
      <div className="flex items-start gap-3">
        <span
          className={`mt-1.5 size-2.5 shrink-0 rounded-full ${style.dot}`}
          aria-hidden="true"
        />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline justify-between gap-x-5 gap-y-1">
            <h2 id="freshness-heading" className="text-sm font-semibold">
              Forecast data: {FRESHNESS_LABEL[snapshot.overallState]}
            </h2>
            <p className="font-mono text-xs">
              {bakedAt ? (
                <>
                  Snapshot{" "}
                  <time dateTime={bakedAt}>{formatUtcDateTime(bakedAt)}</time>
                  {" · "}
                  {formatAge(snapshot.source.artifactAgeHours)}
                </>
              ) : (
                "No snapshot timestamp"
              )}
            </p>
          </div>
          <p className="mt-1 text-sm leading-6">{snapshot.notice}</p>
          <p className="mt-1 text-xs opacity-75">
            {FRESHNESS_DESCRIPTION[snapshot.overallState]}
          </p>
        </div>
      </div>
    </section>
  )
}
