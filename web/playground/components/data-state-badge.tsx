import { FRESHNESS_LABEL } from "@/lib/forecast-display"
import type { DataFreshnessState } from "@/lib/v2-contracts"

const STATUS_STYLES: Record<DataFreshnessState, string> = {
  fresh:
    "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-200",
  delayed:
    "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-200",
  stale:
    "border-orange-300 bg-orange-50 text-orange-900 dark:border-orange-800 dark:bg-orange-950/50 dark:text-orange-200",
  unavailable:
    "border-slate-300 bg-slate-100 text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200",
}

export function DataStateBadge({ state }: { state: DataFreshnessState }) {
  return (
    <span
      className={`inline-flex items-center border px-2 py-1 text-[11px] font-semibold tracking-[0.08em] uppercase ${STATUS_STYLES[state]}`}
    >
      {FRESHNESS_LABEL[state]}
    </span>
  )
}
