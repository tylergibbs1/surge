"use client"

import { useMemo, useState } from "react"

import { BaCard } from "@/components/ba-card"
import {
  BA_LABEL,
  BA_PEAK_MW,
  BAS,
  type BaCode,
} from "@/lib/us-grid-geo"

// Same color map the grid-map.tsx uses. Inlined rather than exported to
// avoid the cross-file dependency churn — 7 RTO accents + two muted
// interconnect tints for everything else.
const ISO_COLOR: Partial<Record<BaCode, string>> = {
  PJM:  "#4F8DF7", CISO: "#F4B740", ERCO: "#E05263", MISO: "#5CC8A2",
  NYIS: "#A770F0", ISNE: "#F07FB0", SWPP: "#6BC4E0",
}
const EAST_TINT = "#7A8B6E"
const WEST_TINT = "#6E8A8B"

// Mirror of the INTERCONNECT map in grid-map.tsx. Duplicated in the name
// of loose coupling — if they ever drift the grid will just render the
// wrong pill, not crash.
const INTERCONNECT: Record<BaCode, "Eastern" | "Western" | "Texas"> = {
  PJM: "Eastern", CISO: "Western", ERCO: "Texas", MISO: "Eastern",
  NYIS: "Eastern", ISNE: "Eastern", SWPP: "Eastern",
  SOCO: "Eastern", TVA: "Eastern", FPL: "Eastern", DUK: "Eastern",
  CPLE: "Eastern", BPAT: "Western", FPC: "Eastern", AZPS: "Western",
  LGEE: "Eastern", PSCO: "Western", SRP: "Western", NEVP: "Western",
  PACE: "Western", LDWP: "Western", SCEG: "Eastern", PSEI: "Western",
  SC: "Eastern", TEC: "Eastern", AECI: "Eastern", PGE: "Western",
  IPCO: "Western", FMPP: "Eastern", PACW: "Western", WACM: "Western",
  BANC: "Western", JEA: "Eastern", TEPC: "Western", SEC: "Eastern",
  WALC: "Western", CPLW: "Eastern", EPE: "Western", PNM: "Western",
  NWMT: "Western", AVA: "Western", SCL: "Western", SPA: "Eastern",
  IID: "Western", TPWR: "Western", WAUW: "Western", TAL: "Eastern",
  TIDC: "Western", GCPD: "Western", GVL: "Eastern", CHPD: "Western",
  DOPD: "Western", HST: "Eastern",
}

const RTO_SET = new Set<BaCode>([
  "PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP",
])

// Size-tier buckets used by the sidebar filter. Picked by peak_mw so the
// biggest non-RTO utilities (SOCO, TVA, DUK, FPL, ...) have their own row.
function sizeTier(ba: BaCode): "RTO" | "Utility" | "Small" {
  if (RTO_SET.has(ba)) return "RTO"
  return (BA_PEAK_MW[ba] ?? 0) >= 5_000 ? "Utility" : "Small"
}

function accent(ba: BaCode): string {
  return (
    ISO_COLOR[ba] ??
    (INTERCONNECT[ba] === "Western" ? WEST_TINT : EAST_TINT)
  )
}

type ForecastPoint = {
  ts_utc: string
  median_mw: number
  p10_mw: number
  p90_mw: number
}
type Forecast = {
  ba: string
  points: ForecastPoint[]
}
type Sort = "peak-pct" | "peak-mw" | "name"
type InterconnectFilter = "All" | "Eastern" | "Western" | "Texas"
type TierFilter = "All" | "RTO" | "Utility" | "Small"

function peak24(points: ForecastPoint[]): number {
  let max = 0
  for (let i = 0; i < Math.min(24, points.length); i++) {
    if (points[i].median_mw > max) max = points[i].median_mw
  }
  return max
}

export function BaGrid({
  forecasts,
  bakedAt,
}: {
  forecasts: Forecast[]
  bakedAt?: string
}) {
  const [interFilter, setInterFilter] = useState<InterconnectFilter>("All")
  const [tierFilter, setTierFilter] = useState<TierFilter>("All")
  const [sort, setSort] = useState<Sort>("peak-pct")

  // Index by BA for O(1) lookup; drop any forecasts for BA codes we don't
  // know (defensive — registry and bake output should match, but network
  // intermediate state during a rolling deploy could expose us).
  const byBa = useMemo(() => {
    const m = new Map<BaCode, Forecast>()
    for (const f of forecasts) {
      if ((BAS as readonly string[]).includes(f.ba)) {
        m.set(f.ba as BaCode, f)
      }
    }
    return m
  }, [forecasts])

  const visible = useMemo(() => {
    const rows = BAS.flatMap((ba) => {
      const f = byBa.get(ba)
      if (!f) return []
      if (interFilter !== "All" && INTERCONNECT[ba] !== interFilter) return []
      if (tierFilter !== "All" && sizeTier(ba) !== tierFilter) return []
      const p24 = peak24(f.points)
      const pct = BA_PEAK_MW[ba] ? (p24 / BA_PEAK_MW[ba]) * 100 : 0
      return [{ ba, forecast: f, peak24Mw: p24, pctOfPeak: pct }]
    })

    if (sort === "peak-pct") rows.sort((a, b) => b.pctOfPeak - a.pctOfPeak)
    else if (sort === "peak-mw") rows.sort((a, b) => b.peak24Mw - a.peak24Mw)
    else rows.sort((a, b) => BA_LABEL[a.ba].localeCompare(BA_LABEL[b.ba]))

    return rows
  }, [byBa, interFilter, tierFilter, sort])

  const total = BAS.length
  return (
    <div className="grid gap-6 md:grid-cols-[220px_minmax(0,1fr)]">
      <aside className="space-y-6">
        <FilterGroup
          label="Interconnection"
          options={["All", "Eastern", "Western", "Texas"] as const}
          value={interFilter}
          onChange={setInterFilter}
        />
        <FilterGroup
          label="Size"
          options={["All", "RTO", "Utility", "Small"] as const}
          value={tierFilter}
          onChange={setTierFilter}
          counts={{
            All: total,
            RTO: 7,
            Utility: BAS.filter((b) => sizeTier(b) === "Utility").length,
            Small: BAS.filter((b) => sizeTier(b) === "Small").length,
          }}
        />
        <div className="space-y-2">
          <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
            Sort
          </p>
          <div className="flex flex-col gap-1">
            <SortOption value="peak-pct" active={sort === "peak-pct"} onSelect={setSort}>
              % of all-time peak
            </SortOption>
            <SortOption value="peak-mw" active={sort === "peak-mw"} onSelect={setSort}>
              Peak load (GW)
            </SortOption>
            <SortOption value="name" active={sort === "name"} onSelect={setSort}>
              Name
            </SortOption>
          </div>
        </div>
        {bakedAt ? (
          <p className="pt-3 text-[10px] text-muted-foreground tabular-nums">
            Baked {formatRelative(bakedAt)}
          </p>
        ) : null}
      </aside>

      <section className="space-y-3">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>
            Showing <span className="tabular-nums text-foreground">{visible.length}</span>{" "}
            of {total} balancing authorities
          </span>
          {interFilter !== "All" || tierFilter !== "All" ? (
            <button
              type="button"
              onClick={() => {
                setInterFilter("All")
                setTierFilter("All")
              }}
              className="rounded-sm text-xs underline-offset-4 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-foreground/50"
            >
              Clear filters
            </button>
          ) : null}
        </div>
        {visible.length === 0 ? (
          <div className="rounded-xl border border-dashed border-foreground/15 p-8 text-center text-sm text-muted-foreground">
            No BAs match these filters.
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {visible.map(({ ba, forecast }) => (
              <BaCard
                key={ba}
                ba={ba}
                points={forecast.points}
                interconnect={INTERCONNECT[ba]}
                isRto={RTO_SET.has(ba)}
                accentHex={accent(ba)}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

function FilterGroup<T extends string>({
  label,
  options,
  value,
  onChange,
  counts,
}: {
  label: string
  options: readonly T[]
  value: T
  onChange: (v: T) => void
  counts?: Partial<Record<T, number>>
}) {
  return (
    <div className="space-y-2">
      <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      <div className="flex flex-col gap-1">
        {options.map((opt) => {
          const active = value === opt
          return (
            <button
              key={opt}
              type="button"
              onClick={() => onChange(opt)}
              className={`flex items-center justify-between rounded-md px-2 py-1.5 text-left text-xs transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-foreground/50 ${
                active
                  ? "bg-foreground/8 font-medium text-foreground"
                  : "text-muted-foreground hover:bg-foreground/5 hover:text-foreground"
              }`}
              aria-pressed={active}
            >
              <span>{opt}</span>
              {counts?.[opt] !== undefined ? (
                <span className="font-mono text-[10px] tabular-nums text-muted-foreground">
                  {counts[opt]}
                </span>
              ) : null}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function SortOption<T extends string>({
  value,
  active,
  onSelect,
  children,
}: {
  value: T
  active: boolean
  onSelect: (v: T) => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(value)}
      className={`rounded-md px-2 py-1.5 text-left text-xs transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-foreground/50 ${
        active
          ? "bg-foreground/8 font-medium text-foreground"
          : "text-muted-foreground hover:bg-foreground/5 hover:text-foreground"
      }`}
      aria-pressed={active}
    >
      {children}
    </button>
  )
}

function formatRelative(isoUtc: string): string {
  const ms = Date.now() - Date.parse(isoUtc)
  if (!Number.isFinite(ms)) return ""
  const min = Math.round(ms / 60_000)
  if (min < 1) return "just now"
  if (min < 60) return `${min} min ago`
  const h = Math.round(min / 60)
  if (h < 24) return `${h} h ago`
  return `${Math.round(h / 24)} d ago`
}
