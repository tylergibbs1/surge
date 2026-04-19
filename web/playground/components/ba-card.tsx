// One BA card for the /grid view. Dense enough to get 53 on screen with
// filters, informative enough to rank at a glance:
//   - BA code + full name
//   - Interconnect pill + RTO badge
//   - Tomorrow's forecast peak (max over next 24 h) in GW
//   - That peak as a % of the BA's published all-time peak (our "utilisation")
//   - 24 h median sparkline with p10/p90 shading
// Click → /?ba=XXX&horizon=24 so the user lands back in the map view with
// the card's BA selected.

import Link from "next/link"

import { BA_LABEL, BA_PEAK_MW, type BaCode } from "@/lib/us-grid-geo"

type ForecastPoint = {
  ts_utc: string
  median_mw: number
  p10_mw: number
  p90_mw: number
}

type Props = {
  ba: BaCode
  points: ForecastPoint[]  // at least 24 — we render the first 24
  interconnect: "Eastern" | "Western" | "Texas"
  isRto: boolean
  accentHex: string
}

const INTERCONNECT_PILL: Record<Props["interconnect"], string> = {
  Eastern: "bg-[#7a8b6e]/15 text-[#56684a] ring-[#7a8b6e]/30",
  Western: "bg-[#6e8a8b]/15 text-[#4a6264] ring-[#6e8a8b]/30",
  Texas:   "bg-[#e05263]/15 text-[#b33442] ring-[#e05263]/30",
}

function Sparkline({
  points,
  accentHex,
}: {
  points: ForecastPoint[]
  accentHex: string
}) {
  // Fixed viewBox; SVG scales to container width. 24 points across 240 px
  // ≈ 10 px per hour — enough to read diurnal shape in a card.
  const W = 240
  const H = 48
  const xs = points.map((_, i) => (i / (points.length - 1)) * W)

  // Shared y-scale across p10/median/p90 so the band reads at a glance.
  const lo = Math.min(...points.map((p) => p.p10_mw))
  const hi = Math.max(...points.map((p) => p.p90_mw))
  const span = hi - lo || 1
  const y = (v: number) => H - ((v - lo) / span) * (H - 4) - 2

  const linePath = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xs[i].toFixed(1)} ${y(p.median_mw).toFixed(1)}`)
    .join(" ")

  // Band polygon: p90 top → p10 bottom, reversed.
  const bandTop = points.map((p, i) => `${xs[i].toFixed(1)},${y(p.p90_mw).toFixed(1)}`)
  const bandBot = points
    .map((p, i) => `${xs[i].toFixed(1)},${y(p.p10_mw).toFixed(1)}`)
    .reverse()
  const bandPoints = [...bandTop, ...bandBot].join(" ")

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className="h-12 w-full"
      aria-hidden="true"
    >
      <polygon points={bandPoints} fill={accentHex} opacity={0.18} />
      <path d={linePath} fill="none" stroke={accentHex} strokeWidth={1.5} />
    </svg>
  )
}

export function BaCard({ ba, points, interconnect, isRto, accentHex }: Props) {
  const next24 = points.slice(0, 24)
  const peak24 = next24.reduce((m, p) => Math.max(m, p.median_mw), 0)
  const allTimePeak = BA_PEAK_MW[ba]
  const pctOfPeak = allTimePeak ? (peak24 / allTimePeak) * 100 : 0

  // Status dot: how hot the next 24 h look relative to all-time peak.
  // Coarser than the numeric % so the eye catches it fast.
  const status =
    pctOfPeak >= 85 ? "near-peak"
    : pctOfPeak >= 55 ? "moderate"
    : "low"
  const statusColor =
    status === "near-peak" ? "bg-red-500"
    : status === "moderate" ? "bg-amber-500"
    : "bg-emerald-500"

  return (
    <Link
      href={`/?ba=${ba}&horizon=24`}
      className="group relative flex flex-col gap-3 rounded-xl bg-card p-4 text-card-foreground ring-1 ring-foreground/10 transition hover:ring-foreground/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-foreground/50"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span
              className="inline-block size-2.5 shrink-0 rounded-full"
              style={{ background: accentHex }}
              aria-hidden="true"
            />
            <span className="truncate font-mono text-sm font-semibold tracking-wide">
              {ba}
            </span>
            {isRto ? (
              <span className="rounded-sm bg-foreground/8 px-1.5 py-0.5 font-mono text-[10px] font-medium text-foreground/70">
                RTO
              </span>
            ) : null}
          </div>
          <p className="mt-1 truncate text-xs text-muted-foreground">
            {BA_LABEL[ba]}
          </p>
        </div>
        <span
          className={`inline-flex items-center rounded-full px-2 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wider ring-1 ring-inset ${INTERCONNECT_PILL[interconnect]}`}
        >
          {interconnect}
        </span>
      </div>

      <Sparkline points={next24} accentHex={accentHex} />

      <div className="flex items-end justify-between border-t border-foreground/5 pt-3">
        <div>
          <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
            Peak · next 24h
          </p>
          <p className="mt-0.5 font-mono text-lg font-semibold tabular-nums">
            {(peak24 / 1000).toFixed(peak24 > 10000 ? 0 : 1)}
            <span className="ml-0.5 text-xs font-normal text-muted-foreground">GW</span>
          </p>
        </div>
        <div className="text-right">
          <div className="flex items-center gap-1.5">
            <span
              className={`size-1.5 rounded-full ${statusColor}`}
              aria-hidden="true"
            />
            <span className="font-mono text-xs tabular-nums">
              {pctOfPeak.toFixed(0)}
              <span className="text-muted-foreground">% of peak</span>
            </span>
          </div>
          <p className="mt-0.5 text-[10px] text-muted-foreground tabular-nums">
            all-time {allTimePeak >= 1000
              ? `${(allTimePeak / 1000).toFixed(0)} GW`
              : `${allTimePeak} MW`}
          </p>
        </div>
      </div>
    </Link>
  )
}
