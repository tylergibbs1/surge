import Link from "next/link"

import { DataStateBadge } from "@/components/data-state-badge"
import { ForecastSparkline } from "@/components/forecast-sparkline"
import {
  formatAge,
  formatLocalDateTime,
  formatPower,
  formatUtcDateTime,
} from "@/lib/forecast-display"
import type { ScoreboardRegion } from "@/lib/v2-contracts"

function PeakCell({ region }: { region: ScoreboardRegion }) {
  const peak = region.next24hPeak
  if (!peak) {
    return (
      <span className="block max-w-44 text-xs leading-5 text-muted-foreground">
        No complete current 24-hour window
      </span>
    )
  }

  return (
    <div>
      <span className="block text-base font-semibold tabular-nums">
        {formatPower(peak.medianMw)}
      </span>
      <time
        dateTime={peak.validAtUtc}
        className="mt-0.5 block text-xs whitespace-nowrap text-muted-foreground"
      >
        {formatLocalDateTime(peak.validAtUtc, region.timezone)}
      </time>
    </div>
  )
}

export function ScoreboardTable({ regions }: { regions: ScoreboardRegion[] }) {
  return (
    <div className="overflow-x-auto border border-border bg-card">
      <table className="w-full min-w-[980px] border-collapse text-left">
        <caption className="sr-only">
          Current next-24-hour probabilistic load forecast summary for seven US
          regional transmission organizations. Every row includes an explicit
          freshness state and forecast timestamp.
        </caption>
        <thead className="border-b border-border bg-muted/60 font-mono text-[11px] tracking-[0.1em] text-muted-foreground uppercase">
          <tr>
            <th scope="col" className="px-4 py-3 font-medium sm:px-5">
              Region
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Data state
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              24h median shape
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Forecast peak
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              80% range at peak
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Issued
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Detail
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {regions.map((region) => {
            const peak = region.next24hPeak
            return (
              <tr
                key={region.code}
                className="align-middle transition-colors hover:bg-muted/35"
              >
                <th scope="row" className="px-4 py-4 font-normal sm:px-5">
                  <span className="block text-base font-semibold tracking-[-0.02em]">
                    {region.shortName}
                  </span>
                  <span className="mt-1 block max-w-52 text-xs leading-4 text-muted-foreground">
                    {region.name} · {region.interconnection}
                  </span>
                </th>
                <td className="px-4 py-4">
                  <DataStateBadge state={region.state} />
                </td>
                <td className="px-4 py-3">
                  <ForecastSparkline
                    label={region.shortName}
                    points={region.next24hTrend}
                  />
                </td>
                <td className="px-4 py-4">
                  <PeakCell region={region} />
                </td>
                <td className="px-4 py-4">
                  {peak ? (
                    <span className="font-mono text-xs whitespace-nowrap tabular-nums">
                      {formatPower(peak.p10Mw)}–{formatPower(peak.p90Mw)}
                    </span>
                  ) : (
                    <span className="text-sm text-muted-foreground">—</span>
                  )}
                </td>
                <td className="px-4 py-4">
                  {region.issueAtUtc ? (
                    <div className="text-xs whitespace-nowrap">
                      <span className="block font-medium">
                        {formatAge(region.issueAgeHours)}
                      </span>
                      <time
                        dateTime={region.issueAtUtc}
                        className="mt-1 block text-muted-foreground"
                      >
                        {formatUtcDateTime(region.issueAtUtc)}
                      </time>
                    </div>
                  ) : (
                    <span className="text-xs text-muted-foreground">
                      No issuance
                    </span>
                  )}
                </td>
                <td className="px-4 py-4">
                  <Link
                    href={region.detailHref}
                    className="inline-flex min-h-10 items-center text-sm font-semibold whitespace-nowrap text-[#315e9f] underline-offset-4 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-ring dark:text-[#8cb5f7]"
                    aria-label={`Open ${region.shortName} forecast explorer`}
                  >
                    Open →
                  </Link>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
