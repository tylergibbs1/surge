import type { Metadata } from "next"

import { DataFreshnessBanner } from "@/components/data-freshness-banner"
import { DataStateBadge } from "@/components/data-state-badge"
import { SiteFooter } from "@/components/site-footer"
import { SiteHeader } from "@/components/site-header"
import {
  FRESHNESS_DESCRIPTION,
  FRESHNESS_LABEL,
  formatAge,
  formatUtcDateTime,
} from "@/lib/forecast-display"
import { loadScoreboardSnapshot } from "@/lib/server/load-scoreboard"
import type { DataFreshnessState } from "@/lib/v2-contracts"

export const metadata: Metadata = {
  title: "Data status",
  description:
    "Current freshness and coverage status for the seven regional forecasts on Surge.",
}

export const runtime = "nodejs"
export const revalidate = 300

const STATES: DataFreshnessState[] = [
  "fresh",
  "delayed",
  "stale",
  "unavailable",
]

export default async function StatusPage() {
  const snapshot = await loadScoreboardSnapshot()

  return (
    <div className="min-h-svh bg-background">
      <SiteHeader active="status" />

      <section className="border-b border-white/10 bg-[#07111f] text-white">
        <div className="mx-auto max-w-7xl px-5 py-12 sm:px-8 lg:px-10 lg:py-16">
          <p className="font-mono text-xs tracking-[0.18em] text-[#8cb5f7] uppercase">
            Data status
          </p>
          <h1 className="mt-5 max-w-4xl text-5xl leading-[0.98] font-semibold tracking-[-0.05em] sm:text-6xl">
            Is the forecast actually current?
          </h1>
          <p className="mt-6 max-w-2xl text-base leading-7 text-slate-300">
            Region-by-region timestamps and coverage from the exact snapshot
            powering the public scoreboard.
          </p>
        </div>
      </section>

      <main
        id="main"
        className="mx-auto max-w-7xl space-y-10 px-5 py-8 sm:px-8 lg:px-10 lg:py-12"
      >
        <DataFreshnessBanner snapshot={snapshot} />

        <section
          aria-label="Status summary"
          className="grid border border-border sm:grid-cols-2 lg:grid-cols-4"
        >
          <div className="border-b border-border p-5 sm:border-r lg:border-b-0">
            <span className="block font-mono text-xs tracking-[0.12em] text-muted-foreground uppercase">
              Overall
            </span>
            <span className="mt-3 block text-2xl font-semibold">
              {FRESHNESS_LABEL[snapshot.overallState]}
            </span>
          </div>
          <div className="border-b border-border p-5 lg:border-r lg:border-b-0">
            <span className="block font-mono text-xs tracking-[0.12em] text-muted-foreground uppercase">
              Current rows
            </span>
            <span className="mt-3 block text-2xl font-semibold tabular-nums">
              {snapshot.currentRegions} / {snapshot.expectedRegions}
            </span>
          </div>
          <div className="border-b border-border p-5 sm:border-r sm:border-b-0">
            <span className="block font-mono text-xs tracking-[0.12em] text-muted-foreground uppercase">
              Valid rows
            </span>
            <span className="mt-3 block text-2xl font-semibold tabular-nums">
              {snapshot.availableRegions} / {snapshot.expectedRegions}
            </span>
          </div>
          <div className="p-5">
            <span className="block font-mono text-xs tracking-[0.12em] text-muted-foreground uppercase">
              Artifact
            </span>
            <span className="mt-3 block text-sm font-semibold">
              {snapshot.source.artifactPath}
            </span>
            <span className="mt-1 block text-xs text-muted-foreground">
              {formatAge(snapshot.source.artifactAgeHours)}
            </span>
          </div>
        </section>

        <section aria-labelledby="regional-status-heading">
          <div className="mb-5">
            <p className="font-mono text-xs tracking-[0.16em] text-[#315e9f] uppercase dark:text-[#8cb5f7]">
              Coverage ledger
            </p>
            <h2
              id="regional-status-heading"
              className="mt-2 text-3xl font-semibold tracking-[-0.04em]"
            >
              Regional timestamps
            </h2>
          </div>
          <div className="overflow-x-auto border border-border">
            <table className="w-full min-w-[860px] border-collapse text-left text-sm">
              <caption className="sr-only">
                Forecast freshness, issuance, and coverage for each scoreboard
                region
              </caption>
              <thead className="border-b border-border bg-muted/60 font-mono text-[11px] tracking-[0.1em] text-muted-foreground uppercase">
                <tr>
                  <th scope="col" className="px-4 py-3 font-medium">
                    Region
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium">
                    State
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium">
                    Issued
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium">
                    Coverage ends
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium">
                    Note
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {snapshot.regions.map((region) => (
                  <tr key={region.code} className="align-top">
                    <th scope="row" className="px-4 py-4 font-semibold">
                      {region.shortName}
                    </th>
                    <td className="px-4 py-4">
                      <DataStateBadge state={region.state} />
                    </td>
                    <td className="px-4 py-4">
                      {region.issueAtUtc ? (
                        <time
                          dateTime={region.issueAtUtc}
                          className="whitespace-nowrap"
                        >
                          {formatUtcDateTime(region.issueAtUtc)}
                        </time>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td className="px-4 py-4">
                      {region.coverageEndsAtUtc ? (
                        <time
                          dateTime={region.coverageEndsAtUtc}
                          className="whitespace-nowrap"
                        >
                          {formatUtcDateTime(region.coverageEndsAtUtc)}
                        </time>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td className="max-w-sm px-4 py-4 text-muted-foreground">
                      {region.warnings.at(0) ?? "No freshness warning."}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section
          aria-labelledby="labels-heading"
          className="border-t border-border pt-9"
        >
          <h2
            id="labels-heading"
            className="text-2xl font-semibold tracking-[-0.03em]"
          >
            What the labels mean
          </h2>
          <dl className="mt-5 grid gap-px bg-border sm:grid-cols-2 lg:grid-cols-4">
            {STATES.map((state) => (
              <div key={state} className="bg-background p-4">
                <dt>
                  <DataStateBadge state={state} />
                </dt>
                <dd className="mt-3 text-sm leading-6 text-muted-foreground">
                  {FRESHNESS_DESCRIPTION[state]}
                </dd>
              </div>
            ))}
          </dl>
          <p className="mt-5 max-w-3xl text-sm leading-6 text-muted-foreground">
            This reports publication and coverage health only. It is not a
            model-worker uptime monitor and it does not establish forecast
            accuracy. The page is regenerated at most every five minutes.
          </p>
        </section>
      </main>

      <SiteFooter />
    </div>
  )
}
