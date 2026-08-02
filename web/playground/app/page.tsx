import Link from "next/link"

import { DataFreshnessBanner } from "@/components/data-freshness-banner"
import { LegacyExplorerPage } from "@/components/legacy-explorer-page"
import { ScoreboardTable } from "@/components/scoreboard-table"
import { SiteFooter } from "@/components/site-footer"
import { SiteHeader } from "@/components/site-header"
import { loadScoreboardSnapshot } from "@/lib/server/load-scoreboard"

export const runtime = "nodejs"
export const revalidate = 300

type SearchParams = Promise<Record<string, string | string[] | undefined>>

export default async function Page({
  searchParams,
}: {
  searchParams: SearchParams
}) {
  const params = await searchParams

  // Existing /?ba=...&horizon=... links remain the stable entry point for
  // the detailed map. Bare `/` is the new v0.2 scoreboard.
  if (params.ba !== undefined || params.horizon !== undefined) {
    return <LegacyExplorerPage />
  }

  const snapshot = await loadScoreboardSnapshot()

  return (
    <div className="min-h-svh bg-background">
      <SiteHeader active="scoreboard" />

      <section className="border-b border-white/10 bg-[#07111f] text-white">
        <div className="mx-auto grid max-w-7xl gap-10 px-5 pt-9 pb-12 sm:px-8 lg:grid-cols-[minmax(0,1fr)_300px] lg:px-10 lg:pt-12 lg:pb-16">
          <div className="max-w-4xl">
            <p className="font-mono text-xs tracking-[0.18em] text-[#8cb5f7] uppercase">
              v0.2 · public preview
            </p>
            <h1 className="mt-5 max-w-4xl text-5xl leading-[0.96] font-semibold tracking-[-0.055em] sm:text-6xl lg:text-7xl">
              The grid&apos;s next peak, with the timestamp attached.
            </h1>
            <p className="mt-6 max-w-2xl text-base leading-7 text-slate-300 sm:text-lg">
              A server-rendered outlook across seven major US power markets.
              Surge shows what the current model snapshot can support—and says
              plainly when the data is late, stale, or missing.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <a
                href="#scoreboard"
                className="inline-flex min-h-11 items-center border border-[#73a6ff] bg-[#73a6ff] px-4 text-sm font-semibold text-[#07111f] transition-colors hover:bg-[#a9c8ff] focus:outline-none focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-[#07111f]"
              >
                View the scoreboard
              </a>
              <Link
                href="/methodology"
                className="inline-flex min-h-11 items-center border border-slate-600 px-4 text-sm font-semibold text-white transition-colors hover:border-slate-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-[#07111f]"
              >
                Read the methodology
              </Link>
            </div>
          </div>

          <aside
            aria-label="Scoreboard scope"
            className="grid grid-cols-3 border-y border-white/15 lg:grid-cols-1 lg:border-x-0 lg:border-y-0 lg:border-l lg:pl-8"
          >
            <div className="py-5 lg:border-b lg:border-white/15">
              <span className="block font-mono text-3xl tabular-nums">07</span>
              <span className="mt-1 block text-xs tracking-[0.12em] text-slate-400 uppercase">
                Major RTOs
              </span>
            </div>
            <div className="border-x border-white/15 px-4 py-5 lg:border-x-0 lg:border-b lg:px-0">
              <span className="block font-mono text-3xl tabular-nums">24h</span>
              <span className="mt-1 block text-xs tracking-[0.12em] text-slate-400 uppercase">
                Current window
              </span>
            </div>
            <div className="py-5 pl-4 lg:pl-0">
              <span className="block font-mono text-3xl tabular-nums">
                {String(snapshot.currentRegions).padStart(2, "0")}
              </span>
              <span className="mt-1 block text-xs tracking-[0.12em] text-slate-400 uppercase">
                Current rows
              </span>
            </div>
          </aside>
        </div>
      </section>

      <main
        id="main"
        className="mx-auto max-w-7xl space-y-12 px-5 py-8 sm:px-8 lg:px-10 lg:py-12"
      >
        <DataFreshnessBanner snapshot={snapshot} />

        <section
          id="scoreboard"
          aria-labelledby="scoreboard-heading"
          className="scroll-mt-6"
        >
          <div className="mb-5 grid gap-3 md:grid-cols-[1fr_minmax(280px,440px)] md:items-end">
            <div>
              <p className="font-mono text-xs tracking-[0.16em] text-[#315e9f] uppercase dark:text-[#8cb5f7]">
                Regional outlook
              </p>
              <h2
                id="scoreboard-heading"
                className="mt-2 text-3xl font-semibold tracking-[-0.04em] sm:text-4xl"
              >
                Seven regions. No silent gaps.
              </h2>
            </div>
            <p className="text-sm leading-6 text-muted-foreground md:text-right">
              Peak is the highest median forecast in a complete next-24-hour
              window. The range is the model&apos;s p10–p90 interval at that
              hour.
            </p>
          </div>
          <ScoreboardTable regions={snapshot.regions} />
          <p className="mt-3 text-xs leading-5 text-muted-foreground">
            Peak times use each region&apos;s configured IANA time zone;
            issuance timestamps use UTC. A stale value is never relabeled as a
            current forecast.
          </p>
        </section>

        <section
          aria-labelledby="evidence-heading"
          className="grid border-y border-border py-9 md:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)] md:gap-12"
        >
          <div>
            <p className="font-mono text-xs tracking-[0.16em] text-[#315e9f] uppercase dark:text-[#8cb5f7]">
              Evidence before confidence
            </p>
            <h2
              id="evidence-heading"
              className="mt-2 text-3xl font-semibold tracking-[-0.04em]"
            >
              What is deliberately absent
            </h2>
          </div>
          <div className="mt-6 space-y-4 text-sm leading-6 text-muted-foreground md:mt-0">
            <p>
              The v0.2 service now writes immutable issuance IDs and can pin
              outcomes 72 hours after each forecast window. The currently
              hosted snapshot predates that ledger and contains no complete,
              settled forward history, so Surge does not invent an operator
              delta, accuracy badge, or confidence ranking for this page.
            </p>
            <p>
              Until the restored publisher produces complete seven-region runs
              and they mature, this interface treats the daily artifact as a
              snapshot and exposes its limits. The ledger contracts and
              settlement policy are documented and ready to populate.
            </p>
            <Link
              href="/methodology"
              className="inline-flex min-h-10 items-center font-semibold text-[#315e9f] underline-offset-4 hover:underline dark:text-[#8cb5f7]"
            >
              See the derivation and freshness rules →
            </Link>
          </div>
        </section>
      </main>

      <SiteFooter />
    </div>
  )
}
