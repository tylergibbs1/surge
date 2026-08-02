import type { Metadata } from "next"
import Link from "next/link"

import { SiteFooter } from "@/components/site-footer"
import { SiteHeader } from "@/components/site-header"

export const metadata: Metadata = {
  title: "Methodology",
  description:
    "How Surge derives its seven-RTO forecast scoreboard, labels data freshness, and handles evidence the current snapshot does not contain.",
}

const FRESHNESS_ROWS = [
  {
    state: "Current",
    rule: "Issue and artifact are at most 26 hours old, with 24 consecutive future hourly points.",
  },
  {
    state: "Delayed",
    rule: "The issue is 26–36 hours old, the artifact is aging, or current coverage is incomplete but still useful.",
  },
  {
    state: "Stale",
    rule: "The issue or artifact is more than 36 hours old, or fewer than 18 hours remain in the current window.",
  },
  {
    state: "Unavailable",
    rule: "The snapshot, region, timestamp, points, or quantile ordering cannot be validated.",
  },
] as const

export default function MethodologyPage() {
  return (
    <div className="min-h-svh bg-background">
      <SiteHeader active="methodology" />

      <section className="border-b border-white/10 bg-[#07111f] text-white">
        <div className="mx-auto max-w-5xl px-5 py-12 sm:px-8 lg:px-10 lg:py-16">
          <p className="font-mono text-xs tracking-[0.18em] text-[#8cb5f7] uppercase">
            Methodology · v0.2 interface
          </p>
          <h1 className="mt-5 max-w-4xl text-5xl leading-[0.98] font-semibold tracking-[-0.05em] sm:text-6xl">
            Every displayed number has a derivation. Every missing proof stays
            missing.
          </h1>
          <p className="mt-6 max-w-2xl text-base leading-7 text-slate-300">
            This page documents the public scoreboard adapter. The v0.2 service
            has an auditable ledger, but the currently hosted snapshot predates
            it and has no complete, settled forward-scoring history yet.
          </p>
        </div>
      </section>

      <main
        id="main"
        className="mx-auto max-w-5xl px-5 py-10 sm:px-8 lg:px-10 lg:py-14"
      >
        <div className="grid gap-12 lg:grid-cols-[220px_minmax(0,1fr)]">
          <nav
            aria-label="On this page"
            className="lg:sticky lg:top-6 lg:self-start"
          >
            <p className="font-mono text-xs tracking-[0.14em] text-muted-foreground uppercase">
              On this page
            </p>
            <ol className="mt-3 space-y-2 text-sm">
              <li>
                <a
                  href="#source"
                  className="underline-offset-4 hover:underline"
                >
                  1. Source snapshot
                </a>
              </li>
              <li>
                <a
                  href="#derivation"
                  className="underline-offset-4 hover:underline"
                >
                  2. Scoreboard derivation
                </a>
              </li>
              <li>
                <a
                  href="#freshness"
                  className="underline-offset-4 hover:underline"
                >
                  3. Freshness rules
                </a>
              </li>
              <li>
                <a
                  href="#limits"
                  className="underline-offset-4 hover:underline"
                >
                  4. Current limits
                </a>
              </li>
            </ol>
          </nav>

          <article className="space-y-14 text-[15px] leading-7">
            <section
              id="source"
              aria-labelledby="source-heading"
              className="scroll-mt-6"
            >
              <p className="font-mono text-xs tracking-[0.14em] text-[#315e9f] uppercase dark:text-[#8cb5f7]">
                01
              </p>
              <h2
                id="source-heading"
                className="mt-2 text-3xl font-semibold tracking-[-0.035em]"
              >
                Source snapshot
              </h2>
              <div className="mt-5 space-y-4 text-muted-foreground">
                <p>
                  The v0.2 interface resolves the same private Vercel Blob
                  current pointer as the existing{" "}
                  <code className="font-mono text-xs text-foreground">
                    /api/forecast-all
                  </code>{" "}
                  route:
                  <code className="ml-1 font-mono text-xs text-foreground">
                    forecasts/v2/current.json
                  </code>
                  . The pointer must reference a matching immutable all-region
                  run. It does not call the site&apos;s own HTTP API from the server.
                </p>
                <p>
                  The adapter validates the payload as unknown input, selects
                  only PJM, CAISO, ERCOT, MISO, NYISO, ISO-NE, and SPP, and
                  always returns exactly seven rows. Missing or malformed data
                  becomes an explicit unavailable row; it never becomes zero.
                </p>
              </div>
            </section>

            <section
              id="derivation"
              aria-labelledby="derivation-heading"
              className="scroll-mt-6"
            >
              <p className="font-mono text-xs tracking-[0.14em] text-[#315e9f] uppercase dark:text-[#8cb5f7]">
                02
              </p>
              <h2
                id="derivation-heading"
                className="mt-2 text-3xl font-semibold tracking-[-0.035em]"
              >
                Scoreboard derivation
              </h2>
              <ol className="mt-5 space-y-5 text-muted-foreground">
                <li className="grid grid-cols-[2rem_1fr] gap-3">
                  <span className="font-mono text-sm text-foreground">01</span>
                  <span>
                    Keep valid, finite forecast points with ordered quantiles:
                    p10 ≤ median ≤ p90.
                  </span>
                </li>
                <li className="grid grid-cols-[2rem_1fr] gap-3">
                  <span className="font-mono text-sm text-foreground">02</span>
                  <span>
                    Select the 24 consecutive hourly timestamps strictly after
                    page-generation time and within the next 24 hours.
                  </span>
                </li>
                <li className="grid grid-cols-[2rem_1fr] gap-3">
                  <span className="font-mono text-sm text-foreground">03</span>
                  <span>
                    Choose the largest median in that window as the forecast
                    peak. Show p10 and p90 from that same hour as the nominal
                    80% range.
                  </span>
                </li>
                <li className="grid grid-cols-[2rem_1fr] gap-3">
                  <span className="font-mono text-sm text-foreground">04</span>
                  <span>
                    Format peak time in the RTO&apos;s configured IANA time zone
                    and issuance time in UTC, so daylight-saving changes are
                    handled by the runtime.
                  </span>
                </li>
              </ol>
              <p className="mt-6 border-l-2 border-[#315e9f] pl-4 text-sm text-muted-foreground dark:border-[#8cb5f7]">
                “80% range” names the interval between model quantiles; it is
                not a claim that prospective coverage has already been proven to
                equal 80%.
              </p>
            </section>

            <section
              id="freshness"
              aria-labelledby="freshness-rules-heading"
              className="scroll-mt-6"
            >
              <p className="font-mono text-xs tracking-[0.14em] text-[#315e9f] uppercase dark:text-[#8cb5f7]">
                03
              </p>
              <h2
                id="freshness-rules-heading"
                className="mt-2 text-3xl font-semibold tracking-[-0.035em]"
              >
                Freshness rules
              </h2>
              <div className="mt-5 overflow-x-auto border border-border">
                <table className="w-full min-w-[620px] border-collapse text-left text-sm">
                  <caption className="sr-only">
                    Rules used to label forecast freshness
                  </caption>
                  <thead className="border-b border-border bg-muted/60 font-mono text-xs tracking-[0.08em] text-muted-foreground uppercase">
                    <tr>
                      <th scope="col" className="px-4 py-3 font-medium">
                        Label
                      </th>
                      <th scope="col" className="px-4 py-3 font-medium">
                        Rule
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {FRESHNESS_ROWS.map((row) => (
                      <tr key={row.state}>
                        <th scope="row" className="px-4 py-3 font-semibold">
                          {row.state}
                        </th>
                        <td className="px-4 py-3 text-muted-foreground">
                          {row.rule}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-4 text-sm text-muted-foreground">
                The overall banner is stale when any available regional row is
                stale. It is delayed for mixed current/unavailable results, and
                unavailable only when no valid regional forecast exists.
              </p>
            </section>

            <section
              id="limits"
              aria-labelledby="limits-heading"
              className="scroll-mt-6 border border-orange-300 bg-orange-50 p-5 dark:border-orange-800 dark:bg-orange-950/30"
            >
              <p className="font-mono text-xs tracking-[0.14em] text-orange-800 uppercase dark:text-orange-300">
                04 · important
              </p>
              <h2
                id="limits-heading"
                className="mt-2 text-3xl font-semibold tracking-[-0.035em]"
              >
                Current limits
              </h2>
              <ul className="mt-5 list-disc space-y-3 pl-5 text-muted-foreground marker:text-orange-700 dark:marker:text-orange-300">
                <li>
                  The v1 daily artifact is a replaceable snapshot, not an
                  immutable issuance ledger.
                </li>
                <li>
                  It does not pin outcomes or revisions to forecast points, so
                  this page cannot publish a prospective track record.
                </li>
                <li>
                  It does not include operator forecasts, so this page cannot
                  publish an operator comparison.
                </li>
                <li>
                  Research evaluation results are not presented as proof of live
                  production accuracy.
                </li>
              </ul>
              <p className="mt-5 text-sm">
                <a
                  href="https://github.com/tylergibbs1/surge/issues/1"
                  target="_blank"
                  rel="noreferrer"
                  className="font-semibold underline underline-offset-4"
                >
                  Follow the open evaluation-integrity issue
                  <span className="sr-only"> (opens in a new tab)</span>
                </a>
                .
              </p>
            </section>

            <p>
              <Link
                href="/"
                className="font-semibold text-[#315e9f] underline-offset-4 hover:underline dark:text-[#8cb5f7]"
              >
                ← Back to the scoreboard
              </Link>
            </p>
          </article>
        </div>
      </main>

      <SiteFooter />
    </div>
  )
}
