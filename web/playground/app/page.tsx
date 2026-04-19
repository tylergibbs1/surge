// Playground page. A small client wrapper (<PlaygroundApp>) lifts
// (ba, horizon) so the US grid map and the chart stay synced — map click
// updates the chart; dropdown update highlights the map.

import Link from "next/link"
import { Suspense } from "react"

import { Glossary } from "@/components/glossary"
import { PlaygroundApp } from "@/components/playground-app"

export default function Page() {
  return (
    <div className="bg-background min-h-svh p-6 md:p-10">
      <div className="mx-auto max-w-5xl space-y-6">
        <header className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-2">
            <h1 className="text-3xl font-semibold tracking-tight">
              Surge playground
            </h1>
            <p className="text-muted-foreground max-w-3xl">
              Open probabilistic day-ahead load forecasts for all 53 US
              balancing authorities that publish a demand series to EIA-930.
              Click a region on the map or pick from the dropdown. Model:
              Chronos-2 fine-tuned on 7 years of public data — matching what
              utilities pay tens of thousands per year for.
            </p>
          </div>
          <nav className="flex overflow-hidden rounded-full bg-foreground/5 p-1 text-xs font-medium ring-1 ring-foreground/10">
            <Link
              href="/"
              className="rounded-full bg-background px-3 py-1.5 shadow-sm"
              aria-current="page"
            >
              Map
            </Link>
            <Link
              href="/grid"
              className="rounded-full px-3 py-1.5 text-muted-foreground transition hover:text-foreground"
            >
              Grid
            </Link>
          </nav>
        </header>

        {/* useSearchParams() inside PlaygroundApp forces client-side URL
            reading; wrap in Suspense so the page can still statically
            prerender its shell. */}
        <Suspense fallback={null}>
          <PlaygroundApp />
        </Suspense>

        <Glossary />

        <footer className="text-muted-foreground space-y-2 text-xs">
          <p className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <a
              href="https://github.com/tylergibbs1/surge"
              className="underline-offset-4 hover:underline"
              target="_blank"
              rel="noreferrer"
            >
              github.com/tylergibbs1/surge
            </a>
            <span aria-hidden="true" className="text-foreground/20">·</span>
            <a
              href="https://huggingface.co/Tylerbry1/surge-fm-v3"
              className="underline-offset-4 hover:underline"
              target="_blank"
              rel="noreferrer"
            >
              huggingface.co/Tylerbry1/surge-fm-v3
            </a>
          </p>
          <p>
            Research and reference use only. Not for trading, regulated bidding,
            or bankability-graded decisions.
          </p>
        </footer>
      </div>
    </div>
  )
}
