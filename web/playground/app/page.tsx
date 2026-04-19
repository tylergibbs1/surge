// Playground page. A small client wrapper (<PlaygroundApp>) lifts
// (ba, horizon) so the US grid map and the chart stay synced — map click
// updates the chart; dropdown update highlights the map.

import { PlaygroundApp } from "@/components/playground-app"

export default function Page() {
  return (
    <div className="bg-background min-h-svh p-6 md:p-10">
      <div className="mx-auto max-w-5xl space-y-6">
        <header className="space-y-2">
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
        </header>

        <PlaygroundApp />

        <footer className="text-muted-foreground text-xs">
          <p>
            Research and reference use only. Not for trading, regulated bidding,
            or bankability-graded decisions.{" "}
            <a
              href="https://github.com/tylergibbs1/surge"
              className="underline"
              target="_blank"
              rel="noreferrer"
            >
              Code on GitHub
            </a>
            .
          </p>
        </footer>
      </div>
    </div>
  )
}
