// Server component. Static content (header, footer, copy) renders on the
// server; only the interactive Playground ships to the client.

import { Playground } from "@/components/playground"

export default function Page() {
  return (
    <div className="bg-background min-h-svh p-6 md:p-10">
      <div className="mx-auto max-w-5xl space-y-6">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold tracking-tight">
            Surge playground
          </h1>
          <p className="text-muted-foreground max-w-3xl">
            Open probabilistic day-ahead load forecasts for 7 US balancing
            authorities. Pick a grid, pick a horizon, see the forecast. Model:
            Chronos-2 fine-tuned on 7 years of public data. Test MASE 0.45,
            ~2% MAPE — matching what utilities pay tens of thousands per year for.
          </p>
        </header>

        <Playground />

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
