// Plain-English explainer for non-industry visitors. Collapsed by default —
// most viewers who know the domain won't see it; the rest click once and
// get oriented without the UI turning into a textbook.

export function Glossary() {
  return (
    <details className="group rounded-xl border border-foreground/10 bg-card/50 open:bg-card">
      <summary className="flex cursor-pointer items-center justify-between rounded-xl px-4 py-3 text-sm font-medium marker:hidden transition-colors duration-150 hover:bg-foreground/[0.02] focus:outline-none focus-visible:ring-2 focus-visible:ring-foreground/50 [&::-webkit-details-marker]:hidden">
        <span>What does this all mean?</span>
        {/* Optical nudge: the ▾ glyph is top-weighted in most fonts, so
            a bare geometric center reads as slightly low. -translate-y-px
            pulls it one pixel up into the line's vertical midpoint. */}
        <span
          aria-hidden="true"
          className="-translate-y-px text-xs text-muted-foreground transition-transform duration-300 group-open:rotate-180 group-open:translate-y-0"
        >
          ▾
        </span>
      </summary>

      <div className="space-y-4 border-t border-foreground/5 px-4 py-4 text-sm">
        <Term name="Balancing authority (BA)">
          A company or agency that keeps a region&apos;s grid in balance —
          matching electricity generation to demand in real time. <b>PJM</b>{" "}
          runs the DC-to-Chicago corridor; <b>CAISO</b> runs California;{" "}
          <b>ERCOT</b> runs most of Texas. 53 BAs publish a live demand feed
          to the US Energy Information Administration (EIA-930) — surge
          forecasts all of them.
        </Term>

        <Term name="Interconnection">
          The three giant synchronised AC grids of North America —{" "}
          <b>Eastern</b>, <b>Western</b>, and <b>Texas</b>. They barely talk
          to each other (only a handful of DC ties). A blackout in one can&apos;t
          easily spread to the others.
        </Term>

        <Term name="Day-ahead forecast">
          How much electricity the model thinks will be consumed over the
          next 24 hours, made the day before. Grid operators use day-ahead
          forecasts to decide which power plants to start up, how much fuel
          to buy, and how to price wholesale electricity.
        </Term>

        <Term name="MW and GW (megawatts, gigawatts)">
          Units of instantaneous power demand. <b>1 GW = 1 000 MW ≈ 1 large
          nuclear reactor</b> at full output, or roughly enough to power
          750 000 typical US homes on a hot afternoon. PJM peaks at ~165 GW;
          a small utility like Homestead, FL peaks at ~100 MW.
        </Term>

        <Term name="Median · p10–p90 (the shaded band)">
          Surge returns a probability range, not a single number. The{" "}
          <b>median</b> is the middle guess. The <b>p10–p90 band</b> covers
          the middle 80% of likely outcomes — roughly: there&apos;s a 10%
          chance actual demand comes in below the band, 10% above it. A
          narrow band means the model is confident; a wide band means
          weather or weekday uncertainty matters.
        </Term>

        <Term name="% of all-time peak">
          How close tomorrow&apos;s forecasted maximum is to that BA&apos;s
          historical record. 100% = likely to tie or break the peak — that&apos;s
          when grid stress is real. &lt;60% is a comfortable shoulder day.
        </Term>

        <Term name="RTO / ISO">
          Seven US regions run competitive wholesale electricity markets
          (PJM, CAISO, ERCOT, MISO, NYISO, ISO-NE, SPP). Generation and
          retail are separate companies bidding through an auction. Non-RTO
          BAs are <em>vertically integrated</em> — one utility owns
          everything from the power plant to your meter.
        </Term>

        <Term name="MASE (model accuracy score)">
          How well the model does compared to a naive &quot;same as last
          week&quot; baseline. <b>Lower is better.</b> 1.0 = no better than
          naive; 0.5 = half the error of naive; 0.0 = perfect. surge-fm-v3
          scores 0.52 on the 7 biggest grids — about 10× better than a
          typical weather-adjusted regression.
        </Term>
      </div>
    </details>
  )
}

function Term({
  name,
  children,
}: {
  name: string
  children: React.ReactNode
}) {
  return (
    <div>
      <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
        {name}
      </p>
      <p className="mt-1 leading-relaxed text-foreground/90">{children}</p>
    </div>
  )
}
