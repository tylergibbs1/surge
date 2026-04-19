"use client"

import { Liveline } from "liveline"
import { useEffect, useState } from "react"

type ApiPayload = {
  as_of_utc: string
  latest_ts_utc: string
  latest_total_mw: number
  hours: number
  points: Array<{ ts_utc: string; total_mw: number; ba_count: number }>
}

type Point = { time: number; value: number }

type State =
  | { kind: "loading" }
  | { kind: "ready"; data: Point[]; value: number; asOf: string }
  | { kind: "error"; message: string }

// 60 s poll matches the 50 s edge cache on /api/live-load — at most one
// round trip per minute reaches Modal. EIA-930 publishes hourly, so
// there's no new signal in between; the smoothness Liveline adds is
// purely visual easing, not new data.
const POLL_MS = 60_000

async function fetchSnapshot(): Promise<Point[]> {
  const r = await fetch("/api/live-load?hours=24", { cache: "no-store" })
  if (!r.ok) throw new Error(`live-load ${r.status}`)
  const payload = (await r.json()) as ApiPayload
  return payload.points.map((p) => ({
    // Liveline times are epoch seconds (it divides Date.now() by 1000
    // internally and compares against point.time). Passing ms puts every
    // point 1000× in the future — outside the `window` and invisible.
    time: Date.parse(p.ts_utc) / 1000,
    // GW reads cleaner than MW in the value overlay: ~180 vs 180_000.
    value: p.total_mw / 1000,
  }))
}

// 24 h rolling window in seconds. Liveline defaults to 30 s, which is
// right for tick-by-tick market feeds but completely wrong for our
// hourly grid data — we'd show only the latest point (or nothing).
const WINDOW_SECS = 24 * 60 * 60

function formatValue(v: number): string {
  return `${v.toFixed(1)} GW`
}

export function UsDemandHero() {
  const [state, setState] = useState<State>({ kind: "loading" })

  useEffect(() => {
    let cancelled = false
    const tick = async () => {
      try {
        const data = await fetchSnapshot()
        if (cancelled || data.length === 0) return
        const latest = data[data.length - 1]
        setState({
          kind: "ready",
          data,
          value: latest.value,
          asOf: new Date(latest.time).toISOString(),
        })
      } catch (e) {
        if (cancelled) return
        setState({ kind: "error", message: String(e).slice(0, 120) })
      }
    }
    tick()
    const id = window.setInterval(tick, POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  return (
    <section
      aria-label="US grid demand — live"
      className="relative overflow-hidden rounded-xl bg-card/60 ring-1 ring-foreground/10"
    >
      <div className="flex items-center justify-between px-4 pt-3 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block size-1.5 rounded-full bg-emerald-500"
            style={{ boxShadow: "0 0 6px rgba(16, 185, 129, 0.8)" }}
            aria-hidden="true"
          />
          US demand · live
        </span>
        <span className="tabular-nums">
          sum of reporting balancing authorities
        </span>
      </div>

      {/* Liveline fills its parent height, so the explicit height on the
          wrapper is what controls the canvas size. Kept compact so the
          hero doesn't push the map below the fold on a 13-inch laptop. */}
      <div className="h-[180px] w-full px-1 pb-1">
        <Liveline
          data={state.kind === "ready" ? state.data : []}
          value={state.kind === "ready" ? state.value : 0}
          loading={state.kind === "loading"}
          emptyText={
            state.kind === "error" ? "live feed unavailable" : "no data yet"
          }
          theme="dark"
          color="#5CC8A2"
          window={WINDOW_SECS}
          fill
          grid
          badge={false}
          scrub={false}
          showValue
          valueMomentumColor
          momentum
          formatValue={formatValue}
          // Matches Liveline's default but makes the intent explicit: the
          // line breathes toward the new value at 8% per frame, tying
          // every discrete hourly update to the 60-fps render loop.
          lerpSpeed={0.08}
        />
      </div>
    </section>
  )
}
