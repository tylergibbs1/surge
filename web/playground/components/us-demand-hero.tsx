"use client"

import { Liveline } from "liveline"
import { useEffect, useRef, useState } from "react"

type ApiPayload = {
  as_of_utc: string
  latest_ts_utc: string
  latest_total_mw: number
  hours: number
  points: Array<{ ts_utc: string; total_mw: number; ba_count: number }>
}

type Point = { time: number; value: number }

// `ready` + a transient `staleError` flag: we keep the last good snapshot
// visible across intermittent 502s so a single Modal blip doesn't erase
// 24 h of chart. `loading` is only the initial pre-first-success state.
type Snapshot = { data: Point[]; value: number; latestTsMs: number }
type State =
  | { kind: "loading" }
  | { kind: "ready"; snap: Snapshot; staleError: boolean }
  | { kind: "error"; message: string } // first fetch failed — no data to show

// 60 s poll matches the 50 s edge cache on /api/live-load — at most one
// round trip per minute reaches Modal. EIA-930 publishes hourly, so
// there's no new signal in between; the smoothness Liveline adds is
// purely visual easing, not new data.
const POLL_MS = 60_000

async function fetchSnapshot(): Promise<Snapshot> {
  const r = await fetch("/api/live-load?hours=24", { cache: "no-store" })
  if (!r.ok) throw new Error(`live-load ${r.status}`)
  const payload = (await r.json()) as ApiPayload
  // Liveline times are epoch seconds (it divides Date.now() by 1000
  // internally and compares against point.time). Passing ms puts every
  // point 1000× in the future — outside the `window` and invisible.
  const data = payload.points.map((p) => ({
    time: Date.parse(p.ts_utc) / 1000,
    // GW reads cleaner than MW in the value overlay: ~180 vs 180_000.
    value: p.total_mw / 1000,
  }))
  if (data.length === 0) throw new Error("empty payload")
  const latest = data[data.length - 1]
  return {
    data,
    value: latest.value,
    latestTsMs: latest.time * 1000, // back to ms for display/aria
  }
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
  const intervalRef = useRef<number | null>(null)

  useEffect(() => {
    let cancelled = false

    const tick = async () => {
      try {
        const snap = await fetchSnapshot()
        if (cancelled) return
        setState({ kind: "ready", snap, staleError: false })
      } catch (e) {
        if (cancelled) return
        // Keep the last good data on transient failure; only surface the
        // error state if we've never successfully fetched. That way a
        // single 502 doesn't wipe 24 h of chart.
        setState((prev) =>
          prev.kind === "ready"
            ? { ...prev, staleError: true }
            : { kind: "error", message: String(e).slice(0, 120) },
        )
      }
    }

    const start = () => {
      tick()
      if (intervalRef.current != null) return
      intervalRef.current = window.setInterval(tick, POLL_MS)
    }
    const stop = () => {
      if (intervalRef.current != null) {
        window.clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }

    // Skip polling when the tab is hidden — no point spinning an RAF
    // inside Liveline and hitting /api/live-load while the user's
    // looking at another tab. Resume with an immediate fetch on return.
    const onVisibility = () => {
      if (document.visibilityState === "visible") start()
      else stop()
    }
    document.addEventListener("visibilitychange", onVisibility)

    if (document.visibilityState === "visible") start()
    return () => {
      cancelled = true
      stop()
      document.removeEventListener("visibilitychange", onVisibility)
    }
  }, [])

  return (
    <section
      aria-label="US grid demand — live"
      className="relative overflow-hidden rounded-xl bg-card/60 p-4 ring-1 ring-foreground/10"
    >
      <div className="mb-3 flex items-center justify-between text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span
            className={`inline-block size-1.5 rounded-full ${
              state.kind === "ready" && state.staleError
                ? "bg-amber-500"
                : "bg-emerald-500"
            }`}
            style={{
              boxShadow:
                state.kind === "ready" && state.staleError
                  ? "0 0 6px rgba(245, 158, 11, 0.8)"
                  : "0 0 6px rgba(16, 185, 129, 0.8)",
            }}
            aria-hidden="true"
          />
          US demand · {state.kind === "ready" && state.staleError ? "stale" : "live"}
        </span>
        <span className="tabular-nums">
          sum of reporting balancing authorities
        </span>
      </div>

      {/* Liveline fills its parent height, so the explicit height on the
          wrapper is what controls the canvas size. Kept compact so the
          hero doesn't push the map below the fold on a 13-inch laptop.
          No horizontal padding here — Liveline draws its own internal
          margins (see `padding` prop) and the outer p-4 on <section>
          already gives breathing room from the card edge. */}
      <div className="h-[180px] w-full">
        <Liveline
          data={state.kind === "ready" ? state.snap.data : []}
          value={state.kind === "ready" ? state.snap.value : 0}
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
          // Explicit padding override — Liveline's default right=12 (with
          // badge=false) puts the live dot and momentum arrows right up
          // against the canvas edge. 40 px gives them room to breathe
          // and keeps the grid labels ("200.0") from kissing the card
          // border. Left matches for visual symmetry.
          padding={{ top: 12, right: 40, bottom: 28, left: 16 }}
          // Matches Liveline's default but makes the intent explicit: the
          // line breathes toward the new value at 8% per frame, tying
          // every discrete hourly update to the 60-fps render loop.
          lerpSpeed={0.08}
        />
      </div>

      {/* Screen-reader announcement for every update. aria-live=polite so
          it doesn't interrupt, aria-atomic so the whole sentence re-reads
          (not just the diffed words). Visually hidden — the big number
          inside the Liveline canvas isn't text AT readers can see. */}
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {state.kind === "ready"
          ? `US grid demand is ${state.snap.value.toFixed(1)} gigawatts as of ${new Date(
              state.snap.latestTsMs,
            ).toLocaleString()}${state.staleError ? ", feed stale" : ""}`
          : state.kind === "error"
            ? "US demand feed unavailable"
            : "loading US demand"}
      </div>
    </section>
  )
}
