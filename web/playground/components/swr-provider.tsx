"use client"

import { SWRConfig } from "swr"

// Shared SWR fetcher. Attaches HTTP `status` + parsed error body to the
// thrown Error so consumers can branch on status without re-parsing.
// Kept alongside the provider so every useSWR in the tree sees the same
// error shape — callers need only `err.message` for display, with
// `.status` / `.info` available when they want to do more.
//
// Exported so `preload(key, swrFetcher)` calls (which bypass the
// provider context) use the same request shape and land in the same
// cache as the useSWR hooks that read the result.
export const swrFetcher = async (url: string): Promise<unknown> => {
  const r = await fetch(url)
  if (!r.ok) {
    const body = (await r.json().catch(() => ({}))) as {
      detail?: string
      error?: string
    }
    const err = new Error(
      body.detail ?? body.error ?? `HTTP ${r.status}`,
    ) as Error & { status?: number; info?: unknown }
    err.status = r.status
    err.info = body
    throw err
  }
  return r.json()
}

// One place to tune freshness policy for every client fetch in the app.
//
// - `dedupingInterval: 300_000` matches the 5-minute edge-cache on our
//   forecast/actuals endpoints, so identical keys collapse into a single
//   network request within that window.
// - `revalidateOnFocus: false` — nothing here changes on tab refocus;
//   the daily bake runs at 06:15 UTC, the intraday EIA drop once an hour.
// - `keepPreviousData` keeps the chart painted while the user switches
//   BAs instead of flashing a skeleton.
//
// Hooks that want different behavior (e.g., live polling) can still
// pass their own options — per-hook options override provider defaults.
export function SwrProvider({ children }: { children: React.ReactNode }) {
  return (
    <SWRConfig
      value={{
        fetcher: swrFetcher,
        revalidateOnFocus: false,
        dedupingInterval: 300_000,
        keepPreviousData: true,
      }}
    >
      {children}
    </SWRConfig>
  )
}
