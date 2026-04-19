"use client"

// Analytics + SpeedInsights aren't needed for first paint or
// interactivity — load them after hydration so they don't sit in the
// critical bundle path. Lives in its own client boundary because
// `dynamic(..., { ssr: false })` isn't allowed in Server Components.

import dynamic from "next/dynamic"

const Analytics = dynamic(
  () => import("@vercel/analytics/next").then((m) => m.Analytics),
  { ssr: false },
)

const SpeedInsights = dynamic(
  () => import("@vercel/speed-insights/next").then((m) => m.SpeedInsights),
  { ssr: false },
)

export function ThirdPartyAnalytics() {
  return (
    <>
      <Analytics />
      <SpeedInsights />
    </>
  )
}
