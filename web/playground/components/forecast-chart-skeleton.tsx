// Small deliberately-simple skeleton. Lives in its own file so it's in the
// server bundle and can render instantly while the heavy `recharts` chunk
// loads on first forecast arrival.

export function ForecastChartSkeleton() {
  return (
    <div
      className="bg-muted/30 h-[360px] w-full animate-pulse rounded-md"
      aria-hidden="true"
    />
  )
}
