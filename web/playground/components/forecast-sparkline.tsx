import { formatPower } from "@/lib/forecast-display"
import type { ScoreboardTrendPoint } from "@/lib/v2-contracts"

const WIDTH = 132
const HEIGHT = 42
const PAD = 2

function coordinate(
  value: number,
  index: number,
  count: number,
  minimum: number,
  span: number
): [number, number] {
  const x = PAD + (index / Math.max(1, count - 1)) * (WIDTH - PAD * 2)
  const y = HEIGHT - PAD - ((value - minimum) / span) * (HEIGHT - PAD * 2)
  return [x, y]
}

function pointsAttribute(points: Array<[number, number]>): string {
  return points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ")
}

export function ForecastSparkline({
  label,
  points,
}: {
  label: string
  points: ScoreboardTrendPoint[]
}) {
  if (points.length < 2) {
    return (
      <span className="text-xs text-muted-foreground">No current shape</span>
    )
  }

  const minimum = Math.min(...points.map((point) => point.p10Mw))
  const maximum = Math.max(...points.map((point) => point.p90Mw))
  const span = Math.max(1, maximum - minimum)
  const upper = points.map((point, index) =>
    coordinate(point.p90Mw, index, points.length, minimum, span)
  )
  const lower = points.map((point, index) =>
    coordinate(point.p10Mw, index, points.length, minimum, span)
  )
  const median = points.map((point, index) =>
    coordinate(point.medianMw, index, points.length, minimum, span)
  )
  const band = `${pointsAttribute(upper)} ${pointsAttribute([...lower].reverse())}`
  const first = points.at(0)!
  const last = points.at(-1)!
  const direction =
    last.medianMw > first.medianMw
      ? "rises"
      : last.medianMw < first.medianMw
        ? "falls"
        : "is flat"

  return (
    <span className="block w-[132px] text-[#315e9f] dark:text-[#8cb5f7]">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        width={WIDTH}
        height={HEIGHT}
        aria-hidden="true"
        focusable="false"
      >
        <polygon points={band} fill="currentColor" opacity="0.14" />
        <polyline
          points={pointsAttribute(median)}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.75"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <span className="sr-only">
        {label} median forecast {direction} from {formatPower(first.medianMw)}{" "}
        to {formatPower(last.medianMw)} across the current 24-hour window.
      </span>
    </span>
  )
}
