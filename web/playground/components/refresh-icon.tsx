// Cross-fades between a static refresh glyph and a spinning loader, using
// CSS transitions (no motion library dep). Both icons stay in the DOM;
// one is absolutely positioned over the other. Scale+opacity+blur per the
// contextual icon animation recipe.

import { Loader2, RefreshCw } from "lucide-react"

const CROSS = "opacity,scale,filter"
const EASE = "cubic-bezier(0.2, 0, 0, 1)"
const DUR = "300ms"

export function RefreshIcon({ loading }: { loading: boolean }) {
  return (
    <span className="relative inline-block size-4" aria-hidden="true">
      <RefreshCw
        className="absolute inset-0 size-4"
        style={{
          transitionProperty: CROSS,
          transitionDuration: DUR,
          transitionTimingFunction: EASE,
          opacity: loading ? 0 : 1,
          transform: loading ? "scale(0.25)" : "scale(1)",
          filter: loading ? "blur(4px)" : "blur(0px)",
        }}
      />
      <Loader2
        className="absolute inset-0 size-4 animate-spin"
        style={{
          transitionProperty: CROSS,
          transitionDuration: DUR,
          transitionTimingFunction: EASE,
          opacity: loading ? 1 : 0,
          transform: loading ? "scale(1)" : "scale(0.25)",
          filter: loading ? "blur(0px)" : "blur(4px)",
        }}
      />
    </span>
  )
}
