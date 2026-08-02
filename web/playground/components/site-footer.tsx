import Link from "next/link"

export function SiteFooter() {
  return (
    <footer className="border-t border-border bg-muted/30">
      <div className="mx-auto grid max-w-7xl gap-6 px-5 py-8 text-sm text-muted-foreground sm:px-8 md:grid-cols-[1fr_auto] lg:px-10">
        <div className="max-w-2xl space-y-2">
          <p className="font-medium text-foreground">
            Surge is an open research prototype.
          </p>
          <p>
            Forecasts may be delayed, incomplete, or wrong. Do not use them for
            trading, regulated bidding, reliability operations, or bankability
            decisions.
          </p>
        </div>
        <div className="flex flex-wrap items-start gap-x-5 gap-y-2 md:justify-end">
          <Link
            href="/methodology"
            className="underline-offset-4 hover:underline"
          >
            Methodology
          </Link>
          <Link href="/status" className="underline-offset-4 hover:underline">
            Data status
          </Link>
          <a
            href="https://github.com/tylergibbs1/surge"
            target="_blank"
            rel="noreferrer"
            className="underline-offset-4 hover:underline"
          >
            Source code
            <span className="sr-only"> (opens in a new tab)</span>
          </a>
        </div>
      </div>
    </footer>
  )
}
