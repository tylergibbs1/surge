import Link from "next/link"

type SiteSection = "scoreboard" | "methodology" | "status" | "explorer"

const NAV_ITEMS: Array<{
  href: string
  label: string
  section: SiteSection
}> = [
  { href: "/", label: "Scoreboard", section: "scoreboard" },
  { href: "/methodology", label: "Methodology", section: "methodology" },
  { href: "/status", label: "Data status", section: "status" },
  { href: "/?ba=PJM&horizon=24", label: "53-BA explorer", section: "explorer" },
]

export function SiteHeader({ active }: { active: SiteSection }) {
  return (
    <header className="border-b border-white/10 bg-[#07111f] text-white">
      <div className="mx-auto flex max-w-7xl flex-col gap-5 px-5 py-5 sm:px-8 lg:flex-row lg:items-center lg:justify-between lg:px-10">
        <Link
          href="/"
          className="group flex w-fit items-center gap-3 rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-[#73a6ff] focus-visible:ring-offset-4 focus-visible:ring-offset-[#07111f]"
          aria-label="Surge forecast scoreboard home"
        >
          <span
            className="grid size-9 place-items-center border border-[#73a6ff]/60 bg-[#0d2038]"
            aria-hidden="true"
          >
            <span className="h-3 w-4 [transform:skewY(-28deg)] border-y-2 border-[#73a6ff]" />
          </span>
          <span className="leading-none">
            <span className="block text-lg font-semibold tracking-[-0.03em]">
              Surge
            </span>
            <span className="mt-1 block font-mono text-[10px] tracking-[0.18em] text-slate-400 uppercase">
              Public grid outlook
            </span>
          </span>
        </Link>

        <nav aria-label="Primary navigation">
          <ul className="flex flex-wrap items-center gap-x-1 gap-y-2 text-sm">
            {NAV_ITEMS.map((item) => {
              const current = item.section === active
              return (
                <li key={item.section}>
                  <Link
                    href={item.href}
                    aria-current={current ? "page" : undefined}
                    className={
                      current
                        ? "inline-flex min-h-10 items-center border-b-2 border-[#73a6ff] px-3 font-medium text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-[#73a6ff]"
                        : "inline-flex min-h-10 items-center border-b-2 border-transparent px-3 text-slate-300 transition-colors hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-[#73a6ff]"
                    }
                  >
                    {item.label}
                  </Link>
                </li>
              )
            })}
          </ul>
        </nav>
      </div>
    </header>
  )
}
