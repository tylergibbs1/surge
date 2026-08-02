import Link from "next/link"
import { Suspense } from "react"

import { Glossary } from "@/components/glossary"
import { PlaygroundApp } from "@/components/playground-app"
import { SiteFooter } from "@/components/site-footer"
import { SiteHeader } from "@/components/site-header"
import { UsDemandHero } from "@/components/us-demand-hero"

export function LegacyExplorerPage() {
  return (
    <div className="min-h-svh bg-background">
      <SiteHeader active="explorer" />
      <div className="mx-auto max-w-5xl space-y-7 px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
        <header className="space-y-3">
          <p className="font-mono text-xs tracking-[0.16em] text-[#315e9f] uppercase dark:text-[#8cb5f7]">
            Legacy v1 explorer
          </p>
          <h1 className="text-4xl leading-none font-semibold tracking-[-0.045em] md:text-5xl">
            Balancing-authority detail
          </h1>
          <p className="max-w-3xl text-muted-foreground">
            Inspect probabilistic model output and public EIA observations for
            the broader 53-BA research set. The v0.2 homepage narrows its public
            scoreboard to seven major RTOs and makes freshness explicit.
          </p>
          <Link
            href="/"
            className="inline-flex min-h-10 items-center text-sm font-semibold text-[#315e9f] underline-offset-4 hover:underline dark:text-[#8cb5f7]"
          >
            ← Return to the seven-RTO scoreboard
          </Link>
        </header>

        <UsDemandHero />

        <main id="main">
          <Suspense fallback={null}>
            <PlaygroundApp preserveDetailRoute />
          </Suspense>
        </main>

        <Glossary />
      </div>
      <SiteFooter />
    </div>
  )
}
