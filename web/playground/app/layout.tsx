import type { Metadata, Viewport } from "next"

import "./globals.css"
import { SwrProvider } from "@/components/swr-provider"
import { ThemeProvider } from "@/components/theme-provider"
import { ThirdPartyAnalytics } from "@/components/third-party-analytics"

const DESCRIPTION =
  "A public, timestamped next-24-hour load forecast scoreboard for seven major US power markets, with explicit freshness and unavailable states."

export const metadata: Metadata = {
  metadataBase: new URL("https://surgeforecast.com"),
  title: {
    default: "Surge — open forecasts for the US power grid",
    template: "%s · Surge",
  },
  description: DESCRIPTION,
  applicationName: "Surge",
  authors: [{ name: "Tyler Gibbs", url: "https://github.com/tylergibbs1" }],
  keywords: [
    "electricity grid",
    "load forecasting",
    "Chronos-2",
    "open source",
    "PJM",
    "CAISO",
    "ERCOT",
    "MISO",
    "NYISO",
    "ISO-NE",
    "SPP",
    "day-ahead forecast",
    "probabilistic",
    "energy",
    "balancing authority",
    "EIA-930",
  ],
  category: "science",
  openGraph: {
    type: "website",
    url: "/",
    siteName: "Surge",
    title: "Surge — open forecasts for the US power grid",
    description: DESCRIPTION,
  },
  twitter: {
    card: "summary_large_image",
    title: "Surge — open forecasts for the US power grid",
    description: DESCRIPTION,
    creator: "@tylergibbs1",
  },
  robots: { index: true, follow: true },
  alternates: { canonical: "/" },
}

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0a0a" },
  ],
}

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning className="font-sans antialiased">
      <body>
        {/* Keyboard/AT users bypass site navigation and land on main content.
            `sr-only` hides the link visually; focus brings it back on Tab.
            `z-[60]` clears the radix portals (z-50) used by v1 selects. */}
        <a
          href="#main"
          className="sr-only bg-background text-foreground focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-[60] focus:rounded-md focus:px-3 focus:py-2 focus:text-sm focus:font-medium focus:ring-2 focus:ring-foreground/60"
        >
          Skip to content
        </a>
        <ThemeProvider>
          <SwrProvider>{children}</SwrProvider>
        </ThemeProvider>
        <ThirdPartyAnalytics />
      </body>
    </html>
  )
}
