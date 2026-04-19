import type { Metadata, Viewport } from "next"
import { Geist, Geist_Mono } from "next/font/google"

import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import { ThirdPartyAnalytics } from "@/components/third-party-analytics"
import { cn } from "@/lib/utils"

const geist = Geist({ subsets: ["latin"], variable: "--font-sans" })
const fontMono = Geist_Mono({ subsets: ["latin"], variable: "--font-mono" })

const DESCRIPTION =
  "Open probabilistic day-ahead load forecasts for every US balancing authority that publishes a demand series to EIA-930 (53 total). Chronos-2 fine-tuned on 7 years of public data — matches utility-internal accuracy, free and open-source."

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
    "electricity grid", "load forecasting", "Chronos-2", "open source",
    "PJM", "CAISO", "ERCOT", "MISO", "NYISO", "ISO-NE", "SPP",
    "Southern Company", "TVA", "Duke Energy", "Florida Power & Light",
    "Bonneville Power", "PacifiCorp", "Xcel", "Arizona Public Service",
    "day-ahead forecast", "probabilistic", "energy", "balancing authority",
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
    { media: "(prefers-color-scheme: dark)",  color: "#0a0a0a" },
  ],
}

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn("antialiased", fontMono.variable, "font-sans", geist.variable)}
    >
      <body>
        {/* Keyboard/AT users bypass the glossary + nav pills and land on
            the first interactive region of the page. `sr-only` hides it
            visually; `focus:not-sr-only` brings it back on Tab.
            `z-[60]` clears the radix portals (z-50) used by selects. */}
        <a
          href="#main"
          className="sr-only bg-background text-foreground focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-[60] focus:rounded-md focus:px-3 focus:py-2 focus:text-sm focus:font-medium focus:ring-2 focus:ring-foreground/60"
        >
          Skip to content
        </a>
        <ThemeProvider>{children}</ThemeProvider>
        <ThirdPartyAnalytics />
      </body>
    </html>
  )
}
