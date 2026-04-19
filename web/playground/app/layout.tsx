import type { Metadata, Viewport } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import { Analytics } from "@vercel/analytics/next"
import { SpeedInsights } from "@vercel/speed-insights/next"

import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import { cn } from "@/lib/utils"

const geist = Geist({ subsets: ["latin"], variable: "--font-sans" })
const fontMono = Geist_Mono({ subsets: ["latin"], variable: "--font-mono" })

const DESCRIPTION =
  "Open probabilistic day-ahead load forecasts for every US balancing authority that publishes a demand series to EIA-930 (53 total). Chronos-2 fine-tuned on 7 years of public data — matches utility-internal accuracy, free and open-source."

export const metadata: Metadata = {
  metadataBase: new URL("https://surge-omega-nine.vercel.app"),
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
        <ThemeProvider>{children}</ThemeProvider>
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  )
}
