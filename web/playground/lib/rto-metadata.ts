import {
  RTO_CODES,
  type Interconnection,
  type RtoCode,
} from "@/lib/v2-contracts"
import { BA_PEAK_MW } from "@/lib/us-grid-geo"

export type RtoMetadata = {
  code: RtoCode
  name: string
  shortName: string
  timezone: string
  interconnection: Interconnection
  referencePeakMw: number
  detailHref: string
}

export const RTO_METADATA: Record<RtoCode, RtoMetadata> = {
  PJM: {
    code: "PJM",
    name: "PJM Interconnection",
    shortName: "PJM",
    timezone: "America/New_York",
    interconnection: "Eastern",
    referencePeakMw: BA_PEAK_MW.PJM,
    detailHref: "/?ba=PJM&horizon=24",
  },
  CISO: {
    code: "CISO",
    name: "California Independent System Operator",
    shortName: "CAISO",
    timezone: "America/Los_Angeles",
    interconnection: "Western",
    referencePeakMw: BA_PEAK_MW.CISO,
    detailHref: "/?ba=CISO&horizon=24",
  },
  ERCO: {
    code: "ERCO",
    name: "Electric Reliability Council of Texas",
    shortName: "ERCOT",
    timezone: "America/Chicago",
    interconnection: "Texas",
    referencePeakMw: BA_PEAK_MW.ERCO,
    detailHref: "/?ba=ERCO&horizon=24",
  },
  MISO: {
    code: "MISO",
    name: "Midcontinent Independent System Operator",
    shortName: "MISO",
    timezone: "America/Chicago",
    interconnection: "Eastern",
    referencePeakMw: BA_PEAK_MW.MISO,
    detailHref: "/?ba=MISO&horizon=24",
  },
  NYIS: {
    code: "NYIS",
    name: "New York Independent System Operator",
    shortName: "NYISO",
    timezone: "America/New_York",
    interconnection: "Eastern",
    referencePeakMw: BA_PEAK_MW.NYIS,
    detailHref: "/?ba=NYIS&horizon=24",
  },
  ISNE: {
    code: "ISNE",
    name: "ISO New England",
    shortName: "ISO-NE",
    timezone: "America/New_York",
    interconnection: "Eastern",
    referencePeakMw: BA_PEAK_MW.ISNE,
    detailHref: "/?ba=ISNE&horizon=24",
  },
  SWPP: {
    code: "SWPP",
    name: "Southwest Power Pool",
    shortName: "SPP",
    timezone: "America/Chicago",
    interconnection: "Eastern",
    referencePeakMw: BA_PEAK_MW.SWPP,
    detailHref: "/?ba=SWPP&horizon=24",
  },
}

export const RTOS = RTO_CODES.map((code) => RTO_METADATA[code])

export function isRtoCode(value: string): value is RtoCode {
  return (RTO_CODES as readonly string[]).includes(value)
}
