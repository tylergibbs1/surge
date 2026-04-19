// BA geographic metadata — centroids, historical peak demand, and a coarse
// state-to-BA mapping used to shade the polygon overlay. States with split
// footprints pick the majority BA; this is cartographic simplification, not
// a regulatory assignment.

export type BaCode = "PJM" | "CISO" | "ERCO" | "MISO" | "NYIS" | "ISNE" | "SWPP"

export const BAS: BaCode[] = [
  "PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP",
]

export const BA_COORDS: Record<BaCode, [number, number]> = {
  // [longitude, latitude]
  PJM:  [-79.0, 39.5],  // DC / Mid-Atlantic
  CISO: [-119.4, 36.8], // Central CA
  ERCO: [-99.0, 31.0],  // Central TX
  MISO: [-91.0, 43.0],  // Iowa/Minnesota
  NYIS: [-75.5, 43.0],  // Central NY
  ISNE: [-71.5, 43.7],  // NH/MA
  SWPP: [-98.5, 37.0],  // OK / KS
}

// Published historical demand peaks (MW). Approximate and stable across
// recent seasons; used only for color scaling, not for forecasting.
export const BA_PEAK_MW: Record<BaCode, number> = {
  PJM: 165_500,
  CISO: 52_000,
  ERCO: 85_500,
  MISO: 127_000,
  NYIS: 32_500,
  ISNE: 26_000,
  SWPP: 56_000,
}

export const BA_LABEL: Record<BaCode, string> = {
  PJM: "PJM Interconnection",
  CISO: "California ISO",
  ERCO: "ERCOT",
  MISO: "Midcontinent ISO",
  NYIS: "New York ISO",
  ISNE: "ISO-NE",
  SWPP: "Southwest Power Pool",
}

// Rough standard-time UTC offsets for diurnal shading. Close enough that
// the overnight bands on the chart line up visually; not used for any
// computation that needs correctness.
export const BA_UTC_OFFSET: Record<BaCode, number> = {
  PJM: -5, NYIS: -5, ISNE: -5,   // Eastern
  MISO: -6, ERCO: -6, SWPP: -6,  // Central
  CISO: -8,                      // Pacific
}

// Best-fit BA per US state (2-letter code). States overlapping multiple
// BAs get their majority-footprint BA. Unassigned states (WECC-only, FRCC,
// SERC) stay undefined and render as a neutral mid-gray.
export const STATE_TO_BA: Record<string, BaCode | undefined> = {
  // PJM core
  PA: "PJM", NJ: "PJM", MD: "PJM", DE: "PJM", VA: "PJM", WV: "PJM",
  OH: "PJM", IN: "PJM", DC: "PJM", IL: "PJM",       // Chicago = ComEd/PJM

  // CAISO
  CA: "CISO",

  // ERCOT
  TX: "ERCO",

  // MISO
  MN: "MISO", WI: "MISO", IA: "MISO", MI: "MISO", AR: "MISO",
  MS: "MISO", LA: "MISO", MO: "MISO", ND: "MISO", SD: "MISO",

  // NYISO
  NY: "NYIS",

  // ISO-NE
  ME: "ISNE", NH: "ISNE", VT: "ISNE", MA: "ISNE", CT: "ISNE", RI: "ISNE",

  // SPP
  OK: "SWPP", KS: "SWPP", NE: "SWPP", NM: "SWPP",
}
