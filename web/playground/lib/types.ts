export type ForecastPoint = {
  ts_utc: string
  median_mw: number
  p10_mw: number
  p90_mw: number
  /** Assumed future temperature (°C) at BA centroid station. May be null
   *  on older API versions. */
  temp_c?: number | null
}

export type ForecastResponse = {
  ba: string
  model: string
  as_of_utc: string
  context_start_utc: string
  context_end_utc: string
  horizon: number
  units: string
  points: ForecastPoint[]
}

export const BAS = [
  { code: "PJM",  name: "PJM Interconnection (DC, Philly, Chicago)" },
  { code: "CISO", name: "California ISO" },
  { code: "ERCO", name: "ERCOT (Texas)" },
  { code: "MISO", name: "Midcontinent ISO" },
  { code: "NYIS", name: "New York ISO" },
  { code: "ISNE", name: "ISO-NE (New England)" },
  { code: "SWPP", name: "Southwest Power Pool" },
] as const
