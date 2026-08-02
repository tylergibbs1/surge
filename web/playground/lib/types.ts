export type ForecastPoint = {
  ts_utc: string
  median_mw: number
  p10_mw: number
  p90_mw: number
  /** Forecast-vintage temperature when a future feature spec provides it.
   *  Always null/absent under the v0.2 load-v2-core contract. */
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

// Selector options. Mirrors the `BAS` + `BA_LABEL` pair in us-grid-geo; kept
// here so UI components that don't need geographic metadata can import from
// a flatter module. Source of truth is src/surge/bas.py.
import { BAS as BA_CODES, BA_LABEL } from "./us-grid-geo"

export const BAS = BA_CODES.map((code) => ({
  code,
  name: BA_LABEL[code],
}))
