"use client"

import { useEffect, useMemo, useRef } from "react"
import useSWR from "swr"

import {
  Map,
  MapControls,
  MapMarker,
  MarkerContent,
  MarkerPopup,
  useMap,
} from "@/components/ui/map"
import {
  BA_COORDS,
  BA_LABEL,
  BA_PEAK_MW,
  BAS,
  STATE_TO_BA,
  type BaCode,
} from "@/lib/us-grid-geo"
import type { ForecastResponse } from "@/lib/types"

// Vendored under /public/geo — removes runtime dependency on a
// third-party GitHub repo that could be deleted, rate-limited, or taken
// over. (Path is `/geo/` not `/data/` because our top-level .gitignore
// excludes any `data/` directory.) Same Public Domain source:
// PublicaMundi/MappingAPI.
const STATES_GEOJSON_URL = "/geo/us-states.json"

// US-state full-name → 2-letter code, needed because the GeoJSON above
// carries `properties.name` as the long form ("New York", not "NY").
const STATE_NAME_TO_CODE: Record<string, string> = {
  "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
  "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
  "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
  "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
  "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
  "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
  "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
  "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
  "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
  "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
  "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
  "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
  "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
  "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
  "Puerto Rico": "PR",
}

// ISOs get distinctive hues; non-ISO BAs inherit a muted interconnect tint
// (warm olive for Eastern, cool sage for Western) so the map still reads as
// "color == organized market" without going visually noisy at 53 BAs.
const _ISO_COLOR: Partial<Record<BaCode, string>> = {
  PJM:  "#4F8DF7",  // indigo
  CISO: "#F4B740",  // amber
  ERCO: "#E05263",  // coral
  MISO: "#5CC8A2",  // teal
  NYIS: "#A770F0",  // violet
  ISNE: "#F07FB0",  // pink
  SWPP: "#6BC4E0",  // sky
}
// Non-RTO tints brightened from the original muted olive/sage — the old
// #7A8B6E / #6E8A8B read as near-black against the dark basemap at the
// 0.3–0.5 opacity the choropleth settles on, so most of the Western and
// Southeast US looked like empty space. These are ~40% lighter while
// still staying deliberately quieter than the 7 ISO accent hues.
const _EASTERN_NONISO = "#A3B894"  // lighter olive-green
const _WESTERN_NONISO = "#95B6B8"  // lighter sage-teal
const _INTERCONNECT: Record<BaCode, "Eastern" | "Western" | "Texas"> = {
  PJM: "Eastern", CISO: "Western", ERCO: "Texas", MISO: "Eastern",
  NYIS: "Eastern", ISNE: "Eastern", SWPP: "Eastern",
  SOCO: "Eastern", TVA: "Eastern", FPL: "Eastern", DUK: "Eastern",
  CPLE: "Eastern", BPAT: "Western", FPC: "Eastern", AZPS: "Western",
  LGEE: "Eastern", PSCO: "Western", SRP: "Western", NEVP: "Western",
  PACE: "Western", LDWP: "Western", SCEG: "Eastern", PSEI: "Western",
  SC: "Eastern", TEC: "Eastern", AECI: "Eastern", PGE: "Western",
  IPCO: "Western", FMPP: "Eastern", PACW: "Western", WACM: "Western",
  BANC: "Western", JEA: "Eastern", TEPC: "Western", SEC: "Eastern",
  WALC: "Western", CPLW: "Eastern", EPE: "Western", PNM: "Western",
  NWMT: "Western", AVA: "Western", SCL: "Western", SPA: "Eastern",
  IID: "Western", TPWR: "Western", WAUW: "Western", TAL: "Eastern",
  TIDC: "Western", GCPD: "Western", GVL: "Eastern", CHPD: "Western",
  DOPD: "Western", HST: "Eastern",
}
const BA_FILL_COLOR: Record<BaCode, string> = Object.fromEntries(
  BAS.map((b) => [
    b,
    _ISO_COLOR[b] ??
      (_INTERCONNECT[b] === "Western" ? _WESTERN_NONISO : _EASTERN_NONISO),
  ]),
) as Record<BaCode, string>

type Props = {
  horizon?: number
  selected: BaCode
  onSelect: (ba: BaCode) => void
}

// Imperative layer for BA state polygons. Lives as a sibling component so
// it can use `useMap()` inside <Map>'s context provider.
function GridPolygons({
  forecasts,
  selected,
}: {
  forecasts: Record<BaCode, ForecastResponse | undefined>
  selected: BaCode
}) {
  const { map, isLoaded } = useMap()

  useEffect(() => {
    if (!map || !isLoaded) return
    let cancelled = false

    ;(async () => {
      // Fetch + enrich once; paint updates are cheap setPaintProperty calls.
      const res = await fetch(STATES_GEOJSON_URL, { cache: "force-cache" })
      if (!res.ok || cancelled) return
      const data = (await res.json()) as GeoJSON.FeatureCollection
      for (const f of data.features) {
        const name = (f.properties as { name?: string })?.name ?? ""
        const code = STATE_NAME_TO_CODE[name]
        const ba = code ? STATE_TO_BA[code] : undefined
        ;(f.properties as Record<string, unknown>).ba = ba ?? null
        ;(f.properties as Record<string, unknown>).state = code ?? null
      }
      if (cancelled || !map) return

      if (!map.getSource("us-states")) {
        map.addSource("us-states", { type: "geojson", data })
        // Mapbox GL's `match` expression requires alternating
        // literal/value pairs ending in a default. The TS types want a
        // statically-known tuple, so we build the array dynamically and
        // cast — the runtime shape matches the spec.
        const fillColorExpr = [
          "match",
          ["get", "ba"],
          ...BAS.flatMap((b) => [b, BA_FILL_COLOR[b]]),
          "hsl(0 0% 22%)",
        ] as never
        map.addLayer({
          id: "ba-fills",
          type: "fill",
          source: "us-states",
          paint: {
            "fill-color": fillColorExpr,
            "fill-opacity": [
              "case",
              ["has", "ba"],
              // 0.40 baseline for any mapped BA that doesn't have a
              // forecast yet — was 0.25, invisible against the dark map.
              ["coalesce", ["feature-state", "pct_peak"], 0.40],
              0.08,
            ],
          },
        })
        map.addLayer({
          id: "ba-fills-outline",
          type: "line",
          source: "us-states",
          paint: {
            "line-color": ["case",
              ["has", "ba"], "hsla(0 0% 100% / 0.35)", "hsla(0 0% 100% / 0.08)"],
            "line-width": ["case",
              ["==", ["get", "ba"], ["literal", selected]], 2.5, 0.6],
          },
        })
      }
    })()

    return () => { cancelled = true }
  }, [map, isLoaded, selected])

  // Paint % of peak when forecasts update.
  useEffect(() => {
    if (!map || !isLoaded) return
    const src = map.getSource("us-states") as unknown as {
      _data?: GeoJSON.FeatureCollection
    } | undefined
    if (!src?._data) return

    for (const f of src._data.features) {
      const ba = (f.properties as { ba?: BaCode | null })?.ba
      if (!ba || !forecasts[ba] || f.id == null) continue
      const peak = forecasts[ba]!.points.reduce(
        (m, p) => Math.max(m, p.median_mw), 0)
      const pct = Math.min(1, peak / BA_PEAK_MW[ba])
      // feature-state requires feature.id; this GeoJSON has numeric IDs.
      // Narrower dynamic range (0.40..0.80) than before (0.18..0.73) —
      // the old floor made overnight/low-demand BAs look un-filled,
      // which read as "this state isn't part of surge" rather than
      // "this BA is at 40% of its peak."
      map.setFeatureState(
        { source: "us-states", id: f.id as number },
        { pct_peak: 0.40 + pct * 0.40 },
      )
    }
  }, [map, isLoaded, forecasts])

  // Re-style outline when `selected` changes.
  useEffect(() => {
    if (!map || !isLoaded || !map.getLayer("ba-fills-outline")) return
    map.setPaintProperty("ba-fills-outline", "line-width", [
      "case",
      ["==", ["get", "ba"], selected],
      2.5,
      0.6,
    ])
  }, [map, isLoaded, selected])

  return null
}

function peakInfo(fc?: ForecastResponse) {
  if (!fc || fc.points.length === 0) return null
  // Single-pass; no intermediate allocations.
  let maxP = -Infinity
  let maxTs = ""
  for (let i = 0; i < fc.points.length; i++) {
    const p = fc.points[i]
    if (p.median_mw > maxP) { maxP = p.median_mw; maxTs = p.ts_utc }
  }
  return { peakGW: maxP / 1000, peakTs: maxTs }
}

export function GridMap({ horizon = 24, selected, onSelect }: Props) {
  // Fetch all 7 forecasts in parallel via SWR (one key per BA so cache keys
  // are stable and swr-dedup works).
  const forecasts: Record<BaCode, ForecastResponse | undefined> = useMemo(
    () => Object.fromEntries(BAS.map((b) => [b, undefined])) as Record<
      BaCode,
      ForecastResponse | undefined
    >,
    [],
  )

  // Always fetch the full 168 h series so the map shares cache keys with
  // the chart — one inference per BA, regardless of where the slider is.
  for (const ba of BAS) {
    /* eslint-disable react-hooks/rules-of-hooks */
    const { data } = useSWR<ForecastResponse>(
      `/api/forecast?ba=${ba}&horizon=168`,
      (url: string) => fetch(url).then((r) => r.ok ? r.json() : undefined),
      { revalidateOnFocus: false, dedupingInterval: 300_000 },
    )
    // Slice locally to match the user's current horizon for shading/pulse.
    forecasts[ba] = useMemo(() => {
      if (!data) return undefined
      return horizon >= data.points.length
        ? data
        : { ...data, horizon, points: data.points.slice(0, horizon) }
    }, [data, horizon])
    /* eslint-enable react-hooks/rules-of-hooks */
  }

  return (
    <div className="border-border relative h-[420px] w-full overflow-hidden rounded-lg border">
      <Map
        viewport={{ center: [-96, 39], zoom: 3.4 }}
        className="h-full w-full"
      >
        <MapControls showCompass={false} showFullscreen showZoom showLocate={false} />
        <GridPolygons forecasts={forecasts} selected={selected} />

        {BAS.map((ba) => {
          const info = peakInfo(forecasts[ba])
          const pctPeak = info ? info.peakGW * 1000 / BA_PEAK_MW[ba] : 0
          const sizePx = 14 + Math.round(pctPeak * 22)
          return (
            <MapMarker key={ba} longitude={BA_COORDS[ba][0]} latitude={BA_COORDS[ba][1]}>
              <MarkerContent>
                <button
                  type="button"
                  onClick={() => onSelect(ba)}
                  aria-label={`Select ${BA_LABEL[ba]}`}
                  // Invisible hit box expansion for touch devices only —
                  // the visible pin can be as small as 14 px for low-demand
                  // BAs. On desktop, closely-spaced pins (PJM/NYIS,
                  // CISO/BANC, the three Florida BAs) overlap a 40 px
                  // invisible target, so the later-painted pin would steal
                  // clicks. Gating on pointer:coarse limits the expansion
                  // to touch where accurate clicking is genuinely hard and
                  // pin proximity is less of an issue (fingers are big,
                  // maps are zoomed in).
                  className="group/pin relative flex items-center justify-center rounded-full transition-transform duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-white/80 active:scale-95 pointer-coarse:before:absolute pointer-coarse:before:left-1/2 pointer-coarse:before:top-1/2 pointer-coarse:before:h-10 pointer-coarse:before:w-10 pointer-coarse:before:-translate-x-1/2 pointer-coarse:before:-translate-y-1/2 pointer-coarse:before:content-['']"
                  style={{ width: sizePx, height: sizePx }}
                >
                  {selected === ba ? (
                    <span
                      className="absolute inset-0 animate-ping rounded-full opacity-75"
                      style={{ background: BA_FILL_COLOR[ba] }}
                    />
                  ) : null}
                  <span
                    className="relative block rounded-full border-2 border-white/90 shadow-lg"
                    style={{
                      width: sizePx,
                      height: sizePx,
                      background: BA_FILL_COLOR[ba],
                    }}
                  />
                  <span className="pointer-events-none absolute top-full mt-1 font-mono text-[10px] font-semibold tracking-wide whitespace-nowrap text-white mix-blend-difference">
                    {ba}
                  </span>
                </button>
              </MarkerContent>

              <MarkerPopup>
                <div className="bg-popover text-popover-foreground min-w-[180px] rounded-md border p-3 shadow-md">
                  <div className="mb-1 flex items-center gap-2">
                    <span
                      className="size-2.5 rounded-full"
                      style={{ background: BA_FILL_COLOR[ba] }}
                    />
                    <span className="text-sm font-medium">{BA_LABEL[ba]}</span>
                  </div>
                  <div className="text-muted-foreground font-mono text-xs tabular-nums">
                    {info ? (
                      <>
                        <div>peak forecast: {info.peakGW.toFixed(1)} GW</div>
                        <div>
                          {(pctPeak * 100).toFixed(0)}% of all-time peak (
                          {(BA_PEAK_MW[ba] / 1000).toFixed(0)} GW)
                        </div>
                      </>
                    ) : (
                      <div>loading…</div>
                    )}
                  </div>
                </div>
              </MarkerPopup>
            </MapMarker>
          )
        })}
      </Map>
    </div>
  )
}
