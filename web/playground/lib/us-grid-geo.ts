// BA geographic + display metadata for all 53 demand-reporting balancing
// authorities in EIA-930. Source of truth is src/surge/bas.py; this file
// should stay in sync (regenerate if the registry changes).
//
// State → BA is a cartographic simplification — many states overlap several
// BAs, and we assign the majority-footprint BA for colouring only. Not a
// regulatory assignment. Gen-only BAs are intentionally excluded here;
// they're visible via the API's /bas?include_gen_only=true endpoint.

export type BaCode =
  | "PJM" | "CISO" | "ERCO" | "MISO" | "NYIS" | "ISNE" | "SWPP"
  | "SOCO" | "TVA" | "FPL" | "DUK" | "CPLE" | "BPAT" | "FPC" | "AZPS" | "LGEE"
  | "PSCO" | "SRP" | "NEVP" | "PACE" | "LDWP" | "SCEG" | "PSEI" | "SC" | "TEC"
  | "AECI" | "PGE" | "IPCO" | "FMPP" | "PACW" | "WACM" | "BANC" | "JEA"
  | "TEPC" | "SEC" | "WALC" | "CPLW" | "EPE" | "PNM" | "NWMT" | "AVA" | "SCL"
  | "SPA" | "IID" | "TPWR" | "WAUW" | "TAL" | "TIDC" | "GCPD" | "GVL" | "CHPD"
  | "DOPD" | "HST"

// Listed largest-peak first so the selector and map labels prioritise the
// BAs a user is most likely to want.
export const BAS: BaCode[] = [
  "PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP",
  "SOCO", "TVA", "FPL", "DUK", "CPLE", "BPAT", "FPC", "AZPS", "LGEE",
  "PSCO", "SRP", "NEVP", "PACE", "LDWP", "SCEG", "PSEI", "SC", "TEC",
  "AECI", "PGE", "IPCO", "FMPP", "PACW", "WACM", "BANC", "JEA",
  "TEPC", "SEC", "WALC", "CPLW", "EPE", "PNM", "NWMT", "AVA", "SCL",
  "SPA", "IID", "TPWR", "WAUW", "TAL", "TIDC", "GCPD", "GVL", "CHPD",
  "DOPD", "HST",
]

// [longitude, latitude]. Approx load-weighted centre of each BA footprint.
export const BA_COORDS: Record<BaCode, [number, number]> = {
  PJM:  [-79.0, 39.5],  CISO: [-119.4, 36.8], ERCO: [-99.0, 31.0],
  MISO: [-91.0, 43.0],  NYIS: [-75.5, 43.0],  ISNE: [-71.5, 43.7],
  SWPP: [-98.5, 37.0],
  SOCO: [-84.4, 33.0],  TVA:  [-86.3, 35.5],  FPL:  [-80.7, 27.3],
  DUK:  [-81.0, 35.3],  CPLE: [-78.8, 35.9],  BPAT: [-121.5, 45.5],
  FPC:  [-82.3, 28.8],  AZPS: [-112.1, 33.4], LGEE: [-85.8, 38.0],
  PSCO: [-104.7, 39.9], SRP:  [-111.9, 33.4], NEVP: [-115.2, 36.1],
  PACE: [-111.9, 40.8], LDWP: [-118.4, 34.2], SCEG: [-81.1, 34.0],
  PSEI: [-122.3, 47.4], SC:   [-79.5, 33.7],  TEC:  [-82.4, 28.0],
  AECI: [-93.3, 37.2],  PGE:  [-122.6, 45.3], IPCO: [-116.2, 43.6],
  FMPP: [-81.3, 28.5],  PACW: [-122.9, 42.5], WACM: [-105.2, 42.0],
  BANC: [-121.5, 38.5], JEA:  [-81.7, 30.3],  TEPC: [-110.9, 32.1],
  SEC:  [-82.2, 28.8],  WALC: [-113.0, 34.5], CPLW: [-82.5, 35.6],
  EPE:  [-106.4, 31.8], PNM:  [-106.6, 35.1], NWMT: [-108.5, 46.0],
  AVA:  [-117.5, 47.6], SCL:  [-122.3, 47.5], SPA:  [-92.3, 34.8],
  IID:  [-115.5, 33.0], TPWR: [-122.6, 47.1], WAUW: [-101.0, 46.0],
  TAL:  [-84.3, 30.4],  TIDC: [-120.8, 37.5], GCPD: [-119.3, 47.1],
  GVL:  [-82.3, 29.7],  CHPD: [-120.3, 47.4], DOPD: [-120.2, 47.8],
  HST:  [-80.4, 25.5],
}

// Published historical demand peaks (MW). Approximate, stable year-over-year;
// used for color scaling on the map only.
export const BA_PEAK_MW: Record<BaCode, number> = {
  PJM: 165_500, CISO: 52_000, ERCO: 85_500, MISO: 127_000, NYIS: 32_500,
  ISNE: 26_000, SWPP: 56_000,
  SOCO: 40_000, TVA: 34_000, FPL: 25_000, DUK: 23_000, CPLE: 14_000,
  BPAT: 12_000, FPC: 12_000, AZPS: 8_500, LGEE: 8_000, PSCO: 8_000,
  SRP: 8_000, NEVP: 7_500, PACE: 7_500, LDWP: 6_500, SCEG: 5_500,
  PSEI: 5_000, SC: 5_000, TEC: 4_600, AECI: 4_000, PGE: 4_000,
  IPCO: 3_700, FMPP: 3_500, PACW: 3_500, WACM: 3_500, BANC: 3_300,
  JEA: 3_200, TEPC: 3_000, SEC: 2_500, WALC: 2_500, CPLW: 2_300,
  EPE: 2_200, PNM: 2_200, NWMT: 1_900, AVA: 1_800, SCL: 1_800,
  SPA: 1_500, IID: 1_000, TPWR: 850, WAUW: 700, TAL: 650,
  TIDC: 620, GCPD: 550, GVL: 450, CHPD: 350, DOPD: 200, HST: 100,
}

export const BA_LABEL: Record<BaCode, string> = {
  PJM: "PJM Interconnection", CISO: "California ISO", ERCO: "ERCOT",
  MISO: "Midcontinent ISO", NYIS: "New York ISO", ISNE: "ISO-NE",
  SWPP: "Southwest Power Pool",
  SOCO: "Southern Company", TVA: "Tennessee Valley Authority",
  FPL: "Florida Power & Light", DUK: "Duke Energy Carolinas",
  CPLE: "Duke Energy Progress East", BPAT: "Bonneville Power",
  FPC: "Duke Energy Florida", AZPS: "Arizona Public Service",
  LGEE: "LG&E / Kentucky Utilities", PSCO: "Xcel Colorado",
  SRP: "Salt River Project", NEVP: "Nevada Power", PACE: "PacifiCorp East",
  LDWP: "LA Dept of Water & Power", SCEG: "Dominion Energy SC",
  PSEI: "Puget Sound Energy", SC: "Santee Cooper", TEC: "Tampa Electric",
  AECI: "Associated Electric Coop", PGE: "Portland General Electric",
  IPCO: "Idaho Power", FMPP: "Florida Municipal Power Pool",
  PACW: "PacifiCorp West", WACM: "WAPA Rocky Mountain", BANC: "BA of N. California",
  JEA: "JEA (Jacksonville)", TEPC: "Tucson Electric Power",
  SEC: "Seminole Electric Coop", WALC: "WAPA Desert SW",
  CPLW: "Duke Energy Progress West", EPE: "El Paso Electric",
  PNM: "PNM (New Mexico)", NWMT: "NorthWestern Energy (MT)",
  AVA: "Avista", SCL: "Seattle City Light", SPA: "Southwestern Power Admin",
  IID: "Imperial Irrigation District", TPWR: "Tacoma Power",
  WAUW: "WAPA Upper Great Plains West", TAL: "Tallahassee",
  TIDC: "Turlock Irrigation District", GCPD: "Grant County PUD",
  GVL: "Gainesville Utilities", CHPD: "Chelan County PUD",
  DOPD: "Douglas County PUD", HST: "Homestead",
}

// Rough standard-time UTC offsets for diurnal shading on the chart.
export const BA_UTC_OFFSET: Record<BaCode, number> = {
  PJM: -5, CISO: -8, ERCO: -6, MISO: -6, NYIS: -5, ISNE: -5, SWPP: -6,
  SOCO: -6, TVA: -6, FPL: -5, DUK: -5, CPLE: -5, BPAT: -8, FPC: -5,
  AZPS: -7, LGEE: -5, PSCO: -7, SRP: -7, NEVP: -8, PACE: -7, LDWP: -8,
  SCEG: -5, PSEI: -8, SC: -5, TEC: -5, AECI: -6, PGE: -8, IPCO: -7,
  FMPP: -5, PACW: -8, WACM: -7, BANC: -8, JEA: -5, TEPC: -7, SEC: -5,
  WALC: -7, CPLW: -5, EPE: -7, PNM: -7, NWMT: -7, AVA: -8, SCL: -8,
  SPA: -6, IID: -8, TPWR: -8, WAUW: -7, TAL: -5, TIDC: -8, GCPD: -8,
  GVL: -5, CHPD: -8, DOPD: -8, HST: -5,
}

// Best-fit BA per US state — majority footprint chosen for states that
// overlap multiple BAs. States where no single BA dominates (e.g. split
// between several small utilities) stay undefined and render neutral.
export const STATE_TO_BA: Record<string, BaCode | undefined> = {
  // PJM core
  PA: "PJM", NJ: "PJM", MD: "PJM", DE: "PJM", VA: "PJM", WV: "PJM",
  OH: "PJM", IN: "PJM", DC: "PJM", IL: "PJM",

  // MISO
  MN: "MISO", WI: "MISO", IA: "MISO", MI: "MISO", AR: "MISO",
  MS: "MISO", LA: "MISO", MO: "MISO", ND: "MISO", SD: "MISO",

  // SPP
  OK: "SWPP", KS: "SWPP", NE: "SWPP",

  // Single-state ISOs
  CA: "CISO", TX: "ERCO", NY: "NYIS",

  // ISO-NE
  ME: "ISNE", NH: "ISNE", VT: "ISNE", MA: "ISNE", CT: "ISNE", RI: "ISNE",

  // Non-ISO Eastern
  GA: "SOCO", AL: "SOCO",
  TN: "TVA", KY: "LGEE",
  NC: "DUK", SC: "DUK",
  FL: "FPL",

  // Non-ISO Western
  WA: "BPAT", OR: "BPAT",
  ID: "IPCO", MT: "NWMT", WY: "PACE",
  UT: "PACE", CO: "PSCO", NV: "NEVP",
  AZ: "AZPS", NM: "PNM",
}
