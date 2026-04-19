"""Central registry of EIA-930 balancing authorities.

Single source of truth for:
  - BA codes and names
  - Interconnect (Eastern / Western / Texas)
  - Representative ASOS weather station (major load-centre airport)
  - `has_demand` flag — some BAs are generator- or transmission-only and
    publish no type=D (Demand) series to EIA-930; we skip those for load
    forecasting but keep them in the registry for completeness.
  - Approximate centroid, peak demand, and UTC offset (for the frontend map)

Anything that iterates over BAs should read from here, not hardcode a list.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class BA:
    code: str
    name: str
    interconnect: str          # "Eastern" | "Western" | "Texas"
    utc_offset: int            # standard-time UTC offset (used only for diurnal shading)
    station: str | None        # representative ASOS ICAO-minus-K (e.g. "DCA"); None if gen-only
    has_demand: bool           # publishes type=D to EIA-930
    is_rto: bool               # one of the 7 organized markets (PJM/CAISO/ERCOT/MISO/NYISO/ISO-NE/SPP)
    centroid: tuple[float, float]  # (lon, lat) approx load-weighted center
    peak_mw: int | None        # rough historical peak demand; None when unknown / gen-only


# Notes on curation:
#   * Stations are picked as the largest ASOS-reporting airport inside the
#     BA's load footprint. When two BAs share a city (e.g. FPC/TEC in Tampa,
#     AZPS/SRP in Phoenix), we pick distinct nearby airports where practical
#     so the `weather_hourly` store can key by station without collisions.
#   * has_demand=False BAs (14 of 67) are kept in the registry so the
#     frontend map and `/bas` endpoint show the full EIA-930 footprint, but
#     they're skipped by the demand-forecast pipeline.
#   * peak_mw is approximate, for color scaling only — not used for
#     forecasting. Numbers come from EIA-930 historical maxima and published
#     utility reports; treat as order-of-magnitude.
_REGISTRY: tuple[BA, ...] = (
    # ── 7 organized-market RTO/ISOs ──
    BA("PJM",  "PJM Interconnection",                   "Eastern", -5, "DCA", True, True,  (-79.0, 39.5), 165_500),
    BA("CISO", "California ISO",                        "Western", -8, "SFO", True, True,  (-119.4, 36.8),  52_000),
    BA("ERCO", "ERCOT",                                 "Texas",   -6, "AUS", True, True,  (-99.0, 31.0),   85_500),
    BA("MISO", "Midcontinent ISO",                      "Eastern", -6, "MSP", True, True,  (-91.0, 43.0),  127_000),
    BA("NYIS", "New York ISO",                          "Eastern", -5, "JFK", True, True,  (-75.5, 43.0),   32_500),
    BA("ISNE", "ISO New England",                       "Eastern", -5, "BOS", True, True,  (-71.5, 43.7),   26_000),
    BA("SWPP", "Southwest Power Pool",                  "Eastern", -6, "OKC", True, True,  (-98.5, 37.0),   56_000),

    # ── Large non-ISO utilities (Eastern Interconnection) ──
    BA("SOCO", "Southern Company",                      "Eastern", -6, "ATL", True, False, (-84.4, 33.0),   40_000),
    BA("TVA",  "Tennessee Valley Authority",            "Eastern", -6, "BNA", True, False, (-86.3, 35.5),   34_000),
    BA("DUK",  "Duke Energy Carolinas",                 "Eastern", -5, "CLT", True, False, (-81.0, 35.3),   23_000),
    BA("CPLE", "Duke Energy Progress East",             "Eastern", -5, "RDU", True, False, (-78.8, 35.9),   14_000),
    BA("CPLW", "Duke Energy Progress West",             "Eastern", -5, "AVL", True, False, (-82.5, 35.6),    2_300),
    BA("LGEE", "Louisville Gas & Electric / Kentucky Utilities", "Eastern", -5, "SDF", True, False, (-85.8, 38.0),  8_000),
    BA("AECI", "Associated Electric Cooperative",       "Eastern", -6, "SGF", True, False, (-93.3, 37.2),    4_000),
    BA("SCEG", "Dominion Energy South Carolina",        "Eastern", -5, "CAE", True, False, (-81.1, 34.0),    5_500),
    BA("SC",   "Santee Cooper",                         "Eastern", -5, "MYR", True, False, (-79.5, 33.7),    5_000),

    # ── Florida (Eastern) ──
    BA("FPL",  "Florida Power & Light",                 "Eastern", -5, "MIA", True, False, (-80.7, 27.3),   25_000),
    BA("FPC",  "Duke Energy Florida",                   "Eastern", -5, "MCO", True, False, (-82.3, 28.8),   12_000),
    BA("TEC",  "Tampa Electric",                        "Eastern", -5, "TPA", True, False, (-82.4, 28.0),    4_600),
    BA("FMPP", "Florida Municipal Power Pool",          "Eastern", -5, "ORL", True, False, (-81.3, 28.5),    3_500),
    BA("SEC",  "Seminole Electric Cooperative",         "Eastern", -5, "OCF", True, False, (-82.2, 28.8),    2_500),
    BA("JEA",  "JEA (Jacksonville)",                    "Eastern", -5, "JAX", True, False, (-81.7, 30.3),    3_200),
    BA("TAL",  "City of Tallahassee",                   "Eastern", -5, "TLH", True, False, (-84.3, 30.4),      650),
    BA("GVL",  "Gainesville Regional Utilities",        "Eastern", -5, "GNV", True, False, (-82.3, 29.7),      450),
    BA("HST",  "City of Homestead",                     "Eastern", -5, "TMB", True, False, (-80.4, 25.5),      100),

    # ── Western Interconnection — large ──
    BA("BPAT", "Bonneville Power Administration",       "Western", -8, "PDX", True, False, (-121.5, 45.5),  12_000),
    BA("PACE", "PacifiCorp East",                       "Western", -7, "SLC", True, False, (-111.9, 40.8),   7_500),
    BA("PACW", "PacifiCorp West",                       "Western", -8, "MFR", True, False, (-122.9, 42.5),   3_500),
    BA("AZPS", "Arizona Public Service",                "Western", -7, "PHX", True, False, (-112.1, 33.4),   8_500),
    BA("SRP",  "Salt River Project",                    "Western", -7, "IWA", True, False, (-111.9, 33.4),   8_000),
    BA("PSCO", "Public Service Company of Colorado",    "Western", -7, "DEN", True, False, (-104.7, 39.9),   8_000),
    BA("NEVP", "Nevada Power",                          "Western", -8, "LAS", True, False, (-115.2, 36.1),   7_500),
    BA("LDWP", "LA Dept of Water and Power",            "Western", -8, "BUR", True, False, (-118.4, 34.2),   6_500),
    BA("PGE",  "Portland General Electric",             "Western", -8, "SLE", True, False, (-122.6, 45.3),   4_000),
    BA("IPCO", "Idaho Power",                           "Western", -7, "BOI", True, False, (-116.2, 43.6),   3_700),
    BA("PSEI", "Puget Sound Energy",                    "Western", -8, "SEA", True, False, (-122.3, 47.4),   5_000),
    BA("SCL",  "Seattle City Light",                    "Western", -8, "BFI", True, False, (-122.3, 47.5),   1_800),
    BA("TPWR", "Tacoma Power",                          "Western", -8, "TIW", True, False, (-122.6, 47.1),      850),
    BA("AVA",  "Avista Corporation",                    "Western", -8, "GEG", True, False, (-117.5, 47.6),   1_800),
    BA("NWMT", "NorthWestern Energy (Montana)",         "Western", -7, "BIL", True, False, (-108.5, 46.0),   1_900),
    BA("PNM",  "Public Service Company of New Mexico",  "Western", -7, "ABQ", True, False, (-106.6, 35.1),   2_200),
    BA("EPE",  "El Paso Electric",                      "Western", -7, "ELP", True, False, (-106.4, 31.8),   2_200),
    BA("TEPC", "Tucson Electric Power",                 "Western", -7, "TUS", True, False, (-110.9, 32.1),   3_000),

    # ── Western — federal power administrations ──
    BA("WACM", "WAPA Rocky Mountain Region",            "Western", -7, "CYS", True, False, (-105.2, 42.0),   3_500),
    BA("WALC", "WAPA Desert Southwest Region",          "Western", -7, "FLG", True, False, (-113.0, 34.5),   2_500),
    BA("WAUW", "WAPA Upper Great Plains West",          "Western", -7, "BIS", True, False, (-101.0, 46.0),      700),

    # ── Western — California (non-CAISO) ──
    BA("BANC", "Balancing Authority of Northern California", "Western", -8, "SMF", True, False, (-121.5, 38.5), 3_300),
    BA("TIDC", "Turlock Irrigation District",           "Western", -8, "MOD", True, False, (-120.8, 37.5),      620),
    BA("IID",  "Imperial Irrigation District",          "Western", -8, "IPL", True, False, (-115.5, 33.0),    1_000),

    # ── Western — small PNW public utility districts ──
    BA("DOPD", "Douglas County PUD",                    "Western", -8, "EPH", True, False, (-120.2, 47.8),      200),
    BA("CHPD", "Chelan County PUD",                     "Western", -8, "EAT", True, False, (-120.3, 47.4),      350),
    BA("GCPD", "Grant County PUD",                      "Western", -8, "MWH", True, False, (-119.3, 47.1),      550),

    # ── Federal power marketing admins (Eastern) ──
    BA("SPA",  "Southwestern Power Administration",     "Eastern", -6, "LIT", True, False, (-92.3, 34.8),     1_500),

    # ── Generator- or transmission-only BAs (no demand series) ──
    BA("SEPA", "Southeastern Power Administration",     "Eastern", -5, "AGS", False, False, (-82.0, 33.4), None),
    BA("YAD",  "Alcoa Power Generating - Yadkin",       "Eastern", -5, "INT", False, False, (-80.2, 35.7), None),
    BA("GLHB", "GridLiance",                            "Eastern", -6, "MSY", False, False, (-90.3, 30.0), None),
    BA("NSB",  "New Smyrna Beach",                      "Eastern", -5, "DAB", False, False, (-80.9, 29.0), None),
    BA("AEC",  "PowerSouth Energy Cooperative",         "Eastern", -6, "MGM", False, False, (-86.6, 31.9), None),
    BA("EEI",  "Electric Energy, Inc.",                 "Eastern", -6, "PAH", False, False, (-89.0, 37.1), None),
    BA("SIKE", "Sikeston Board of Municipal Utilities", "Eastern", -6, "CGI", False, False, (-89.6, 36.9), None),
    BA("GWA",  "NaturEner Power Watch (MT)",            "Western", -7, "GTF", False, False, (-111.3, 48.5), None),
    BA("WWA",  "NaturEner Wind Watch (MT)",             "Western", -7, "GTF", False, False, (-111.3, 48.5), None),
    BA("HGMA", "Harquahala Generating",                 "Western", -7, "BXK", False, False, (-113.1, 33.5), None),
    BA("GRIF", "Griffith Energy",                       "Western", -7, "IFP", False, False, (-114.6, 35.1), None),
    BA("GRID", "Gridforce Energy Management",           "Western", -7, "PHX", False, False, (-112.0, 33.4), None),
    BA("DEAA", "Arlington Valley (AZ)",                 "Western", -7, "PHX", False, False, (-113.0, 33.2), None),
    BA("AVRN", "Avangrid Renewables",                   "Western", -8, "PDT", False, False, (-119.0, 45.7), None),
)


# Index by code for O(1) lookup.
REGISTRY: dict[str, BA] = {b.code: b for b in _REGISTRY}


def all_codes() -> list[str]:
    """Every BA code EIA-930 publishes (67 as of 2026)."""
    return [b.code for b in _REGISTRY]


def demand_codes() -> list[str]:
    """BAs that publish a demand (load) series — the forecastable ones."""
    return [b.code for b in _REGISTRY if b.has_demand]


def rto_codes() -> list[str]:
    """The 7 organized-market RTO/ISOs (the original v1 footprint)."""
    return [b.code for b in _REGISTRY if b.is_rto]


def get(code: str) -> BA:
    """Look up a BA by code. Raises KeyError if unknown."""
    try:
        return REGISTRY[code]
    except KeyError:
        raise KeyError(f"Unknown BA code '{code}'. Known codes: {sorted(REGISTRY)}") from None


def stations() -> dict[str, str]:
    """{ba_code: asos_station} for every BA with a weather-station assignment."""
    return {b.code: b.station for b in _REGISTRY if b.station is not None}


def filter_codes(
    codes: Iterable[str] | None = None,
    *,
    has_demand: bool | None = None,
    interconnect: str | None = None,
) -> list[str]:
    """Filter the registry. `codes=None` means 'all'."""
    src = codes if codes is not None else all_codes()
    out = []
    for c in src:
        b = REGISTRY.get(c)
        if b is None:
            continue
        if has_demand is not None and b.has_demand != has_demand:
            continue
        if interconnect is not None and b.interconnect != interconnect:
            continue
        out.append(c)
    return out
