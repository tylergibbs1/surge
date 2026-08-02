"""Forecast weather may enter a backtest only with its issue time attached.

Surge published an accuracy claim built on realized future temperature once
already. The retraction is in `docs/accuracy-restatement.md`. The lesson was not
"be careful with weather" -- it was that nothing in the code made the mistake
impossible, so a reviewer had to notice it.

This module is that missing rail. A forecast-weather observation is only usable
at a forecast origin when the observation was published before that origin, and
the type system refuses to represent one that cannot answer the question.

The trap this exists to stop is specific and live. Several archives are named as
though they retain issue time and do not:

- Open-Meteo's "Historical Forecast API" stitches the first hours of successive
  runs into a continuous series. That is an analysis. There is no issue time.
- Open-Meteo's Historical Weather API, ERA5, NSRDB, GOES-derived irradiance and
  SURFRAD are observations or reanalysis.
- NYISO's P-70A behind-the-meter series is an estimated actual. P-70B is its
  honest, forecast-vintage sibling.

Every one of those would score beautifully and mean nothing. Use
``ForecastVintage`` for anything that will inform a published number, and keep
observational sources in the diagnostic lane where they belong.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np


class VintageError(ValueError):
    """A weather value cannot be shown to have existed before its origin."""


#: Sources that cannot answer "what was known at time T". Naming them here means
#: a future reader meets the trap before they meet the bug.
OBSERVATIONAL_SOURCES = frozenset(
    {
        "open-meteo-historical-forecast",  # stitched runs; not a forecast
        "open-meteo-historical-weather",  # ERA5 reanalysis
        "era5",
        "nsrdb",
        "goes-ghi",
        "surfrad",
        "asos-observed",
        "nyiso-p70a",  # estimated actuals; P-70B is the forecast
        "iso-ne-btm-realized",
    }
)


@dataclass(frozen=True)
class ForecastVintage:
    """One forecast value, and the moment it became knowable.

    ``issued_at_utc`` is when the producing run was published, not when it was
    downloaded. A lead-time offset (for example "the value forecast about 24
    hours earlier") is acceptable only when it errs older, and the offset must
    be recorded in ``lead_convention`` so a reader can tell.
    """

    source: str
    variable: str
    valid_at_utc: datetime
    issued_at_utc: datetime
    value: float
    lead_convention: str

    def __post_init__(self) -> None:
        if self.source in OBSERVATIONAL_SOURCES:
            raise VintageError(
                f"{self.source} is observational or reanalysis, so it has no issue "
                "time and cannot inform a published forecast"
            )
        if self.issued_at_utc.tzinfo is None or self.valid_at_utc.tzinfo is None:
            raise VintageError("vintage timestamps must be timezone-aware")
        if self.issued_at_utc > self.valid_at_utc:
            raise VintageError(
                "a forecast cannot be issued after the hour it describes"
            )

    def known_at(self, origin_utc: datetime) -> bool:
        return self.issued_at_utc <= origin_utc


def usable_at_origin(
    vintages: Iterable[ForecastVintage], *, origin_utc: datetime
) -> list[ForecastVintage]:
    """Keep only values published at or before the origin.

    This is the whole guarantee. Call it on the way in, not on the way out.
    """
    if origin_utc.tzinfo is None:
        raise VintageError("origin_utc must be timezone-aware")
    return [vintage for vintage in vintages if vintage.known_at(origin_utc)]


def assert_no_leakage(
    vintages: Sequence[ForecastVintage], *, origin_utc: datetime
) -> None:
    """Fail loudly when a caller has already mixed in a future-issued value.

    Filtering silently is the wrong default for an evaluation path: a backtest
    that quietly drops leaked rows still reports a number, and the number is
    wrong in a way nobody sees.
    """
    leaked = [v for v in vintages if not v.known_at(origin_utc)]
    if leaked:
        first = leaked[0]
        raise VintageError(
            f"{len(leaked)} weather value(s) were issued after the forecast origin; "
            f"first is {first.variable} from {first.source} issued "
            f"{first.issued_at_utc.isoformat()} for an origin of "
            f"{origin_utc.isoformat()}"
        )


def clear_sky_index(
    forecast_ghi: np.ndarray, clear_sky_ghi: np.ndarray, *, floor: float = 20.0
) -> np.ndarray:
    """Cloud signal as a share of the clear-sky maximum.

    Raw irradiance carries a seasonal and diurnal shape the load model already
    knows from its own history. Dividing it out leaves the part it does not
    know, which is the cloud. Hours below ``floor`` W/m^2 are night or near it,
    where the ratio is noise, so they return 1.0 rather than a large quotient.
    """
    forecast = np.asarray(forecast_ghi, dtype=np.float64)
    clear = np.asarray(clear_sky_ghi, dtype=np.float64)
    if forecast.shape != clear.shape:
        raise ValueError("forecast and clear-sky arrays must have the same shape")
    index = np.ones_like(forecast)
    lit = clear > floor
    index[lit] = np.clip(forecast[lit] / clear[lit], 0.0, 1.5)
    return index
