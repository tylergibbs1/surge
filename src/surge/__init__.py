"""Surge: open forecasts and simulations for the US power grid."""

from surge._version import __version__
from surge.scrapers.eia import load as _eia_load

__all__ = ["__version__", "load"]


def load(ba: str, start: str, end: str):
    """Top-level convenience: hourly BA load via EIA-930."""
    return _eia_load(ba=ba, start=start, end=end)
