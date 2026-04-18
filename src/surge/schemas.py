"""Canonical column conventions for every Surge dataset.

All timestamps UTC. Energy in MW / MWh. Price in $/MWh.
Every table carries `source` and `as_of` for lineage.
"""

from __future__ import annotations

import polars as pl

LOAD = {
    "ts_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "ba": pl.Utf8,
    "load_mw": pl.Float64,
    "source": pl.Utf8,
    "as_of": pl.Datetime(time_unit="us", time_zone="UTC"),
}

LMP = {
    "ts_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "iso": pl.Utf8,
    "node": pl.Utf8,
    "market": pl.Utf8,  # "DA" or "RT"
    "lmp_usd_per_mwh": pl.Float64,
    "energy_usd_per_mwh": pl.Float64,
    "congestion_usd_per_mwh": pl.Float64,
    "losses_usd_per_mwh": pl.Float64,
    "source": pl.Utf8,
    "as_of": pl.Datetime(time_unit="us", time_zone="UTC"),
}

GEN_BY_FUEL = {
    "ts_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    "ba": pl.Utf8,
    "fuel": pl.Utf8,
    "gen_mw": pl.Float64,
    "source": pl.Utf8,
    "as_of": pl.Datetime(time_unit="us", time_zone="UTC"),
}


def enforce(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Reorder and cast a frame to match a canonical schema. Missing cols raise."""
    missing = set(schema) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df.select([pl.col(c).cast(t) for c, t in schema.items()])
