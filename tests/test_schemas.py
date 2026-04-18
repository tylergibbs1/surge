from datetime import datetime, timezone

import polars as pl
import pytest

from surge import schemas


def test_enforce_reorders_and_casts() -> None:
    df = pl.DataFrame({
        "load_mw": [100, 200],
        "ba": ["PJM", "PJM"],
        "as_of": [datetime(2024, 1, 1, tzinfo=timezone.utc)] * 2,
        "source": ["eia-930"] * 2,
        "ts_utc": [datetime(2024, 1, 1, tzinfo=timezone.utc)] * 2,
    })
    out = schemas.enforce(df, schemas.LOAD)
    assert out.columns == list(schemas.LOAD)
    assert out.dtypes[2] == pl.Float64


def test_enforce_raises_on_missing_column() -> None:
    df = pl.DataFrame({"ts_utc": [], "ba": []})
    with pytest.raises(ValueError, match="missing columns"):
        schemas.enforce(df, schemas.LOAD)
