"""Tests for split-conformal interval calibration (src/surge/api/conformal.py).

No GPU and no model: apply_delta() is pure array arithmetic over a quantile
block shaped exactly like the one forecast_ba() gets back from the pipeline,
(H, 3) at levels [0.1, 0.5, 0.9].
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from surge.api import conformal

LO, MID, HI = 0, 1, 2


@pytest.fixture(autouse=True)
def _fresh_cache():
    """deltas() caches the artifact for the process; drop it around every test.

    Also clears the once-per-BA warning memo so caplog assertions don't depend
    on test order.
    """
    conformal.deltas.cache_clear()
    conformal._warned.clear()
    yield
    conformal.deltas.cache_clear()
    conformal._warned.clear()


def _artifact(tmp_path, monkeypatch, payload) -> None:
    """Point the module at a temp artifact. DELTAS_PATH is resolved from env at
    import time (like forecaster.MODEL_PATH), so tests set the attribute."""
    p = tmp_path / "conformal_deltas.json"
    p.write_text(json.dumps(payload))
    monkeypatch.setattr(conformal, "DELTAS_PATH", str(p))


def _quants() -> np.ndarray:
    # Three hours, p10 < median < p90, deliberately asymmetric.
    return np.array([
        [90.0, 100.0, 115.0],
        [80.0, 95.0, 101.0],
        [70.0, 71.0, 130.0],
    ], dtype=np.float32)


def test_delta_widens_outer_pair_only(tmp_path, monkeypatch) -> None:
    _artifact(tmp_path, monkeypatch, {"deltas_mw": {"PJM": 10.0, "CISO": 2.5}})
    q = _quants()
    out = conformal.apply_delta(q, "PJM", lo=LO, mid=MID, hi=HI)

    assert np.allclose(out[:, LO], q[:, LO] - 10.0)
    assert np.allclose(out[:, HI], q[:, HI] + 10.0)
    # The point forecast is the product being sold; it must be bit-identical.
    assert np.array_equal(out[:, MID], q[:, MID])
    # And the input must not have been mutated in place.
    assert np.array_equal(q, _quants())


def test_lookup_is_case_insensitive_and_accepts_bare_mapping(tmp_path, monkeypatch) -> None:
    _artifact(tmp_path, monkeypatch, {"PJM": 10.0})
    out = conformal.apply_delta(_quants(), "pjm")
    assert np.allclose(out[:, HI] - _quants()[:, HI], 10.0)


def test_missing_ba_serves_uncalibrated(tmp_path, monkeypatch, caplog) -> None:
    _artifact(tmp_path, monkeypatch, {"deltas_mw": {"PJM": 10.0}})
    q = _quants()
    with caplog.at_level("WARNING"):
        out = conformal.apply_delta(q, "ERCO")
    assert np.array_equal(out, q)
    assert "no delta for ERCO" in caplog.text


def test_missing_artifact_serves_uncalibrated(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(conformal, "DELTAS_PATH", str(tmp_path / "absent.json"))
    q = _quants()
    with caplog.at_level("WARNING"):
        out = conformal.apply_delta(q, "PJM")
    assert np.array_equal(out, q)
    assert "no delta artifact" in caplog.text


def test_unparseable_artifact_serves_uncalibrated(tmp_path, monkeypatch, caplog) -> None:
    p = tmp_path / "conformal_deltas.json"
    p.write_text("{not json")
    monkeypatch.setattr(conformal, "DELTAS_PATH", str(p))
    with caplog.at_level("WARNING"):
        assert conformal.deltas() == {}
    assert "cannot read" in caplog.text


def test_non_numeric_delta_is_dropped(tmp_path, monkeypatch) -> None:
    _artifact(tmp_path, monkeypatch,
              {"deltas_mw": {"PJM": "nope", "CISO": None, "MISO": float("nan"),
                             "NYIS": 5.0}})
    assert conformal.deltas() == {"NYIS": 5.0}


def test_ordering_holds_for_crossed_input(tmp_path, monkeypatch) -> None:
    """A negative delta narrows, and the model can cross its own quantiles.
    Either way consumers must never see p10 > median or p90 < median."""
    _artifact(tmp_path, monkeypatch, {"deltas_mw": {"PJM": -40.0}})
    q = np.array([
        [90.0, 100.0, 115.0],
        [105.0, 100.0, 95.0],     # fully crossed straight from the model
    ], dtype=np.float32)
    out = conformal.apply_delta(q, "PJM")

    assert np.all(out[:, LO] <= out[:, MID])
    assert np.all(out[:, MID] <= out[:, HI])
    assert np.array_equal(out[:, MID], q[:, MID])


def test_ordering_holds_for_every_positive_delta(tmp_path, monkeypatch) -> None:
    _artifact(tmp_path, monkeypatch, {"deltas_mw": {"PJM": 426.6}})
    rng = np.random.default_rng(0)
    base = rng.normal(30_000, 4_000, size=(24, 1))
    q = np.sort(base + rng.normal(0, 800, size=(24, 3)), axis=1).astype(np.float32)
    out = conformal.apply_delta(q, "PJM")

    assert np.all(out[:, LO] <= out[:, MID])
    assert np.all(out[:, MID] <= out[:, HI])
    # Widening is symmetric in MW, so the interval grows by exactly 2*delta.
    assert np.allclose((out[:, HI] - out[:, LO]) - (q[:, HI] - q[:, LO]), 2 * 426.6,
                       atol=0.05)
