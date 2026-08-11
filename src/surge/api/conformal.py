"""Split-conformal widening of the served prediction interval.

The raw model quantiles under-cover. Offline on the 7 RTOs the nominal-80%
interval held the truth 75.0% of the time; adding a per-BA conformal delta moved
that to 81.25% with *zero* change to MASE or MAE, because the delta only pushes
the outer pair apart (p10 down, p90 up) and never touches the median. That is the
whole appeal: coverage is bought without spending any point accuracy.

One number per BA, in MW: the (1-alpha) empirical quantile of the calibration
conformity score max(q_lo - y, y - q_hi). Fitted offline on the validation split
(calendar 2024) by scripts/compute_conformal_deltas.py, which imports the same
`experiments.eval_c2._conformal_delta` that produced the measurement above.

The artifact lives inside the package rather than in models/ or data_snapshot/
because both are gitignored, and the Modal image only ships src/ and the data
snapshot — a table of 53 floats has to travel with the code to reach production.

A missing artifact, or a BA with no fitted delta, serves the raw quantiles and
logs. Inventing a delta would be worse than under-covering: the interval would
then advertise a width nothing ever measured.
"""
from __future__ import annotations

import json
import logging
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

DELTAS_PATH = os.environ.get(
    "SURGE_CONFORMAL_PATH",
    str(Path(__file__).resolve().parent / "conformal_deltas.json"),
)

_log = logging.getLogger(__name__)

# BAs already warned about, so /forecast (53 BAs, every 5 minutes) doesn't
# reprint the same line forever.
_warned: set[str] = set()


@lru_cache(maxsize=1)
def deltas() -> dict[str, float]:
    """Per-BA conformal delta in MW. Empty when the artifact is unusable.

    Cached: the file is a deploy-time constant, and forecast_ba() runs per BA
    per request.
    """
    path = Path(DELTAS_PATH)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        _log.warning("conformal: no delta artifact at %s; serving uncalibrated "
                     "quantiles", path)
        return {}
    except (OSError, ValueError) as e:
        _log.warning("conformal: cannot read %s (%s); serving uncalibrated "
                     "quantiles", path, type(e).__name__)
        return {}

    # Accept the script's envelope (deltas plus provenance) or a bare {BA: mw}
    # mapping, so a hand-written file still loads.
    table = raw.get("deltas_mw", raw) if isinstance(raw, dict) else None
    if not isinstance(table, dict):
        _log.warning("conformal: %s has no BA->MW mapping; serving uncalibrated "
                     "quantiles", path)
        return {}

    out: dict[str, float] = {}
    dropped: list[str] = []
    for ba, v in table.items():
        try:
            d = float(v)
        except (TypeError, ValueError):
            dropped.append(str(ba))
            continue
        # NaN/inf would silently poison every interval for that BA.
        if math.isfinite(d):
            out[str(ba).upper()] = d
        else:
            dropped.append(str(ba))
    if dropped:
        _log.warning("conformal: ignoring non-numeric deltas for %s",
                     ", ".join(sorted(dropped)))
    return out


def apply_delta(q: np.ndarray, ba: str, *, lo: int = 0, mid: int = 1,
                hi: int = 2) -> np.ndarray:
    """Return `q` with the outer quantile pair widened for `ba`.

    `q` is (..., n_quantiles) as the pipeline emits it; `lo`/`mid`/`hi` index the
    quantile axis. The median column is returned byte-for-byte unchanged.
    """
    d = deltas().get(ba.upper())
    if d is None:
        if ba.upper() not in _warned:
            _warned.add(ba.upper())
            _log.warning("conformal: no delta for %s; serving uncalibrated "
                         "quantiles", ba)
        return q

    out = q.copy()
    out[..., lo] -= d
    out[..., hi] += d
    # Clamp against the median so p10 <= median <= p90 holds regardless. Two ways
    # it can fail otherwise: a negative delta (calibration over-covered, so the
    # honest correction narrows) and the model's own quantile crossing, which
    # Chronos-2 can produce on flat stretches.
    out[..., lo] = np.minimum(out[..., lo], out[..., mid])
    out[..., hi] = np.maximum(out[..., hi], out[..., mid])
    return out
