"""Runtime proof that a forecast setup does not peek at the future.

READ-ONLY during autonomous search. Any experiment loop that is scored on MASE
has an incentive to "win" by feeding realized future values back in as
covariates, which is exactly the bug this repo just finished removing. Policy
labels are not enough — a label can be changed. So this module *measures*
causality instead of trusting it.

The test: perturb every covariate at and after the forecast origin, rebuild the
future covariates, and require byte-identical output. Anything that reads a
post-origin actual will move, and gets rejected.

Calendar features are deliberately exempt: they are pure functions of the
timestamp, so they are legitimately knowable in advance.
"""
from __future__ import annotations

import copy

import numpy as np

# Deterministic functions of the timestamp — knowable arbitrarily far ahead.
CALENDAR_EXEMPT = frozenset({
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday",
})


class LeakageError(AssertionError):
    """Raised when a setup reads values at or after the forecast origin."""


def assert_causal(bd, origin: int, horizon: int, *, bump: float = 1e4) -> None:
    """Raise LeakageError if `bd.future_at(origin, horizon)` sees the future.

    `bd` is a BAData. Called with a real origin from the eval loop, so this
    checks the actual configuration being scored, not a synthetic stand-in.
    """
    baseline = bd.future_at(origin, horizon)

    probe = copy.deepcopy(bd)
    touched = []
    for k, v in probe.covariates.items():
        if k in CALENDAR_EXEMPT:
            continue
        arr = np.asarray(v)
        if arr.dtype.kind not in "fiu" or arr.size <= origin:
            continue
        # Perturb everything the forecaster must not be able to see.
        arr = arr.astype(np.float64, copy=True)
        arr[origin:] += bump
        probe.covariates[k] = arr.astype(v.dtype, copy=False)
        touched.append(k)

    perturbed = probe.future_at(origin, horizon)

    leaked = []
    for k, base in baseline.items():
        if k in CALENDAR_EXEMPT:
            continue
        other = perturbed.get(k)
        if other is None:
            continue
        if not np.array_equal(np.asarray(base), np.asarray(other)):
            leaked.append(k)

    if leaked:
        raise LeakageError(
            f"{getattr(bd, 'ba', '?')}: future covariates {sorted(leaked)} changed when "
            f"post-origin actuals were perturbed at origin={origin}. These are being "
            f"read from realized future data, which is leakage. Future covariates must "
            f"depend only on data at indices < origin (or on the timestamp alone). "
            f"Perturbed: {sorted(touched)}."
        )

    # A setup that declares a future key but silently drops it is also suspect,
    # because the eval would then differ from what the checkpoint was trained on.
    declared = set(getattr(bd, "future_keys", []))
    missing = declared - set(baseline)
    if missing:
        raise LeakageError(
            f"{getattr(bd, 'ba', '?')}: declared future_keys {sorted(missing)} were not "
            f"produced by future_at(); the eval config is inconsistent."
        )


def audit(bas: dict, horizon: int = 24, *, per_ba_origins: int = 3) -> dict:
    """Check several origins per BA. Returns a summary; raises on any leak."""
    checked = 0
    for ba, bd in bas.items():
        lo, hi = bd.val_end, bd.test_end
        span = hi - horizon - lo
        if span <= 0:
            lo, hi = bd.train_end, bd.val_end
            span = hi - horizon - lo
        if span <= 0:
            continue
        for i in range(per_ba_origins):
            origin = lo + (span * i) // max(per_ba_origins, 1)
            assert_causal(bd, int(origin), horizon)
            checked += 1
    return {"bas": len(bas), "origins_checked": checked, "causal": True}
