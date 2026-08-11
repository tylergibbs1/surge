"""Robust outlier rejection for load series.

The original rule was `load_mw > 200_000 -> null`. It has two faults.

It is absolute, so it cannot tell a 70 GW spike in a 1.8 GW balancing authority
from a normal afternoon in PJM. That inflated the seasonal-naive MASE
denominator for BANC, SPA, LGEE, SEC and TEPC, which flattered their scores.

It is also one-sided. EIA-930 carries large negative values (LGEE reaches
-16,773,374 MW against a median of 3,992 MW) and the old rule passed every one
of them through. Those negatives, not high spikes, are what corrupted LGEE.

The replacement is a two-sided robust rule, scaled to each series:

    reject when |x - median| > MAD_K * 1.4826 * MAD(x)

1.4826 * MAD estimates the standard deviation of a normal distribution, so
MAD_K is roughly a robust z-score. Load has wide but legitimate swings -- PJM's
daily range reaches about 8 robust deviations from its median -- so the
threshold is deliberately loose at 20. It removes values that are impossible
rather than values that are merely extreme.
"""
from __future__ import annotations

import polars as pl

# Robust z threshold. Chosen well above the widest legitimate daily swing
# observed across the 53 BAs (about 8) so that real peaks and troughs survive.
MAD_K = 20.0

# Absolute ceiling kept as a second line of defense. 2,147,483,647 (2^31-1)
# appears in the feed as an overflow sentinel and must never reach the model.
ABS_CEILING_MW = 500_000.0


def robust_load_filter(col: str = "load_mw") -> pl.Expr:
    """Return an expression that nulls impossible values in `col`.

    Per-series: apply after filtering to one BA, or inside `over("ba")`.
    """
    x = pl.col(col)
    median = x.median()
    mad = (x - median).abs().median()
    # A degenerate MAD (constant series) would reject everything, so fall back
    # to the absolute ceiling alone in that case.
    scale = pl.when(mad > 0).then(1.4826 * mad).otherwise(None)
    too_far = ((x - median).abs() > MAD_K * scale) & scale.is_not_null()
    return (
        pl.when(too_far | (x.abs() > ABS_CEILING_MW))
        .then(None)
        .otherwise(x)
        .alias(col)
    )


def clean_load(df: pl.DataFrame, col: str = "load_mw",
               by: str | None = None) -> pl.DataFrame:
    """Null impossible load values, per series when `by` is given."""
    expr = robust_load_filter(col)
    return df.with_columns(expr.over(by) if by else expr)
