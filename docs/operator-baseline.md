# The operator baseline

EIA-930 publishes each BA's own day-ahead demand forecast as `type=DF`. It is
the single most important comparison Surge makes, because "how does this
compare to what the operator already publishes" is the question every reader
actually has. It is also free and already inside the data we ingest.

## PJM and CISO are labelled against a different hour

For two of the seven RTOs, `DF` and `D` for the same period string describe
different hours. Mean absolute percentage error of the operator forecast
against realized demand, comparing `DF(t)` with `D(t)` versus `D(t+1)`:

| BA | Window | MAPE, as published | MAPE, shifted +1h | Change |
|---|---|---:|---:|---:|
| PJM | Jan 2024 | 3.27% | 2.48% | −24.3% |
| PJM | Jan 2025 | 3.09% | 2.45% | −20.7% |
| PJM | Aug 2025 | 3.85% | 2.66% | −31.0% |
| CISO | Jan 2024 | 5.91% | 4.13% | −30.1% |
| CISO | Jan 2025 | 7.84% | 5.92% | −24.5% |
| CISO | Aug 2025 | 7.45% | 6.64% | −10.8% |
| ERCO | Jan 2024 | 3.46% | 4.29% | **+24.0%** |
| ERCO | Jan 2025 | 3.12% | 3.89% | **+24.7%** |
| ERCO | Aug 2025 | 2.16% | 3.69% | **+71.2%** |

The effect is stable across seasons and years, and it reverses on ERCO — so
the correction is per-BA and must never be applied globally. MISO, NYIS, ISNE
and SWPP all align as published.

**Uncorrected, an operator baseline overstates PJM and CISO forecast error by
roughly a quarter.** That error flatters Surge, which is precisely the
direction that must not go unnoticed: it would have produced a headline
comparison that was wrong in our own favour.

This is an empirical finding about label conventions. It is not a claim about
why the two operators differ, and the cause has not been established.

## What the ingest does

`surge.scrapers.eia.forecast` applies the per-BA offset and stores three things
together: `ts_utc`, the valid hour the forecast describes after correction;
`published_ts_utc`, the period exactly as EIA labelled it; and
`hour_offset_applied`. A silent correction is indistinguishable from a data
error, so the correction travels with the data.

Raw responses are archived by `surge.vintage` before any reshaping, so the
alignment can be re-derived from what EIA actually served.

## The offset is a measurement, not a constant

CISO has been observed switching between hour-start and hour-end conventions
between years, so an offset validated on one window can be wrong on another.
`DF_HOUR_OFFSET` is therefore scoped: it is validated for 2024-2025 only, and
`DF_OFFSET_VALIDATED_WINDOW` records that. Anyone scoring a different window
must re-measure with `surge.scrapers.eia.measure_df_hour_offset`, which returns
every candidate's error so the sharpness of the minimum is visible. A shallow
minimum means the convention is ambiguous in that window and the comparison
should not be published.

The measured offset and the applied offset share one sign convention -- hours to
add to the published timestamp -- and a test pins them together. A measurement
that read backwards from the correction it validates would be worse than no
measurement at all.

## `DF` is not comparable across balancing authorities

EIA's form instructions ask each BA to report "the day-ahead demand forecast
generated in the normal course of business", with **no specified issuance hour
and no specified lead time**, and EIA never imputes or adjusts the column. ISO-NE
publishes a same-morning product for the next six days; another BA's `DF` may be
made at a different hour with a different information set.

So `DF` is a valid benchmark for one BA against itself over time, and a trap for
one BA against another. A scoreboard must never present a cross-BA operator
column as a single comparable metric, and any Surge-versus-operator comparison
must state the operator's assumed issuance time or mark itself indicative-only.

## Before publishing any comparison

- Declare the operator's issuance time and information cutoff, or mark the
  comparison indicative-only. `DF` is day-ahead, but EIA does not document the
  exact issuance time, and a comparison against a forecast made with a
  different information set is contestable.
- Re-measure the offsets when extending beyond these seven RTOs. The registry
  covers only what has been measured.
- Report the operator and Surge on identical origins and identical scoring, and
  state the ground-truth vintage as part of the metric identity.
