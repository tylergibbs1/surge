# Accuracy restatement

The v0.1 README claimed Surge "beats EIA's day-ahead demand forecast on 6 of 7
major RTOs", with macro MAE 40% below the operators'. That claim was wrong, and
it was wrong for **two independent reasons that both flattered Surge**. The
oracle-covariate half was withdrawn during v0.2. This records the other half,
and replaces the withdrawal with measured numbers.

## Cause 1: oracle covariates (already withdrawn)

The v1–v3 benchmark fed realized future ASOS temperature into the backtest, and
generation-enabled configurations also used realized wind and solar. Production
has none of these. Those results are an upper bound, not a production forecast.

## Cause 2: the operator baseline was misaligned by an hour

`scripts/compare_eia_df.py` joined EIA `D` and `DF` on the period string with no
hour correction. For PJM and CISO those two series are labelled against
different hour conventions, so the operator was scored against the wrong hour.

Operator MAE on the published 2025 window (MW):

| BA | Published | Reproduced, uncorrected | Corrected | Overstatement |
|---|---:|---:|---:|---:|
| PJM | 3297 | 3303 | **2368** | **+39.5%** |
| CISO | 2098 | 2095 | **1836** | **+14.1%** |
| ERCO | 1366 | 1366 | 1366 | 0% |
| MISO | 1786 | 1793 | 1793 | 0% |
| NYIS | 560 | 561 | 561 | 0% |
| ISNE | 306 | 306 | 306 | 0% |
| SWPP | 2590 | 2600 | 2600 | 0% |

The uncorrected column reproduces the published figures to within rounding,
which confirms the published table came from the uncorrected join. Correcting
it, "PJM: 1.70× better than the operator" becomes 1.22×, and macro operator MAE
falls from 1715 to 1547 MW. See `docs/operator-baseline.md`.

## The honest comparison

2024 validation lane, seven RTOs, identical target hours, calendar-only future
covariates, operator hour-corrected. MAPE:

| BA | Surge (zero-shot) | Operator | Winner |
|---|---:|---:|---|
| MISO | **2.18** | 2.78 | Surge |
| CISO | **2.64** | 5.50 | Surge* |
| SWPP | 2.73 | **2.33** | operator |
| PJM | 2.86 | **2.43** | operator |
| NYIS | **3.15** | 3.27 | Surge |
| ERCO | 3.17 | **2.13** | operator |
| ISNE | 4.89 | **2.77** | operator |
| **mean** | **3.09** | **3.03** | parity |

Seasonal-naive(168h) scores 8.42% on the same hours.

Three caveats, all of which cut against Surge:

- **Lead time is not equalized.** Surge forecasts from a 00:00 UTC origin for
  that same day, so leads are 1–24 h. EIA's `DF` is a genuine day-ahead
  submission with leads closer to 24–48 h. Surge is being graded on an easier
  task, so the true gap is larger than shown.
- **CISO is not a real win.** Its `DF` field carries a large midday
  behind-the-meter-solar artifact, so Surge is beating a broken column.
- Data is `retrospective_final`, not the vintage available at forecast time.

Against *published* load-only deep-learning baselines at the same horizon
(arXiv:2602.21415: MISO 2.33, ERCOT 2.85, PJM 2.97, CAISO 3.12, SPP 3.43,
NYISO 4.25, ISO-NE 5.86), zero-shot Surge is better on six of seven. Surge is
competitive with the published literature. It is not competitive with operators,
who have weather forecasts Surge does not use.

## Intervals are the larger problem

`surge-fm-v3`'s own published evaluation reports mean 80% interval coverage of
**0.725** across 53 BAs — on the favourable oracle backtest. ERCO is 0.660 and
MISO 0.697. A band labelled 80% was wrong about a fifth of the time it claimed
to be right.

Measured head-to-head on 2024, calendar-only, identical origins:

| Model | Mean MAPE | Mean 80% coverage |
|---|---:|---:|
| chronos-2 zero-shot (v0.2 default) | 3.092 | **0.759** |
| `surge-fm-v3` | **2.948** | 0.719 |

The fine-tune is 4.7% better on MAPE and worse on coverage, and its MAPE
advantage is optimistic because 2024 falls inside its training window. It buys
point accuracy by degrading interval honesty. This independently supports the
v0.2 decision to serve the pinned upstream base rather than the fine-tune.

Calibration closes the interval gap: canonical CQR holds every RTO within 1.6
points of nominal across 2021, 2022, 2023 and the 2024 reporting half
(`docs/adaptive-conformal-experiment.md`). It is not yet wired into serving.

## What may be claimed today

- Surge is competitive with published open load-forecasting baselines.
- Surge is **not** currently better than RTO operators' own day-ahead forecasts.
- No locked-test accuracy claim exists for v0.2; that lane was consumed by a
  fail-closed error before any metric was produced.
- Interval coverage claims require the calibration lane to ship first.
