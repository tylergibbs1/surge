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

The operator column this project used was itself unreliable. EIA-930's `DF` is
not the forecast an operator runs on: the form instructions excuse respondents
from making `DF` consistent with the `D` beside it, and EIA warns the comparison
"is not very meaningful" for some BAs.

Checked against what the operators publish themselves, `DF` is faithful for
ERCOT and materially wrong for PJM and CAISO, always in the direction that
flatters Surge:

| RTO | `DF`-derived | Operator, published | Verdict |
|---|---:|---:|---|
| ERCO | 2.13 | **2.16** | consistent; the one clean comparison |
| PJM | 2.43 | **1.43** | `DF` is a worse product than PJM's own forecast |
| CISO | 5.50 | **2.04** | `DF` is broken for CAISO; see below |
| MISO | 2.78 | 1.6 (daily peak only) | different metric, not comparable |
| NYIS, ISNE, SWPP | — | not published | no comparison possible |

So the earlier "parity" framing was wrong, and wrong in our favour for the third
time. Restricted to the three RTOs where a comparable operator number exists:

| | PJM | ERCO | CISO | mean |
|---|---:|---:|---:|---:|
| Surge, zero-shot, calendar-only | 2.86 | 3.17 | 2.64 | **2.89** |
| Operator, published | 1.43 | 2.16 | 2.04 | **1.88** |

**Surge is about one percentage point behind the operators**, not at parity. The
gap is still understated, because Surge forecasts from a same-day 00:00 UTC
origin (1-24 h leads) while PJM issues at 18:00 D-1 and CAISO around 10:00 PT
D-1 (14-38 h leads).

The seven-RTO "operator mean of 3.03" is retired. It averaged three real
measurements, one broken column, and three numbers that measure nothing
published.

### CISO: the cause was not behind-the-meter solar

An earlier version of this document attributed CISO's `DF` error to
behind-the-meter PV. **That attribution was wrong, and wrong on the sign.** If
`DF` were gross load and `D` were net of BTM PV, `DF` would sit *above* `D` at
midday. The observed midday error is strongly negative, and it grows over time
while BTM PV was already large at the start:

| Year | Signed `DF` error, local midday | All-hours MAPE |
|---|---:|---:|
| 2020 | +0.7% | 2.32 |
| 2021 | −2.1% | 2.09 |
| 2022 | −4.3% | 3.56 |
| 2023 | −9.4% | 5.50 |
| 2024 | −16.0% | 6.98 |
| 2025 | −21.5% | 8.15 |

In 2020-21 the column agreed with CAISO's published ~2%. The divergence tracks
CAISO's battery fleet, which went from roughly nothing to about 13 GW over the
same window, and midday is when it charges. CAISO defines its own load as the
forecast component plus pumps and the charging side of storage, while EIA-930's
`D = NG - TI` still contains that charging.

The likely mechanism is therefore a storage-charging definitional wedge: `DF`
excludes storage charging and `D` does not. This is inference. Neither EIA nor
CAISO documents it, and it should be described as unexplained divergence rather
than asserted as fact.

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

## The blend, and what it changes

Two model families with different inductive biases landed within 0.04 points of
each other on this cohort: Chronos-2 at 3.09% and a LightGBM baseline at 3.13%.
That is the classic setup for a useful combination, and it is.

On the untouched second half of 2024, blend weight fitted only on the first half
(`artifacts/research/baseline-blend-2024.json`):

| | mean MAPE |
|---|---:|
| Seasonal-naive (t-24) | 4.789 |
| GBM baseline | 2.978 |
| Chronos-2 zero-shot | 2.878 |
| **Blend** | **2.717** |

Every RTO improves over its own best single model, by 1.3% to 6.6%, and the
optimal weight on Chronos sits between 0.34 and 0.64 everywhere -- neither model
dominates. The macro gain over the best single model is 4.9%, which is larger
than context length, the calendar-clock fix and the LoRA fine-tune produced
combined.

The operator, scored on those same hours, averages 2.855%. The blend is ahead of
that, but **the lead-time caveat still applies and still cuts against Surge**, so
this does not establish that Surge beats the operators. What it establishes is
that the blend beats both of Surge's own models, robustly and on data that did
not choose its weight.

## ISO-NE is not a bug, and the gap is mostly physics

ISNE is the worst RTO for both model families by a wide margin, which pointed at
data or regime rather than architecture. It is regime.

EIA-930's ISNE demand series is a **net load** series with several GW of
unobservable, cloud-driven behind-the-meter PV subtracted from a system whose
mean load is only about 13 GW. No other RTO in the fleet carries BTM PV that
large relative to its own size. ISO-NE counted 134 "duck curve days" in 2025, up
from 45 in 2022, and set a record-low net load of 5,318 MW in April 2025.

The error is concentrated exactly where that predicts: midday hours are roughly
29% of the sample but around 41% of ISNE's total absolute error, and collapsing
midday to ISNE's own overnight rate would close most of the gap to the fleet.
Retraining on recent data only, adding a time trend, or adding deterministic
clear-sky irradiance all fail to help -- the residual is day-to-day cloud cover,
which is not in the feature set at all and cannot be without an irradiance input.

Three consequences worth stating plainly:

- **4.86% for a calendar-only ISNE forecast is good, not broken.** An
  independent reference LightGBM built for this diagnosis scored 6.36% on the
  same task; both Surge models beat it by about 1.5 points.
- **ISO-NE's 2.77% is not a fair target.** It is produced by an ensemble fed by
  three weather vendors across 23 airports, three separate BTM-PV vendors, and a
  staffed desk. The gap is the value of weather data and people, not a modelling
  deficiency.
- **Roughly 1.5 of the 2.1-point gap is midday BTM-solar variance that is
  physically unobservable without an irradiance forecast.** That is a
  quantified statement of what the no-future-weather constraint costs.

ISNE also has the *highest* raw interval coverage of the seven RTOs. That is a
symptom of the same diagnosis rather than a separate finding: its error is a
broad, near-symmetric, high-variance weather term rather than rare fat tails, so
intervals sized for that variance cover well. The point forecast is weak there
and the uncertainty estimate is honest about it.

## What may be claimed today

- Surge is competitive with published open load-forecasting baselines.
- Blending the foundation model with the open GBM baseline beats either alone by
  4.9%, measured on data that did not choose the blend weight.
- Surge is **about one percentage point behind** the three operators that
  publish their own day-ahead accuracy, and the gap is understated because
  Surge's forecast lead time is shorter.
- No operator baseline may be derived from EIA-930 `DF` again.
- No locked-test accuracy claim exists for v0.2; that lane was consumed by a
  fail-closed error before any metric was produced.
- Interval coverage claims require the calibration lane to ship first.
