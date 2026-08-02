# Methodology

This document describes the evidence behind Surge Grid v0.2 and the minimum
information required for future claims. It is deliberately explicit about the
difference between a useful research experiment and a deployable forecast.

## Forecasting task

The broad research task is probabilistic balancing-authority demand forecasting:

- target: hourly EIA-930 demand in MW;
- entities: the 53 balancing authorities in `surge.bas` with `has_demand=True`;
- default context: 2,048 hourly observations;
- default horizon: 24 hours for evaluation, with the API accepting 1-168;
- quantiles: p10, p50, and p90;
- timestamps: UTC.

The v0.2 trusted publication and live-forward evaluation subset is exactly
`PJM`, `CISO`, `ERCO`, `MISO`, `NYIS`, `ISNE`, and `SWPP`. A run is complete
only when all seven compatible per-BA issuances exist and the Python ledger
writes one immutable `forecast_runs` marker. Public run listings and the
scoreboard expose only those marked complete runs; direct issuance detail
remains available by ID for audit. The Vercel `current` pointer is a separate
atomic boundary and advances only after the complete run validates. Forecasts
for the other 46 demand-reporting BAs remain a legacy/best-effort exploration
surface and are not counted in the v0.2 release gate or scoreboard.

The v0.2 serving default is the upstream `amazon/chronos-2` checkpoint at
immutable revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, used with the
shared no-peeking feature contract. The legacy `surge-fm-v3` checkpoint was
fine-tuned across multiple balancing authorities under the archived feature
and training path; it is retained for historical benchmark study and is not a
v0.2 serving model. Every published result must identify the exact model
repository and immutable revision, not only a friendly name.

## Data and availability

| Input | Source | Stored availability field | Forecast-time treatment |
|---|---|---|---|
| Demand | EIA-930 | ingestion `as_of` | values at or before the origin only |
| Temperature | ASOS / Iowa Environmental Mesonet | not complete in the historical snapshot | archived issue-time weather is required for replay |
| Wind and solar | EIA generation mix | ingestion `as_of` where present | only an archived forecast may be future-known |
| Calendar | deterministic UTC timestamp transforms | inherent | known in advance |
| Operator day-ahead forecast | EIA Grid Monitor `DF` | historical endpoint does not preserve every submission vintage | comparison requires a captured issuance vintage |

Source terms and licenses remain authoritative. A release data manifest must
record source URLs, retrieval times, row counts, time ranges, BA coverage, and
file hashes. It must not imply that third-party source availability or terms
are guaranteed by this project.

### Cleaning and missingness in the archived experiment

- Demand values above 200,000 MW are treated as invalid.
- Missing demand and temperature are forward-filled; leading gaps are
  backfilled from the first available value.
- A completely missing temperature series becomes zero-filled.
- When generation features are enabled, wind and solar are forward-filled.
- The store is append-only and overlapping EIA pulls can contain revisions;
  evaluation must select the last record available by the declared origin,
  rather than the latest record visible today.

These fallbacks keep training code running, but they can hide weak coverage.
They describe the original experiment at commit `36ceaff`, not the v0.2
feature loader. Future result artifacts must report missingness and fallback
rates per BA.

### v0.2 feature and missingness contract

The shared `load-v2-core` contract used by current training, evaluation, and
serving is stricter:

- observed temperature is required as a historical covariate;
- observed wind and solar are optional historical covariates only;
- only deterministic calendar fields may appear as future covariates;
- hourly gaps remain explicit missing values instead of being backward-filled
  or replaced with zero, and the leading prefix is trimmed to the first jointly
  usable timestamp;
- live reads select source records available by a frozen cutoff and reject an
  issuance when either the load or observed-temperature valid-time watermark is
  more than 12 hours behind issuance.

Historical experiments currently use `retrospective_final`, which can select
source revisions published after an old forecast origin. Removing future
observations fixes one leakage class, but it does not turn those experiments
into vintage replay.

## Archived split and metrics

The current experiment code defines:

- training: timestamps before 2024-01-01;
- validation: calendar year 2024;
- test: timestamps from 2025-01-01 onward in the local snapshot;
- rolling origins: 24-hour horizon with a 24-hour step;
- MASE denominator: each BA's training-set mean absolute 24-hour seasonal
  difference;
- aggregation: macro average across BAs.

The original Chronos-2 evaluator at commit `36ceaff` calculated MAE, RMSE, and
MASE from the model mean returned by the pipeline. The v0.2 evaluator instead
scores p50, matching the public point forecast, while retaining the predictive
mean as a separately named field. Reports must identify both the code revision
and whether a point metric uses mean or median.

The v3 values quoted in the README are historical outputs from this code path.
The repository did not originally publish the exact input snapshot, archived
configuration, environment lock, and result hash together. They therefore
cannot yet be independently reproduced byte-for-byte.

## Model-selection and overfitting gate

`experiments.finetune_c2` is a seven-RTO promotion workflow, not a generic
leaderboard search. Its data query stops before `2025-01-01T00:00:00Z`, and a
second validation-only view physically truncates every array at the same
boundary. The locked 2025 test partition is therefore not materialized for
training, checkpoint selection, baseline comparison, or promotion.
Promotion is also lineage-locked to `amazon/chronos-2` at commit
`29ec3766d36d6f73f0696f85560a422f50e8498c`. Legacy `surge-fm-v3`, oracle,
forked, and unknown lineages remain research ablations and cannot receive a
v0.2 `best/` directory or promotion marker.

Checkpoint selection uses 90 shared, fully observed daily origins from the
earlier portion of 2024 validation. Surge reserves the next/latest 90 shared
origins for promotion, so checkpoint loss and promotion metrics never score the
same days. After Chronos finishes, Surge evaluates the **returned best-loaded
candidate** on those reserved origins and on the latest 90 shared origins from
the pre-2024 training partition. The availability-only filter runs before
cohort selection and before model inference; skipped-window counts and exact
origin hashes are retained. Surge also evaluates the immutable upstream base
revision on the identical promotion origins. The audit records:

- macro and per-RTO MASE and MASE-scaled WIS for train, validation, and the
  frozen upstream baseline;
- absolute and ratio generalization gaps;
- validation dispersion, worst-RTO performance, and worst-RTO regression;
- a 2,000-resample paired seven-origin moving-block bootstrap interval for the
  candidate/base macro-MASE ratio;
- every captured training-loss, evaluation-loss, learning-rate, and checkpoint
  step, including the reported best checkpoint and any late loss rebound; and
- exact Chronos, Transformers, Torch, NumPy, Polars, Holidays, and PyArrow
  versions.

Policy `surge-v0.2-overfit-gate-v1` creates `best/` and
`surge-promotion.json` only when every gate passes:

| Gate | Threshold |
|---|---:|
| Shared diagnostic windows per RTO in train, validation, and baseline | exactly 90 |
| Validation macro MASE | at most 1.00 |
| Validation/train macro MASE ratio | at most 1.75 |
| Validation/train macro scaled-WIS ratio | at most 1.75 |
| Worst validation RTO MASE | at most 1.25 |
| Worst per-RTO validation/train MASE ratio | at most 2.25 |
| Validation RTO MASE coefficient of variation | at most 0.40 |
| Candidate/base macro MASE and scaled-WIS ratios | at most 1.00 each |
| Worst per-RTO candidate/base MASE ratio | at most 1.10 |
| Paired-bootstrap candidate/base MASE-ratio 95% upper bound | at most 1.05 |
| Training and evaluation trace | all planned steps, at least two train and two evaluation logs |
| Final/best evaluation-loss ratio | at most 1.25 |
| Reported best checkpoint | must match the observed minimum and a saved step |

A failed or incomplete diagnostic writes `surge-overfit-audit.json`, leaves the
raw model under `candidate-unpromoted/`, omits the promotion marker, and exits
non-zero. Thresholds are conservative release governance chosen before the
locked test; they are not confidence bounds or proof against distribution
shift. Changing them requires a versioned code/policy change, not a command-line
override.

The promotion marker hashes both audit documents, and the training manifest
records the size and SHA-256 of every file below `best/`. Before the one allowed
locked-test run, Surge verifies that complete inventory and binds the run to
the manifest's base-model, code, data-snapshot, feature-contract, and RTO
identities. It reserves the test atomically before loading 2025 rows; a crash is
therefore an auditable consumed run rather than an opportunity for another
look. The exclusive reservation is keyed by a deterministic frozen experiment-
protocol SHA-256 in the operator-controlled locked-test registry, independent
of winner and artifact outcomes, while a readable receipt is stored beside the
selection artifact. Both started records are completely written and fsynced
before an atomic no-clobber link makes them visible. Training artifacts also
record exact Python/platform, CUDA/cuDNN, H100 capability/memory, deterministic,
TF32, and dependency identities; both candidates must match, and stable system
and accelerator fields must match the locked run.
The runner actively enables deterministic Torch/cuDNN algorithms, disables
cuDNN benchmarking, fixes Python/NumPy/Torch and trainer data seeds, and sets
the CUDA BLAS workspace policy before model load; these are enforced settings,
not descriptive labels.

Chronos itself retains only its best checkpoint according to aggregate loss
across the frozen rolling validation tasks. The post-fit audit evaluates that
returned checkpoint with the richer MASE/WIS gate; it does not retrospectively
rescore every discarded checkpoint.
This is a remaining limitation and is why the full step trace and late-loss
rebound are preserved.

## The oracle limitation

At commit `36ceaff`, `experiments.features` marked realized future temperature
as a future covariate. With `with_gen=true`, realized wind and solar were also
marked future-known. This was the **oracle upper-bound lane**: it measured how
the model behaved with information unavailable to an operational forecaster.

The v0.2 experiment facade and production service both use the shared
`load-v2-core` contract, which forbids those observed fields in the future and
uses calendar-only future covariates. The archived oracle result still cannot
be used as the production service's expected accuracy or as an apples-to-apples
operator comparison. It also cannot be regenerated by current v0.2 code
without checking out the archived implementation.

## Required evaluation lanes

All future claims must identify one of the following lanes:

1. **Oracle upper bound.** Realized future covariates are allowed and named.
2. **Vintage replay.** Every input has an availability or issue timestamp no
   later than the forecast origin. Weather comes from an archived NWP run that
   existed at that origin.
3. **Live forward.** A forecast issuance is written immutably before any target
   outcome, then verified later using a declared actuals-vintage policy. The
   v0.2 lane covers the seven RTO/ISOs listed above and defaults to
   `eia-latest-at-plus72h-v1`, which freezes the latest eligible EIA revision
   available no later than 72 hours after each target.

The implementation details and promotion rules are in
[benchmark-protocol.md](benchmark-protocol.md).

## Uncertainty

p10-p90 is an 80% model interval, not a guarantee. A complete evaluation must
report empirical interval coverage, interval width, and pinball or CRPS-style
scores per BA and horizon. Calibration should be measured separately for
ordinary periods and declared stress/event slices. Crossing quantiles,
non-finite values, and negative-load outputs are publication failures.

The source tree includes a validation-only rolling conformal research harness.
It compares per-BA and seven-RTO pooled residual calibration windows using
2024 validation origins, then selects the lowest interval score among
candidates within two percentage points of the 80% coverage target. Each
calibrated origin can read only earlier validation residuals. The selected
window and pooling rule must be frozen before a locked-once test evaluation.

That harness deliberately uses `retrospective_final` source data and does not
establish production calibration, vintage-replay performance, or a public
live-forward claim. A live calibrator would additionally need to consume only
outcomes whose declared maturity boundary had passed before each issuance.
Until prospective evidence exists, its output is model-development evidence
only.

## Limitations

- EIA demand is revised after first publication.
- Weather coverage is incomplete across the 53-BA registry.
- One station or centroid is not a full spatial weather representation for a
  large BA.
- Grid topology, outages, prices, behind-the-meter generation, and demand
  response are not modeled directly.
- Historical average performance does not establish performance during
  extremes or structural change.
- The public service has no SLA and is not intended for dispatch, bidding,
  trading, or other regulated decisions.

## Claim review

A claim may move into the README or website only when its result bundle
contains the configuration, code SHA, model revision, data-manifest hash,
origin list, per-BA metrics, aggregate method, and known limitations. Oracle,
replay, and live-forward numbers must never be combined in one unlabeled table.
