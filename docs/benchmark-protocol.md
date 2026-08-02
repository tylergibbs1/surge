# Benchmark protocol

This protocol prevents future information, data revisions, and publication
time from being confused with forecast skill. A run that cannot satisfy its
lane's requirements may still be useful for debugging, but it is not a
publishable benchmark.

## Common definitions

- **Origin:** the UTC timestamp at which a forecast is considered issued.
- **Input cutoff:** the newest source record available to the model.
- **Availability time:** when a source record or revision became retrievable.
- **Weather issue time:** when an NWP forecast run was published.
- **Issuance:** the immutable forecast payload and its provenance.
- **Actuals vintage:** the rule selecting the target value used for scoring.

Each issuance must carry a deterministic ID, BA, origin, horizon, code SHA,
model ID and revision, input cutoff, data-manifest hash, generated time, and
forecast points. Retries with the same ID and different bytes are conflicts.

## Lane 1: oracle upper bound

Purpose: model and feature research under deliberately privileged inputs.

- Realized future temperature, wind, or solar is permitted.
- The result title and every table must include `oracle`.
- Results may compare architectures under the same oracle inputs.
- Results must not be described as deployable, live, production, or
  apples-to-apples with a forecast-time system.

The archived v2/v3 README values belong to this lane.

## Lane 2: vintage replay

Purpose: estimate historical forecast-time performance using only information
that existed at each origin.

For every origin:

1. select demand revisions with `availability_time <= origin` before
   deduplicating;
2. select the latest weather run with `issue_time <= origin`, then only lead
   times published in that run;
3. include deterministic calendar fields directly;
4. exclude observed wind/solar after the origin unless replaced by a forecast
   whose issue time is no later than the origin;
5. save the selected record IDs or an input hash with the issuance.

If an input has no trustworthy availability timestamp, it cannot be used in
this lane. Missing archived weather must be reported as missing, not silently
replaced with realized weather.

## Lane 3: live forward

Purpose: measure the behavior users actually received.

The v0.2 live-forward contract covers seven RTO/ISOs: PJM, CISO, ERCO, MISO,
NYIS, ISNE, and SWPP. A scheduled run is publishable only when all seven
compatible issuances validate and the Python ledger writes one immutable
complete-run marker. Public run listings and the scoreboard are marker-gated;
staged per-BA records remain available only through direct issuance lookup for
audit. The Vercel `current` pointer is a second atomic boundary and must
reference that same complete immutable run. The broader 53-BA endpoint is not
part of this publication lane.

1. Generate the issuance using the current data cutoff.
2. Persist it immutably before the first target timestamp.
3. Record publication and verification separately.
4. Never rewrite a forecast after outcomes arrive.
5. Score only after the actuals-vintage delay has elapsed.
6. Append verification records; do not mutate the issuance.

The default actuals policy for v0.2 is
`eia-latest-at-plus72h-v1`: for each target hour, select the latest eligible EIA
revision whose availability time is no later than 72 hours after that target.
First-published or later-finalized actuals may be reported as separately named
policies, but vintages must not be mixed.

## Model-selection gate

Fine-tuned v0.2 candidates must be selected without reading any 2025+ target.
The selection loader filters valid times before 2025 at query time and all
training/evaluation arrays are truncated again at the validation boundary. A
promotion audit must contain exactly PJM, CISO, ERCO, MISO, NYIS, ISNE, and
SWPP; immutable base-model, code, and data-snapshot revisions; train-versus-
validation MASE and scaled WIS; per-RTO dispersion and worst-RTO diagnostics;
the complete checkpoint/step trace; and an identical-origin comparison with
the frozen upstream baseline.

Every diagnostic and checkpoint-selection label window must be shared by all
seven RTOs and contain all 24 realized target hours. Target availability is
checked before cohort selection and inference, so missing observations cannot
create a model-dependent cohort. The earlier 90 validation origins select the
checkpoint; the latest 90 are reserved for promotion. The audit records
candidate, excluded-incomplete, reserved, and selected counts plus a hash of
each exact origin schedule. Candidate and baseline promotion hashes must match.

Every gate in `surge-v0.2-overfit-gate-v1` must pass before a checkpoint may be
named `best` or receive `surge-promotion.json`. Missing metrics, missing RTOs,
unknown revisions, inadequate origin coverage, unstable training, material
generalization gaps, or regression against the upstream baseline reject the
candidate. A rejected raw checkpoint remains an audit artifact only.

Only after the promotion marker is frozen may an explicitly authorized
`test_protocol=locked-once` run open 2025. Test results cannot be used to tune
the gate, choose another checkpoint, or revise the winning configuration. The
runner verifies every checkpoint file against the promoted manifest, requires
the test code/data/base identities and canonical RTO list to match that
manifest, and atomically creates `surge-locked-test-receipt.json` before it
loads test rows. An interrupted run remains consumed, and a second run against
the same frozen selection is rejected. A controlled
`SURGE_LOCKED_TEST_REGISTRY` stores an exclusive reservation keyed by a
deterministic experiment-protocol SHA-256 over the frozen policies, base/data/
feature/RTO identity, candidate configurations, and raw-test configuration.
It deliberately excludes outcomes, winner, timestamps, paths, and artifact
hashes, so copying, regenerating, or retraining the declared experiment does
not create another permitted look. A fully written started record is published
with an atomic no-clobber link; the readable receipt is written the same way
beside the frozen selection artifact.
The locked test uses the promoted context, horizon, and generation-covariate
configuration and the fixed full-partition daily-origin rule (`step=24`, no
origin cap), 2,000 paired origin-block bootstrap resamples, seed 42, and
per-horizon output; these choices cannot be changed after looking at test
outcomes.

## Leakage gates

A benchmark fails before scoring if any of these checks fail:

- an input availability or weather issue time is after the origin;
- the context contains a target timestamp at or after the origin;
- forecast timestamps are not consecutive hours beginning after the context;
- a live issuance was first persisted after any target outcome was available;
- code, model, configuration, or data identifiers are missing;
- oracle fields appear in a replay or live-forward configuration;
- a duplicate issuance ID contains different bytes.
- a forecast quantile is non-finite, negative, or crosses another quantile.

Automated tests should include deliberately leaky fixtures and prove that each
is rejected.

## Metrics and aggregation

At minimum, publish per BA and horizon:

- MAE and RMSE in MW;
- MASE with the training window and seasonal period stated;
- bias, defined as forecast minus actual so positive values mean overforecast;
- p10/p50/p90 pinball loss;
- p10-p90 empirical coverage, mean interval width, and WIS;
- discrete `crps_approx`, labeled as an approximation and calculated as twice
  the mean pinball loss over the reported quantile grid;
- number of origins and scored points.

Publish every metric both over the complete evaluation partition and for each
forecast horizon from 1 through H, with the finite-point count for every row.
Macro averages give each BA equal weight. A separately named load-weighted
view may weight BA-level metrics by mean scored actual MW when every weight is
finite and positive, but it must never replace or be labeled as the macro.
Never infer a national total by summing hours with inadequate BA coverage.

Confidence intervals must resample at an origin/day block level, not individual
hourly errors that share a forecast. A macro interval requires every included
BA to have the identical origin schedule and all H target hours finite at every
origin; unequal schedules and partial windows fail rather than being silently
intersected. Draw each complete cross-BA origin block once per bootstrap sample,
then aggregate with an equal-weight mean across BAs. Sampling BAs independently
would erase the same-origin dependence between regions and is not protocol
compliant.

## Operator comparisons

An operator comparison is publishable only when the exact operator submission
visible at the corresponding origin is archived. A historical endpoint that
returns one final `DF` value per target hour does not prove which submission was
available at a prior origin. The Surge and operator forecasts must use the same
target hours and a documented issuance alignment; scoring denominators alone do
not make the comparison apples-to-apples.

Until those conditions are met, the prior “6 of 7” comparison is withdrawn.

## Result bundle

Store the following together under an immutable run ID:

```text
run.json                 configuration and provenance
origins.parquet          origins and input cutoffs
forecasts.parquet        immutable forecast points
actuals.parquet          selected target vintage
metrics-per-ba.parquet   per-BA/per-horizon results
metrics.json             declared aggregates
SHA256SUMS               hashes for every artifact
```

The release notes or model card must link this bundle directly. A chart without
the bundle is illustrative, not benchmark evidence.
