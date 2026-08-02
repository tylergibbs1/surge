# Chronos-2 conformal experiment

`experiments/run_conformal_c2.py` is a validation-only experiment. It does not
read or score the held-out 2025 lane.

## Protocol

- The runner forecasts the seven v0.2 RTOs on non-overlapping 24-hour origins
  in 2024. All seven origin schedules must match exactly; partial intersections
  fail before model inference.
- A forecast origin's residuals enter calibration only when its final target
  hour plus 72 hours is no later than the current forecast origin. This mirrors
  the timing of `eia-latest-at-plus72h-v1`.
- Historical values are still `retrospective_final`. The experiment is not an
  exact reconstruction of the EIA revision available at +72 hours, and the JSON
  artifact records that limitation.
- Every candidate is evaluated on the intersection of its eligibility mask.
  Missing values therefore cannot give pooled candidates an easier evaluation
  cohort.
- The first chronological half of validation selects the window and pooling
  policy. The selected policy alone is reported on the later, untouched half;
  later residuals enter it only prequentially after maturity.
- Selection minimizes the equal-weight mean of each BA's calibrated-to-baseline
  interval-score ratio. A candidate is feasible only if every BA's empirical
  coverage is within two percentage points of 80%. If none is feasible, the
  fallback is explicit in `selection.rule_applied`.
- Cross-RTO pooling uses normalized conformity scores from complete origin
  blocks. Its finite-sample rank correction counts temporal origin blocks, not
  simultaneous BA cells. This remains a relational heuristic, not a claim that
  simultaneous RTO errors are independent or exchangeable.
- Negative CQR adjustments are clamped to zero. Calibration may widen an
  interval but never silently shrink it or exclude p50.

## Reproducible invocation

The runner requires a full Git commit SHA and data-snapshot SHA-256. Remote
models require an immutable 40-character Hugging Face revision. Local model
artifacts are hashed with `sha256-tree-v1`; a supplied artifact hash must match.

```bash
export SURGE_CODE_REVISION="$(git rev-parse HEAD)"
export SURGE_DATA_SNAPSHOT_SHA256="<64-character snapshot SHA-256>"

python -m experiments.run_conformal_c2 \
  --model amazon/chronos-2 \
  --model-revision 29ec3766d36d6f73f0696f85560a422f50e8498c \
  --out artifacts/conformal-validation.json
```

Inference is performed once for all 2024 origins. Candidate calibration and
selection reuse the in-memory forecast arrays; the runner does not repeat GPU
inference per window.

The stdout prefix remains `CALIBRATION_RESULT:`. `--out` is written atomically.
For compatibility, `best` and `candidates` contain inner-selection results;
`outer_validation.result` is the untouched reporting result and must not be
used to revise the selected policy.

## Result

`artifacts/conformal-validation.json` is the first and only execution of this
protocol. It used the pinned `amazon/chronos-2` revision, code revision
`8284c88a0d3bb9cf49cdd41b420549d2a90bc7dc`, and the frozen data snapshot
`77d80d40...`. The 2025 lane was not opened (`held_out_test_lane` is
`not-opened`).

**The predeclared selection rule found no qualifying policy.** Zero of twelve
candidates kept every RTO's empirical coverage within two percentage points of
80% on the inner half, so `selection.rule_applied` is
`fallback-minimum-relative-score-no-candidate-met-coverage`. The reported
policy is the fallback minimum-relative-score candidate: all-seven-RTO
normalized-block pooling over a 56 mature-origin window.

Reported on the untouched second half of 2024 (2024-07-02 to 2024-12-31,
25,508 points):

| | Coverage | Mean width (MW) | Macro interval score |
|---|---:|---:|---:|
| Uncalibrated | 0.7623 | 3401.5 | — |
| Calibrated | 0.8076 | 3789.2 | 0.9923× uncalibrated |

Rolling-origin conformal calibration therefore removes most of the aggregate
under-coverage — 76.2% to 80.8% against an 80% nominal interval — for about
11% wider intervals and a 0.77% macro interval-score improvement. It does not
equalize coverage across RTOs: ISNE reaches 83.3% and SWPP 82.9%, so the worst
per-RTO coverage error is 3.35 percentage points, still outside the ±2-point
tolerance. Under-coverage is a systematic property of the base model's
quantiles that a single shared window cannot correct uniformly.

Two limitations bound this result. The run executed on CPU in `float32`, not
the frozen H100 `bfloat16` runtime identity used for model selection; the
`runtime` block records this. And the calibration policy is not applied by the
serving path — `src/surge/verification.py` measures
`pi80_calibration_error_pct` but no forecaster consumes these adjustments.
