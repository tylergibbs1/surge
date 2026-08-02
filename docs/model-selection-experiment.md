# Frozen v0.2 model-selection experiment

This document freezes the H100 experiment before either candidate is trained.
It governs an offline, `retrospective_final` research lane; it is not a vintage
replay or a live-forward accuracy claim.

## Immutable inputs

- data snapshot SHA-256:
  `77d80d4031e2391808103ef29bb182b3ee2469cec1c24ae00569d217bd48a4c0`;
- release-safe base: `amazon/chronos-2` at revision
  `29ec3766d36d6f73f0696f85560a422f50e8498c`;
- feature contract: `load-v2-core` (its SHA-256 is recorded by each run);
- RTO order: PJM, CISO, ERCO, MISO, NYIS, ISNE, SWPP;
- train valid times: before 2024; validation valid times: 2024; locked test:
  2025 and later;
- seed 42, 24-hour horizon, 2,048-hour context, batch size 32, LoRA,
  learning rate `1e-5`, no generation covariates;
- deterministic Torch algorithms, deterministic cuDNN, fixed Python/NumPy/
  Torch and trainer data seeds, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, bfloat16,
  and TF32 on the single visible H100;
- overfitting policy: `surge-v0.2-overfit-gate-v1`, including 90 daily
  diagnostic origins per split and comparison with the frozen base on the
  identical validation origins. Origins must be shared across all seven RTOs
  and have all 24 realized targets finite. This availability-only filter is
  applied before cohort selection and before any model prediction is made.

Checkpoint selection evaluates the 90 shared complete origins immediately
preceding the latest 90 validation origins. The later cohort is reserved for
promotion metrics and never contributes to checkpoint `eval_loss`. Chronos
receives the preceding 2,048 hours only as context and the final 24 hours of
each task as validation labels; training-period labels do not enter checkpoint
`eval_loss`.

The reviewed Git commit used for execution is supplied as `--code-revision`
and recorded in every artifact. The selection query excludes 2025 valid times
before materialization.

## Candidates and selection rule

Two candidates differ only in training duration:

1. `official-lora-1000`: 1,000 optimizer steps;
2. `official-lora-2000`: 2,000 optimizer steps.

A candidate is eligible only if every versioned overfitting gate passes. Among
eligible candidates, choose the smallest predeclared score

`0.5 * (validation MASE / base MASE) + 0.5 * (validation scaled WIS / base scaled WIS)`.

An exact tie at six decimal places selects the 1,000-step candidate. If neither
candidate is eligible, the pinned zero-shot model remains the serving default
and the locked test is not opened. Gate thresholds and this rule are not
changed after validation metrics are observed.

Promotion also requires the 95% upper bound from a 2,000-resample paired
circular moving-block bootstrap (seven origins per block, seed 42) for the
candidate/base macro-MASE ratio to be at most 1.05. RTOs stay paired within an
origin, and adjacent origins stay together inside each resampled block.

The downloaded `Tylerbry1/surge-fm-v3` checkpoint is a validation-only research
ablation. Its historical weights descend from the documented oracle-covariate
lane, so neither it nor an adapter initialized from it is eligible for v0.2
production promotion.

## Locked raw-model test and validation-only calibration

After the winner is frozen, one explicitly named `locked-once` run may open
2025 for the **raw selected model**. It uses daily 24-hour origins, p50 point
metrics, p10/p50/p90 probabilistic metrics, and 2,000 paired origin-block
bootstrap resamples with seed 42. The atomic test receipt is created before any
2025 row is loaded. A catchable post-reservation exception is mirrored to the
receipt and registry as a terminal sanitized `failed` record; abrupt process
loss remains `started`. Either state consumes the attempt. Test results may be
reported with their `retrospective_final` limitation but cannot be used to
choose a different candidate, alter the gate, or tune another model.

The rolling conformal harness remains a separate validation-only research
lane. Its selected policy is not applied by `experiments.run_c2`, and this
experiment makes no calibrated-2025 claim. A future calibrated locked test
would require a separately frozen runner and a genuinely unopened outcome
period; it may not reuse 2025 after the raw-model test has opened it.

## Executed outcome

Both frozen candidates passed every overfitting gate on the disjoint promotion
cohort. The 2,000-step candidate won with selection score `0.9722636478503976`
versus `0.9829746249582003` for the 1,000-step candidate. The selector and both
candidate chains are retained under `artifacts/v0.2/`.

The one authorized 2025 run opened on 2026-08-02 and failed closed before
producing metrics. Two NYIS daily origins had only 23 of 24 finite target hours,
which violated the frozen full-partition bootstrap rule. The started receipt
and registry reservation remain consuming and byte-identical; they were not
removed or used to justify a retry. There is therefore no v0.2 locked-test
accuracy result, the adapter is not promoted as a tested serving default, and
the pinned upstream base remains the serving default. See
`artifacts/v0.2/README.md` and the checksum-bound incident sidecar for the exact
evidence.

Any future test must use a genuinely unopened period and a new frozen protocol.
Before model inference, that protocol should derive one shared RTO cohort using
only a predeclared target-availability/quality rule, hash the retained and
excluded origins, enforce an attrition threshold, and terminate as
`not-scoreable` without retry if the threshold is not met. That rule cannot be
applied retroactively to the consumed v0.2 test.
