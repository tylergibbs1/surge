# Surge Grid v0.2 model-selection evidence

This directory is the compact, checksummed evidence retained from the frozen
Surge Grid v0.2 H100 experiment. Execution used reviewed commit
`e6022e494ee79bb10f228bc2fead0d25b905fdd0` and the following immutable
inputs:

- base model: `amazon/chronos-2` at
  `29ec3766d36d6f73f0696f85560a422f50e8498c`;
- data snapshot SHA-256:
  `77d80d4031e2391808103ef29bb182b3ee2469cec1c24ae00569d217bd48a4c0`;
- feature contract: `load-v2-core` at
  `84a740bfe11062e6db03ac8ea61fe727f0c587abed4de89eff3e5c295054cdeb`;
- ordered regions: PJM, CISO, ERCO, MISO, NYIS, ISNE, and SWPP; and
- one NVIDIA H100 80 GB HBM3 with the exact dependency and deterministic
  runtime identity recorded in both training manifests.

## Frozen selection result

The candidates differed only in their predeclared 1,000- or 2,000-step
training duration. Both passed every `surge-v0.2-overfit-gate-v1` gate. The
selection score was frozen as the mean of the validation/base MASE ratio and
the validation/base scaled-WIS ratio; a six-decimal tie favored fewer steps.

| Candidate | Validation MASE | MASE/base | Scaled WIS | WIS/base | Paired MASE-ratio 95% CI | Selection score |
|---|---:|---:|---:|---:|---:|---:|
| `official-lora-1000` | 0.501879 | 0.983058 | 0.326224 | 0.982891 | [0.973719, 0.990611] | 0.982975 |
| `official-lora-2000` | 0.496713 | 0.972939 | 0.322472 | 0.971588 | [0.956031, 0.987785] | 0.972264 |

`official-lora-2000` was therefore frozen as the winner. Its train/validation
macro-MASE ratio was 1.055910, its validation RTO coefficient of variation was
0.126201, and its worst validation RTO MASE was 0.629784. The full-precision
macro, per-RTO, per-origin, checkpoint-trace, dispersion, and paired
moving-block-bootstrap evidence remains in each `surge-overfit-audit.json`.

## Consumed locked-test outcome

The single authorized 2025 look was reserved at
`2026-08-02T04:03:55.659718+00:00`. It failed closed before producing any
forecast metric because NYIS daily origins at 2025-01-15 and 2025-01-16 did not
have all 24 finite target hours required by the frozen bootstrap protocol. The
source contained NYIS `load_mw=0.0` at 2025-01-15 23:00Z and 2025-01-16
00:00Z; the frozen `(0, 200000]` validity rule correctly mapped those values to
missing, leaving 23 finite labels in each affected window. Partial inference
had begun before the bootstrap completeness assertion stopped the run, so the
attempt is unambiguously consumed.

The no-clobber receipt and authoritative registry copy remain byte-identical,
status `started`, and consuming. They were not deleted, rewritten, or used to
authorize a second run. `surge-locked-test-failure.json` is a post-incident
sidecar bound to their SHA-256 and the exact terminal exception. Consequently:

- there is **no v0.2 locked-test accuracy metric**;
- the selected adapter is not promoted as a tested serving default; and
- the pinned upstream base remains the v0.2 serving default.

## Interpretation and limitations

All candidate metrics are offline `retrospective_final` model-development
evidence with `point_in_time_replay=false`. The feature contract uses observed
temperature only in historical context and calendar-only future covariates.
Reported candidate quantiles are raw and uncalibrated. The legacy
`Tylerbry1/surge-fm-v3` checkpoint is excluded from promotion because its
historical training lineage used realized future-weather oracle inputs.

This compact bundle does not retain forecast, actual, or input-cutoff tables.
It is therefore not the complete benchmark bundle described in
`docs/benchmark-protocol.md`, a vintage replay, or live-forward evidence. The
selection and training audits are reproducible only while the checksummed model
artifacts and 216 MB data snapshot are retained in controlled storage; those
large files are intentionally excluded from Git.

Verify all retained files from this directory with:

```bash
shasum -a 256 -c SHA256SUMS
```
