# Adaptive conformal ablation

`experiments/run_aci_c2.py` is validation-only. It does not read or score the
held-out 2025 lane.

## Why

The shipped fixed-level protocol in `docs/conformal-experiment.md` failed its
own acceptance bar. Zero of twelve candidates kept every RTO's empirical
coverage within two percentage points of 80%, so that run fell back to its
documented no-qualifying-candidate rule. One shared quantile level cannot
serve seven RTOs whose base quantiles miscalibrate in different directions.

Adaptive conformal inference learns the level online instead:

    alpha <- alpha + gamma * (alpha_target - 1{y not in C})

## Protocol

- Everything the fixed-level runner does is unchanged: origin schedules, the
  `+72h` maturity rule, normalized CQR scores, the no-shrink clamp, the
  chronological inner/outer split, and the equal-BA scoring. The only variable
  is how the quantile level is chosen.
- Candidates: the shipped fixed-level calibration at each window, plus adaptive
  calibration over `gamma` in {0.002, 0.005, 0.01, 0.02, 0.05}, windows
  {56, 168}, and both alpha scopes.
- `alpha_scope=per-lead` keeps one level per (RTO, lead), as in the multi-step
  ACI literature. `alpha_scope=per-series` shares one level across an RTO's 24
  leads, so it observes 24x more outcomes per matured origin.
- Feedback is delayed. An origin's miscoverage may only move a level once that
  origin's final target hour plus 72 hours has matured, so levels update in
  batches at the origin where the outcome first becomes observable.
- Bias-corrected ACI (arXiv:2604.13253) additionally recenters intervals on an
  EWMA bias estimate. That half is deliberately not implemented: Surge's
  calibration policy forbids moving an interval off p50.

Selection minimizes the equal-BA macro relative interval score among candidates
whose every per-BA coverage error is within two points, on the first
chronological half. The selected candidate alone is reported on the untouched
later half.

## Reproducible invocation

```bash
export SURGE_CODE_REVISION="$(git rev-parse HEAD)"
export SURGE_DATA_SNAPSHOT_SHA256="<64-character snapshot SHA-256>"

python -m experiments.run_aci_c2 --out artifacts/adaptive-conformal-validation.json
```

## Result

`artifacts/adaptive-conformal-validation.json`. Reported on the untouched later
half of 2024, against the same uncalibrated forecasts:

| Calibration | Coverage | Worst per-RTO coverage error | Macro relative interval score | Meets the ±2pt bar |
|---|---:|---:|---:|:--:|
| Fixed level, window 56 | 0.8242 | 0.0399 | 1.0088 | no |
| Fixed level, window 168 | 0.8173 | 0.0344 | 1.0061 | no |
| Adaptive per-lead, best | 0.8063 | 0.0221 | 1.0046 | no |
| Adaptive per-series, γ=0.05, w=168 | 0.7956 | 0.0140 | 1.0051 | **yes** |

The effect is systematic, not a single lucky configuration. Across the grid,
**no** fixed-level or per-lead candidate met the bar, while six of ten
per-series candidates did, with worst-case error falling monotonically as
`gamma` rises. Per-series adaptation is also cheaper in interval score than the
fixed level at equal window.

Three honest limitations:

- **The search was iterative.** The per-lead family was evaluated first,
  including against the reporting half; the per-series family was added after
  seeing that result. Two families have therefore been inspected against this
  segment, so this is a promising result rather than a single-shot confirmation.
  A clean confirmation requires data neither family has seen — in practice the
  live-forward ledger, not the locked 2025 lane.
- **Calibration still costs interval score here.** Every macro relative interval
  score above is greater than 1: on this half, calibration buys coverage
  correctness at a small Winkler cost. Adaptive calibration is the cheaper way
  to buy it, not a free improvement.
- The run executed on CPU in `float32`, not the frozen H100 `bfloat16` runtime
  identity; the artifact's `runtime` block records this.

Nothing here changes serving. No forecaster applies these adjustments, and no
public accuracy or calibration claim rests on this experiment.
