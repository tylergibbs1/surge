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

Two caveats on the table itself:

- **The search was iterative.** The per-lead family was evaluated first,
  including against the reporting half; the per-series family was added after
  seeing that result. Two families were inspected against this segment, so the
  2024 pass is not a single-shot confirmation.
- **Calibration still costs interval score here.** Every macro relative interval
  score above is greater than 1: on this half, calibration buys coverage
  correctness at a small Winkler cost. Adaptive calibration is the cheaper way
  to buy it, not a free improvement.

## Replication, and what it overturns

`experiments/run_aci_replication_c2.py` ran the frozen 2024 winner once, with no
selection step, on all of 2023 — 365 origins no calibration search had touched.
Result in `artifacts/adaptive-conformal-replication-2023.json`:

| Calibration | Coverage | Worst per-RTO coverage error | Macro relative interval score | Meets the ±2pt bar |
|---|---:|---:|---:|:--:|
| Fixed level, window 168 | 0.8258 | 0.0518 | 0.9976 | no |
| Frozen adaptive, γ=0.05, w=168 | 0.8033 | 0.0222 | 0.9935 | **no** |

**The 2024 pass did not replicate on 2023.** Worst-case error came in at 2.22
points against a 2-point bar, missing it, with ISNE over-covering at 0.822.

Widening to four cohorts puts that failure in proportion. Per-series adaptive
calibration at γ ∈ {0.02, 0.05}, window 168, worst per-RTO coverage error:

| Cohort | Adaptive | Fixed level | Adaptive meets ±2pt |
|---|---:|---:|:--:|
| 2021 | 0.0042–0.0081 | 0.0343 | yes |
| 2022 | 0.0080–0.0110 | 0.0376 | yes |
| 2023 | 0.0222–0.0236 | 0.0518 | **no** |
| 2024, reporting half | 0.0140 | 0.0399 | yes |

So the honest conclusion:

- Adaptive calibration is a large and consistent improvement. It cuts the worst
  per-RTO coverage error by roughly three to eight times versus the fixed level,
  on every cohort tested, and on 2021–2023 it also beats the uncalibrated
  forecasts on interval score.
- It clears the ±2-point bar on three of four cohorts and misses on 2023 alone.
  That is not a guarantee, and the ablation's single-cohort pass overstated it.
- The 2023 failure is **not** a persistent per-RTO defect. ISNE is well
  calibrated in 2021 (0.807) and 2022 (0.805) and only drifts in 2023 (0.822)
  and the 2024 half. Whatever breaks it is recent and time-varying, not
  structural to that RTO.

## Can tuning fix 2023? No

`artifacts/adaptive-conformal-grid/` characterizes 16 adaptive policies —
γ ∈ {0.02, 0.05, 0.1, 0.2} × window ∈ {56, 168} × both alpha scopes — plus two
fixed-level references, against all of 2021, 2022 and 2023. One inference pass
per cohort; no selection step. Worst per-RTO coverage error, bar 0.02:

| Policy | 2021 | 2022 | 2023 |
|---|---:|---:|---:|
| per-series, γ=0.05, w=168 | 0.0042 | 0.0080 | 0.0222 |
| per-series, γ=0.05, w=56 | 0.0047 | 0.0045 | 0.0255 |
| per-series, γ=0.1, w=168 | 0.0051 | 0.0051 | 0.0273 |
| per-series, γ=0.2, w=168 | 0.0082 | 0.0118 | 0.0311 |
| per-lead, best | 0.0232 | 0.0292 | 0.0418 |
| fixed level, best | 0.0325 | 0.0358 | 0.0518 |

**No policy in the grid meets the bar on all three cohorts.** Three findings
worth keeping:

1. Sharing one level across a series' leads dominates per-lead adaptation
   everywhere, by a factor of three to five. On a one-year cohort a per-lead
   level simply does not see enough matured outcomes.
2. Raising γ makes 2023 *worse*, not better (0.0222 → 0.0273 → 0.0311). The
   2023 miss is therefore not an adaptation-speed problem, which rules out the
   obvious fix.
3. The policy the 2024 search selected is also the best of the grid on 2023.
   The selection was sound; the target was simply not reachable by tuning these
   knobs.

The open question is what changed for ISNE in 2023 and stayed changed into
2024. That is a base-model or data question, not a calibration one.

Both runs executed on CPU in `float32`, not the frozen H100 `bfloat16` runtime
identity; each artifact's `runtime` block records this.

Nothing here changes serving. No forecaster applies these adjustments, and no
public accuracy or calibration claim rests on this experiment.
