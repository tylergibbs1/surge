# Public issue status

This page records the release impact and closure conditions for public issues
and operational gaps found during the v0.2 trust review. GitHub remains the
discussion source of truth for numbered issues.

## Issue 1: benchmark data leakage

[Issue #1](https://github.com/tylergibbs1/surge/issues/1) is valid and is a
v0.2 documentation/claim release blocker. The archived offline evaluation uses
observed future ASOS temperature; configurations with generation enabled may
also use realized wind and solar. It is an oracle upper bound, not an
apples-to-apples comparison with forecasts issued in production.

The v0.2 `load-v2-core` contract now structurally forbids observed weather and
generation in future covariates. That prevents the same implementation error
in current training, evaluation, and serving, but it does not retroactively
validate the archived scores or supply the missing vintage-weather benchmark.

The README now withdraws the operator-beating headline and separates oracle,
vintage replay, and live-forward lanes. The issue should remain open until a PR
links:

- exact configs and input cutoffs;
- a checksummed data/result bundle;
- per-BA metrics for a valid replay or live-forward run; and
- synchronized README and model-card language.

## Issue 2: ONNX / edge deployment

[Issue #2](https://github.com/tylergibbs1/surge/issues/2) is an open deployment
enhancement, not a v0.2 release blocker. Surge Grid does not currently claim a
supported ONNX export. Chronos-2's pipeline behavior, covariates, quantiles, and
dynamic horizons make an untested one-line export recipe unsafe to recommend.

The issue can close when either:

1. a supported export includes a pinned opset, target runtime/device, fixed or
   dynamic shape contract, numerical-parity tolerances, and runtime smoke tests;
   or
2. maintainers document a deliberate unsupported decision and the supported
   CPU/container alternative.

Until then, keep the issue labeled as an enhancement and ask requesters for the
target runtime, device, opset, and horizon requirements.

## Operational gap: settlement evidence

The ledger, `eia-latest-at-plus72h-v1` verifier, and hourly Modal settlement
job are implemented. The job has not yet been deployed and observed against
the production Modal volume, however. The public scoreboard must therefore
show settled scores as unavailable rather than infer them from current final
data. Closure requires a monitored production deployment, one fully mature
seven-RTO run, a successful ledger audit, and retained logs proving the exact
outcome cutoffs.
