# Changelog

All notable changes to Surge Grid are documented here. The project follows
[Semantic Versioning](https://semver.org/) while the public API remains alpha.

## [0.2.0] - 2026-08-01

### Added

- Methodology, benchmark protocol, reproducibility, operations, restore, known
  issue, security, citation, and release-checklist documentation.
- CI gates for supported Python versions, package artifacts, and the web app.
- Explicit source-canary and bake postcondition checks.
- Checksummed recovery-snapshot generation and verification.
- A versioned feature contract shared by training, evaluation, and serving,
  plus an immutable seven-RTO issuance and verification ledger.
- A validation-only rolling conformal interval-calibration harness with causal
  residual windows and explicit per-BA versus seven-RTO pooling selection.
- An hourly Modal job that pins mature +72-hour EIA outcomes into the immutable
  verification ledger.
- A fail-closed seven-RTO overfitting audit with train/validation MASE and WIS,
  per-RTO dispersion, checkpoint traces, and frozen-upstream regression gates.
- A checksummed promotion chain and atomic one-shot locked-test receipt bound to
  the promoted model, code, data snapshot, feature contract, and RTO identities.
- A stable experiment-protocol reservation key plus Python, platform,
  CUDA/cuDNN, H100, determinism, precision, and TF32 runtime provenance.

### Changed

- Renamed the install distribution to `surge-grid`; the Python import remains
  `surge` and the command-line entry point is `surge-grid`.
- Moved package version metadata to `src/surge/_version.py` as the single source.
- Reclassified archived v2/v3 scores as oracle-covariate research results.
- Restricted v0.2 future covariates to deterministic calendar fields and made
  p50 the consistently scored and served point estimate.
- Replaced the legacy `surge-fm-v3` serving default with a pinned upstream
  `amazon/chronos-2` revision; v3 remains historical benchmark material only.
- Parameterized the Modal app and volume, added clean-clone data bootstrap, and
  added scheduled EIA load and seven-RTO ASOS weather refreshes.
- Withdrew the unsupported “beats operators on 6 of 7 RTOs” and always-on
  claims pending vintage replay and live-forward evidence.

### Security and reliability

- Defined freshness from source/data cutoffs rather than request timestamps.
- Required complete, validated multi-BA publication and explicit rollback
  evidence for release.

## [0.0.1] - 2026-04-20

- Initial public research prototype, data adapters, Chronos-2 checkpoints,
  FastAPI service, and Next.js playground.

[0.2.0]: https://github.com/tylergibbs1/surge/compare/36ceaff...v0.2.0
[0.0.1]: https://github.com/tylergibbs1/surge/tree/36ceaff
