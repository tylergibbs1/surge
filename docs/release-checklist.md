# v0.2 release checklist

All required boxes must be checked with linked evidence before publishing
`v0.2.0`. “Workflow green” is not evidence when a postcondition was not tested.

## Evidence and documentation

- [ ] Issue #1 has an acknowledgement and links to the methodology correction.
- [ ] README, website, and Hugging Face model card use the same oracle/replay/
      live-forward language.
- [ ] Unsupported operator-accuracy and availability claims are removed.
- [ ] Every retained metric links to a checksummed result bundle.
- [ ] Known limitations, missingness, aggregation, and the
      `eia-latest-at-plus72h-v1` actuals policy are documented.
- [ ] Issue #2 states ONNX is unsupported unless parity tests have landed.

## Code and CI

- [ ] Ruff, mypy, and hermetic pytest pass on Python 3.11, 3.12, and 3.13.
- [ ] Frontend lint, typecheck, and production build pass from the lockfile.
- [ ] Route tests reject stale, malformed, and partial baked payloads.
- [ ] Ingest tests cover empty/global-failure and freshness postconditions for
      all seven trust-ledger RTOs; wider BA coverage is reported separately.
- [ ] Package sdist/wheel build; `twine check`, clean install/import, version,
      and `surge-grid --help` smoke tests pass.
- [ ] `main` is protected and all required checks are enforced before Vercel
      production promotion.
- [ ] Independent subagent review is complete and actionable concerns are fixed.
- [ ] Fine-tune selection queried no 2025+ valid times and the overfit audit
      reports `test_opened=false` for exactly the seven trust RTOs.
- [ ] The frozen candidate and tie-break rule in
      `docs/model-selection-experiment.md` were committed before H100 runs;
      legacy oracle-lineage checkpoints were not production-promoted.
- [ ] Every `surge-v0.2-overfit-gate-v1` threshold passes, including the frozen
      upstream-baseline comparison; rejected candidates have no `best/` or
      `surge-promotion.json`.
- [ ] The promotion verifier reproduces every `best/` file hash, and the single
      locked-test run has a persisted receipt bound to matching base/code/data,
      feature-contract, and seven-RTO identities.
- [ ] The authoritative locked-test registry has exactly one reservation for
      the deterministic frozen experiment-protocol SHA; copying, regenerating,
      or retraining the candidate experiment cannot create another reservation
      in that registry.

## Artifacts and identity

- [ ] Distribution is `surge-grid`, import is `surge`, and version is `0.2.0`.
- [ ] PyPI name/control and trusted publisher are verified before tagging.
- [ ] Project URLs resolve to `tylergibbs1/surge` until an org migration exists.
- [ ] Code SHA, model revision, configuration, and data snapshot hash appear in
      forecasts and result bundles.
- [ ] The training manifest and overfit audit contain non-unknown immutable
      base/code/data revisions and exact dependency versions.
- [ ] Candidate manifests agree on Python/platform, CUDA/cuDNN, H100 model,
      capability/count/memory, deterministic, precision, and TF32 identity;
      stable runtime fields also match the locked evaluator.
- [ ] The served artifact is `amazon/chronos-2` at revision
      `29ec3766d36d6f73f0696f85560a422f50e8498c`; the legacy `surge-fm-v3`
      checkpoint is used only in explicitly labeled historical reproduction.
- [ ] The fine-tune and promotion chain enforce that same release-safe lineage;
      legacy, oracle, custom, and unknown base identities cannot be promoted.
- [ ] Wheel, sdist, data snapshot, model export, manifests, and SHA-256 files are
      retained in controlled release storage.
- [ ] `CHANGELOG.md`, `SECURITY.md`, `CITATION.cff`, and release notes agree.

## Restore rehearsal

- [ ] A fresh checkout builds a recovery snapshot and verifies its manifest.
- [ ] The offline API starts with `--no-index` from the prepared wheelhouse.
- [ ] A new versioned Modal volume is seeded without touching the old volume.
- [ ] `/live`, `/ready`, `/health`, PJM/CISO/ERCO, and all-BA probes pass with
      correct model/data/code revisions.
- [ ] A Vercel preview passes API and browser smoke tests before promotion.
- [ ] A forced partial bake leaves the current pointer unchanged.
- [ ] Rollback to the prior blob run, Vercel deployment, Modal release, and
      volume has been exercised and timed.

## Publish and observe

- [ ] Create the signed `v0.2.0` tag from the reviewed commit.
- [ ] Publish through the protected PyPI environment; do not use a local token.
- [ ] Verify PyPI metadata and install the public wheel into a clean environment.
- [ ] Synchronize the Hugging Face model card and immutable revision link.
- [ ] Trigger a complete seven-RTO bake and verify read-side freshness,
      provenance, and the exact PJM/CISO/ERCO/MISO/NYIS/ISNE/SWPP membership.
- [ ] Deploy and observe the +72-hour settlement job against the same production
      ledger/data volume; `scripts/audit_ledger.py` reports zero failures.
- [ ] Observe at least 24 hours, including an hourly ingest and daily bake.
- [ ] Keep the release under observation until one complete 168-hour issuance
      passes its final +72-hour maturity boundary (about ten days), settles, and
      audits cleanly before closing release-related methodology issues.
