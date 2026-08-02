# Contributing

Thanks for helping make Surge Grid more useful and more trustworthy. Bug
reports, data-source corrections, methodology reviews, and focused pull
requests are welcome.

## Development setup

```bash
git clone https://github.com/tylergibbs1/surge.git
cd surge
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api]"
```

Copy `.env.example` to a local file that is not committed. `EIA_API_KEY` is
required for live EIA ingestion; the other source credentials are optional for
the corresponding adapters.

## Required checks

```bash
ruff check src tests modal_app/app.py scripts/rebuild_data_snapshot.py \
  experiments/conformal.py experiments/eval_c2.py experiments/features.py \
  experiments/finetune_c2.py experiments/overfit.py experiments/run_c2.py \
  experiments/run_conformal_c2.py scripts/audit_ledger.py scripts/score_ledger.py \
  scripts/verify_forecasts.py
mypy src modal_app/app.py
pytest
python -m build
```

Frontend changes must also pass:

```bash
cd web/playground
pnpm install --frozen-lockfile
pnpm lint
pnpm typecheck
pnpm build
```

Tests must not read `~/.surge`, depend on an existing manifest, or call the
network unless they are explicitly marked as integration tests. Use temporary
data directories and mocked HTTP transports.

## Methodology changes

Forecasting claims are part of the product surface. A PR that adds or changes a
metric must include:

- the evaluation lane: oracle, vintage replay, or live forward;
- exact forecast origins, horizons, input-availability cutoffs, and actuals
  vintage;
- model, code, configuration, and data identifiers;
- per-BA results alongside any aggregate;
- a machine-readable result artifact and its SHA-256 hash;
- a limitations note covering leakage and missingness.

See [docs/methodology.md](docs/methodology.md) and
[docs/benchmark-protocol.md](docs/benchmark-protocol.md). Do not describe an
oracle result as production, apples-to-apples, or operator-beating.

## Data, models, and generated files

- Source code, small fixtures, and documentation belong in Git.
- Model weights and large Parquet snapshots do not. Publish them as immutable,
  checksummed release or Hugging Face artifacts.
- The Python distribution is `surge-grid`; the import package is `surge`.
- The v0.2 serving default is pinned upstream
  [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2). Checkpoints
  under [Tylerbry1 on Hugging Face](https://huggingface.co/Tylerbry1) are
  legacy research artifacts unless a later release explicitly says otherwise.
- Never publish from a personal working data directory. Build and verify a
  snapshot with `scripts/rebuild_data_snapshot.py` as described in
  [docs/operations.md](docs/operations.md).

## Pull requests and releases

Keep changes scoped and explain user-visible behavior. `main` should require
the Python, package, and frontend checks. Before merging to `main`, the project
workflow requires independent subagent review and resolution of the review's
actionable concerns.

Releases follow [docs/release-checklist.md](docs/release-checklist.md). PyPI
publishing should use the protected `pypi` GitHub environment and trusted
publishing rather than a long-lived API token.
