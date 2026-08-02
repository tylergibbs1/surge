# Surge Grid

**Open, probabilistic load forecasting and auditable grid data for the United States.**

Surge Grid combines a Python data library, a Chronos-2 forecast service, and a
web playground spanning the 53 EIA-930 balancing authorities that publish a
demand series. The v0.2 trust ledger deliberately has a narrower contract: the
Python ledger publishes an immutable complete-run marker only after all seven
organized-market RTO/ISOs have compatible issuances, and the Vercel `current`
pointer separately advances only after that complete run validates. Public run
lists and the scoreboard expose only marked complete runs; a staged per-BA
issuance remains available by direct ID for audit. The remaining 46-BA
explorer/API surface is legacy and best-effort. The public demo at
[surgeforecast.com](https://surgeforecast.com) has no availability or freshness
SLA.

- [Live playground](https://surgeforecast.com)
- [Source](https://github.com/tylergibbs1/surge)
- [Pinned serving model](https://huggingface.co/amazon/chronos-2/tree/29ec3766d36d6f73f0696f85560a422f50e8498c)
- [Legacy benchmark checkpoint](https://huggingface.co/Tylerbry1/surge-fm-v3)
- [Methodology](https://github.com/tylergibbs1/surge/blob/main/docs/methodology.md)
- [Operations and restore runbook](https://github.com/tylergibbs1/surge/blob/main/docs/operations.md)

![Day-ahead forecast vs. reality for US grids](https://raw.githubusercontent.com/tylergibbs1/surge/main/docs/plots/hero_forecast.png)

> [!IMPORTANT]
> The historical v1-v3 benchmark used realized future ASOS temperature as an
> oracle covariate; experiment configurations with generation enabled also use
> realized wind and solar. Those results are an upper-bound research lane, not
> an apples-to-apples production comparison. The earlier "beats operators on
> 6 of 7 RTOs" claim has been withdrawn until a vintage-weather replay and an
> immutable live-forward evaluation are published. See the
> [benchmark protocol](https://github.com/tylergibbs1/surge/blob/main/docs/benchmark-protocol.md).

## What is included

- `surge` is the Python import package. It pulls and harmonizes EIA-930 load,
  ASOS weather, public CAISO/ERCOT/PJM data, and related grid datasets into a
  local Parquet store.
- The v0.2 service defaults to upstream `amazon/chronos-2` at immutable revision
  `29ec3766d36d6f73f0696f85560a422f50e8498c`. It does not serve the legacy
  `surge-fm-v3` fine-tune, whose training provenance belongs to the archived
  oracle benchmark.
- A tuned replacement cannot become `best/` unless its frozen audit passes
  train/validation generalization gaps, per-RTO dispersion, worst-grid and
  upstream-baseline regressions, checkpoint-loss stability, and complete-window
  coverage. The locked 2025 test stays unopened until that decision is frozen.
- `surge.api` is a FastAPI inference service with OpenAPI documentation and
  NDJSON streaming.
- `web/playground` is the Next.js map, grid view, and forecast explorer.
- The seven-RTO v0.2 forecast ledger separates issuance time, input cutoff,
  model revision, publication time, and verification so a newly generated
  response cannot make old source data look fresh. It stages immutable per-BA
  issuances, exposes a public run only after one compatible seven-RTO
  `forecast_runs` marker exists, and keeps direct issuance detail available for
  audit. The validated Vercel `current` pointer is a second atomic publication
  boundary.

The v0.2 `load-v2-core` feature contract uses observed temperature only in the
historical context and permits only deterministic calendar fields in the
forecast horizon. Observed temperature, wind, and solar are structurally
forbidden as future inputs. Surge does not yet ingest an operational weather
forecast, so the service makes no claim of forecast-time weather skill.

## Install

The distribution is named `surge-grid`; the Python import remains `surge`.

```bash
pip install surge-grid
```

```python
import surge

df = surge.load(ba="PJM", start="2025-06-01", end="2025-06-02")
print(df.head())
```

The EIA loader requires a free `EIA_API_KEY`. Copy `.env.example` and keep the
filled file outside version control.

## Run the API locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install "surge-grid[api]==0.2.0"

export EIA_API_KEY="..."
export SURGE_DATA_DIR="$PWD/.surge-data"
python -m surge.ingest \
  --bas PJM CISO ERCO MISO NYIS ISNE SWPP \
  --days 90
uvicorn surge.api.main:app --host 127.0.0.1 --port 8000
```

Then query one balancing authority or stream the complete set:

```bash
curl --fail 'http://127.0.0.1:8000/forecast/PJM?horizon=24'
curl --fail --no-buffer 'http://127.0.0.1:8000/forecast/stream?horizon=24'
curl --fail 'http://127.0.0.1:8000/bas'
```

For source development, replace the package install with
`pip install -e ".[dev,api]"`. A fully offline recovery, including the pinned
model and data snapshot, is documented in the
[operations runbook](https://github.com/tylergibbs1/surge/blob/main/docs/operations.md).

## Evidence status

The archived v3 experiment reports a 2025 hold-out MASE of 0.636 macro across
53 balancing authorities and 0.518 on the original seven-RTO subset. The
original v2 seven-RTO checkpoint reports 0.492. These are **oracle-covariate
offline results**: rolling 24-hour origins, step 24, with MASE denominators
computed per BA from the training split. They are useful for model development,
but they are not live-forward performance estimates.

Surge Grid reports evaluation results in three non-interchangeable lanes:

1. **Oracle upper bound** — realized future covariates; never labeled deployable.
2. **Vintage replay** — only inputs whose issue or availability time precedes
   the forecast origin.
3. **Live forward** — forecasts frozen before outcomes and scored later against
   a declared actuals vintage.

The repository does not currently publish a complete lane-2 or lane-3 result
set. Operator comparisons and the approximately 70-hour useful-horizon result
remain historical oracle experiments until rerun under those protocols.

## Operational contract

- `generated_at_utc` is when inference ran.
- `context_end_utc` / `data_cutoff_utc` identify the newest model input.
- `published_at_utc` identifies publication, not source freshness.
- Model and code revisions travel with each immutable issuance.
- A partial seven-RTO bake must not replace the last complete published run.
- Stale observations may be retained for diagnosis, but must be marked stale or
  unavailable rather than silently presented as current.

The authoritative thresholds, restore procedure, and rollback rules live in
the [operations runbook](https://github.com/tylergibbs1/surge/blob/main/docs/operations.md).

## Development

```bash
pip install -e ".[dev,api]"
ruff check src tests modal_app/app.py scripts/rebuild_data_snapshot.py \
  experiments/conformal.py experiments/eval_c2.py experiments/features.py \
  experiments/finetune_c2.py experiments/overfit.py experiments/run_c2.py \
  experiments/run_conformal_c2.py experiments/select_c2_candidate.py \
  scripts/audit_ledger.py scripts/score_ledger.py \
  scripts/verify_forecasts.py
mypy src modal_app/app.py
pytest

cd web/playground
pnpm install --frozen-lockfile
pnpm lint
pnpm typecheck
pnpm build
```

See the [contribution guide](https://github.com/tylergibbs1/surge/blob/main/CONTRIBUTING.md),
the [reproducibility guide](https://github.com/tylergibbs1/surge/blob/main/docs/reproducibility.md),
and the [release checklist](https://github.com/tylergibbs1/surge/blob/main/docs/release-checklist.md).

## Project status

v0.2 is an alpha research release. Its Vercel pointer publication and
forward-scoring ledger contract cover PJM, CISO, ERCO, MISO, NYIS, ISNE, and
SWPP. The repository does not yet publish a settled live-forward result set.
The frozen adapter experiment is retained as
[checksummed model-selection evidence](artifacts/v0.2/README.md), but its one
authorized 2025 test failed closed on incomplete NYIS target windows before
producing metrics. No locked-test accuracy result is claimed, and the adapter
is not the tested serving default.
The broader
53-BA playground is useful for exploration and demonstration, but it is not a
complete atomic release surface, a production control-plane, or an availability
promise. Current priorities are forecast-time weather replay, calibrated
uncertainty, reproducible snapshots, and decision-focused scenario tooling.

## License and disclaimer

MIT; see [LICENSE](https://github.com/tylergibbs1/surge/blob/main/LICENSE).

Research and reference use only. **Not for trading, regulated bidding,
dispatch, or bankability-grade decisions.** There is no SLA. Results from a
historical hold-out may not generalize to future conditions or extreme events.
