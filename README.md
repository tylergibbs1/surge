# Surge Grid

**Open, probabilistic load forecasting and auditable grid data for the United
States.**

Surge Grid has three parts: a Python data library, a Chronos-2 forecast service,
and a web playground. It covers the 53 EIA-930 balancing authorities that publish
a demand series.

The v0.2 trust ledger has a narrower contract than the playground. The Python
ledger publishes a complete-run marker only after all seven organized-market
RTO/ISOs have compatible issuances. The Vercel `current` pointer advances only
after that complete run passes validation. Public run lists and the scoreboard
show only complete runs. A staged per-BA issuance stays available by direct ID
for audit.

The other 46 balancing authorities are a legacy, best-effort surface. The public
demo at [surgeforecast.com](https://surgeforecast.com) has no availability or
freshness promise.

- [Live playground](https://surgeforecast.com)
- [Source](https://github.com/tylergibbs1/surge)
- [Pinned serving model](https://huggingface.co/amazon/chronos-2/tree/29ec3766d36d6f73f0696f85560a422f50e8498c)
- [Legacy benchmark checkpoint](https://huggingface.co/Tylerbry1/surge-fm-v3)
- [Methodology](https://github.com/tylergibbs1/surge/blob/main/docs/methodology.md)
- [Operations and restore runbook](https://github.com/tylergibbs1/surge/blob/main/docs/operations.md)

![Day-ahead forecast vs. reality for US grids](https://raw.githubusercontent.com/tylergibbs1/surge/main/docs/plots/hero_forecast.png)

> [!IMPORTANT]
> **The earlier claim "beats operators on 6 of 7 RTOs" was wrong.** Two separate
> errors caused it. Both errors made Surge look better than it is.
>
> First, the v1-v3 benchmark used realized future temperature as an oracle input.
> Runs with generation enabled also used realized wind and solar. Production has
> none of these inputs.
>
> Second, the operator baseline was one hour out of alignment for PJM and CISO.
> This overstated their error by 39.5% and 14.1%. After correction, "PJM: 1.70x
> better" becomes 1.22x.
>
> **Surge is behind the grid operators.** On the 2024 validation lane, Surge
> averages 3.09% MAPE across seven RTOs. Three operators publish their own
> day-ahead accuracy: PJM 1.43%, ERCOT 2.16%, and CAISO 2.04%. On those three
> RTOs Surge scores 2.89% against their 1.88%. Surge is about one percentage
> point behind.
>
> The true gap is larger. Surge forecasts from a 00:00 UTC same-day origin, so
> its leads are 1 to 24 hours. The operators issue theirs the afternoon before,
> with leads of 14 to 38 hours. Surge gets the easier task.
>
> **Surge no longer publishes an operator baseline built from EIA-930.** The
> `DF` column is not the forecast the operator uses. The EIA form instructions
> excuse each respondent from making `DF` consistent with the `D` beside it, and
> EIA warns that the comparison "is not very meaningful" for some BAs. We
> checked: for ERCOT the column matches the published 2.16% almost exactly, but
> PJM reads 2.43% against a published 1.43%, and CAISO reads 5.50% against a
> published 2.04%. Every difference favored Surge.
>
> The shipped `surge-fm-v3` intervals are too narrow. Its own published
> evaluation reports 0.725 coverage for a nominal 80% band.
>
> For the numbers and the derivations, read the
> [accuracy restatement](https://github.com/tylergibbs1/surge/blob/main/docs/accuracy-restatement.md)
> and the
> [benchmark protocol](https://github.com/tylergibbs1/surge/blob/main/docs/benchmark-protocol.md).

## What is included

- `surge` is the Python import package. It reads EIA-930 load, ASOS weather, and
  public CAISO/ERCOT/PJM data into a local Parquet store.
- The v0.2 service uses upstream `amazon/chronos-2` at immutable revision
  `29ec3766d36d6f73f0696f85560a422f50e8498c`. It does not serve the legacy
  `surge-fm-v3` fine-tune, because that model comes from the archived oracle
  benchmark.
- A tuned replacement cannot become `best/` until its frozen audit passes. The
  audit covers generalization gaps, per-RTO dispersion, worst-grid and
  upstream-baseline regressions, checkpoint-loss stability, and window coverage.
- `surge.api` is a FastAPI inference service. It has OpenAPI documentation and
  NDJSON streaming.
- `web/playground` is the Next.js map, grid view, and forecast explorer.
- `surge.vintage` stores each EIA response under a content hash before anything
  changes it. EIA revises this data later, so a score is a claim about one
  vintage of the truth. A vintage that nobody captures cannot be rebuilt.
- `surge.calibration` corrects each interval against the settled history of the
  same BA. The raw 80% band of the model covers 73% to 76% of outcomes.

The v0.2 `load-v2-core` feature contract uses observed temperature only in the
historical context. It permits only deterministic calendar fields in the forecast
horizon. Observed temperature, wind, and solar cannot be future inputs. Surge does
not read an operational weather forecast, so it claims no forecast-time weather
skill.

## Measured accuracy

All numbers below use the 2024 validation lane, seven RTOs, and identical target
hours. The locked 2025 lane stays closed.

| Model | Mean MAPE |
|---|---:|
| Same hour, previous day | 4.79 |
| Open GBM baseline | 2.98 |
| Chronos-2, zero shot | 2.88 |
| **Blend of the two** | **2.72** |

The blend weight comes from the first half of 2024 only. The table reports the
second half, which did not choose that weight. Every RTO improves over its own
best single model, by 1.3% to 6.6%.

Against the operators, on the three RTOs that publish day-ahead accuracy:

| RTO | Surge | Operator, published |
|---|---:|---:|
| PJM | 2.86 | 1.43 |
| ERCO | 3.17 | 2.16 |
| CISO | 2.64 | 2.04 |
| **mean** | **2.89** | **1.88** |

MISO, NYISO, ISO-NE and SPP publish no comparable hourly day-ahead MAPE. Surge
reports its own number for those four RTOs, and shows no operator column,
instead of a proxy.

Calibration corrects the intervals. Replayed across 2024, the worst per-RTO
coverage error falls from 7.54 points to 1.19 points. Interval width grows by
10% to 20%.

Note: ISO-NE is the hardest RTO for both models, at about 4.9% MAPE. Its EIA-930
series is net load, and several GW of behind-the-meter solar are invisible in the
inputs. About 1.5 of that gap needs an irradiance forecast that Surge does not
use.

## Install

The distribution is `surge-grid`. The Python import stays `surge`.

```bash
pip install surge-grid
```

```python
import surge

df = surge.load(ba="PJM", start="2025-06-01", end="2025-06-02")
print(df.head())
```

The EIA loader needs a free `EIA_API_KEY`. Copy `.env.example`, then keep the
filled file out of version control.

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

Then query one balancing authority, or stream the complete set:

```bash
curl --fail 'http://127.0.0.1:8000/forecast/PJM?horizon=24'
curl --fail --no-buffer 'http://127.0.0.1:8000/forecast/stream?horizon=24'
curl --fail 'http://127.0.0.1:8000/bas'
```

To develop from source, install `pip install -e ".[dev,api]"` instead. For a
fully offline recovery, read the
[operations runbook](https://github.com/tylergibbs1/surge/blob/main/docs/operations.md).

## Evidence status

The archived v3 experiment reports a 2025 hold-out MASE of 0.636 across 53
balancing authorities. It reports 0.518 on the original seven-RTO subset. The
older v2 seven-RTO checkpoint reports 0.492.

These are **oracle-covariate offline results**. They use rolling 24-hour origins
and step 24. The MASE denominator comes from the training split of each BA. They
are useful for model development. They are not live-forward performance.

Surge Grid reports results in three lanes. You cannot compare a number from one
lane against a number from another lane.

1. **Oracle upper bound** — realized future inputs. Never deployable.
2. **Vintage replay** — only inputs available before the forecast origin.
3. **Live forward** — forecasts frozen before the outcome, and scored later
   against a declared actuals vintage.

The repository does not yet publish a complete lane-2 or lane-3 result set.
Operator comparisons and the 70-hour useful-horizon result stay historical oracle
experiments until someone runs them again under these protocols.

## Operational contract

- `generated_at_utc` is the time that inference ran.
- `context_end_utc` and `data_cutoff_utc` identify the newest model input.
- `published_at_utc` identifies publication, not source freshness.
- Model and code revisions travel with each immutable issuance.
- A partial seven-RTO bake must not replace the last complete published run.
- Surge can keep stale observations for diagnosis. It must mark them stale or
  unavailable. It must not show them as current.
- Each issuance records whether calibration applied, and why not when it did not.

The authoritative thresholds, the restore procedure, and the rollback rules are
in the
[operations runbook](https://github.com/tylergibbs1/surge/blob/main/docs/operations.md).

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

Read the
[contribution guide](https://github.com/tylergibbs1/surge/blob/main/CONTRIBUTING.md),
the
[reproducibility guide](https://github.com/tylergibbs1/surge/blob/main/docs/reproducibility.md),
and the
[release checklist](https://github.com/tylergibbs1/surge/blob/main/docs/release-checklist.md).

## Project status

v0.2 is an alpha research release. Its ledger contract covers PJM, CISO, ERCO,
MISO, NYIS, ISNE, and SWPP. The repository does not yet publish a settled
live-forward result set.

The frozen adapter experiment stays as
[checksummed model-selection evidence](artifacts/v0.2/README.md). Its one
authorized 2025 test failed closed on incomplete NYIS target windows, before it
produced any metric. Surge claims no locked-test accuracy result. The adapter is
not the serving default.

The 53-BA playground is useful for exploration and demonstration. It is not a
complete atomic release surface, a production control plane, or an availability
promise.

Current priorities are a live-forward ledger with a vintage archive, calibrated
uncertainty in serving, and a public scoreboard against frozen open baselines.

## License and disclaimer

MIT. See
[LICENSE](https://github.com/tylergibbs1/surge/blob/main/LICENSE).

Research and reference use only. **Do not use Surge for trading, regulated
bidding, dispatch, or bankability-grade decisions.** There is no SLA. A result
from a historical hold-out can fail to generalize to future conditions or to
extreme events.
