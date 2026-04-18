# Contributing

Thanks for your interest. Surge is a small open-source project; PRs and issues
welcome.

## Development setup

```bash
git clone https://github.com/<your-fork>/surge.git
cd surge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api]"
```

You'll need a few free API keys in `.env` (see `.env.example`):
- `EIA_API_KEY` — register at https://www.eia.gov/opendata/register.php
- `NREL_API_KEY` (optional) — https://developer.nrel.gov/signup/
- `NCDC_TOKEN`  (optional) — https://www.ncdc.noaa.gov/cdo-web/token

## Running tests

```bash
pytest
```

## Running the API locally

```bash
# pull enough data to run inference
python -m surge.ingest --days 90

# download a model checkpoint (separate — see docs/models.md)
# ...

uvicorn surge.api.main:app --reload
```

## Style

- `ruff check src tests` must pass
- `mypy src` must pass
- Match surrounding style — small surface area, keep dependencies minimal.

## What goes in git vs. what doesn't

- **Source code**: git
- **Tests and fixtures**: git
- **Benchmarks / results**: `experiments/results.tsv` is gitignored — publish to HF datasets
- **Model weights**: gitignored — publish to HF Hub
- **Parquet data**: gitignored — can be rebuilt via `python -m surge.ingest`

## Publishing

- Dataset snapshots → `huggingface.co/datasets/surge-grid/*`
- Model checkpoints → `huggingface.co/surge-grid/*`
