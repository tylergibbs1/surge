# surge

**Open, probabilistic day-ahead load forecasts for the US power grid.**

A fine-tuned Chronos-2 foundation model + FastAPI service covering 7 major US
balancing authorities (PJM, CISO, ERCO, MISO, NYIS, ISNE, SWPP). Public data
only, one-command deploy, permissive license.

## What it is

- `surge` — Python library for pulling and harmonising US grid data
  (EIA-930 load, ASOS temperature, wind/solar generation, CAISO OASIS,
  ERCOT public reports, NOAA storm events).
- `surge-fm-v2` — Chronos-2 fine-tuned on 7 years × 7 BAs of load with
  temperature + calendar covariates. **Test MASE 0.49** on 2025 hold-out,
  **1.7–2.3% MAPE** across BAs. Beats seasonal-naive by 52.9%.
- `surge.api` — FastAPI inference service with NDJSON streaming and OpenAPI docs.

## Quick start

### Library

```python
import surge

# 24h of PJM hourly load, written to a local parquet store
df = surge.load(ba="PJM", start="2025-06-01", end="2025-06-02")
print(df.head())
```

### API

```bash
pip install -e ".[api]"
python -m surge.ingest --days 90                    # populate data store
# ... download model checkpoint (see docs/models.md)
uvicorn surge.api.main:app --port 8000

# 24-hour probabilistic forecast for PJM
curl 'http://localhost:8000/forecast/PJM?horizon=24'

# Streaming NDJSON for all 7 BAs
curl -N 'http://localhost:8000/forecast/stream?horizon=24'
```

Response:

```json
{
  "ba": "PJM",
  "model": "chronos-2-ft-v2",
  "as_of_utc": "2026-04-18T20:54:13Z",
  "horizon": 24,
  "units": "MW",
  "points": [
    {"ts_utc": "2026-04-19T00:00:00Z", "median_mw": 112454, "p10_mw": 111570, "p90_mw": 113493},
    ...
  ]
}
```

## Accuracy vs. the status quo

| Model | Test MASE | vs. seasonal-naive-24 | Cost |
|---|---:|---:|---|
| seasonal-naive-24 (baseline) | 1.044 | — | — |
| Prophet (with temp regressor) | 2.023 | +94% worse | — |
| XGBoost hourly-binned (Roy '25) | 0.901 | −14% | — |
| N-BEATS (Pelekis '23) | 0.714 | −32% | — |
| Chronos-Bolt zero-shot | 0.688 | −34% | — |
| Chronos-2 zero-shot + covariates | 0.567 | −46% | — |
| **Chronos-2 full FT + covariates (this repo)** | **0.492** | **−53%** | free |
| Ensemble of 3 Chronos-2 variants | **0.453** | **−57%** | free |

All numbers: 7-BA macro average, 2025 hold-out, 367 rolling 24h-ahead windows,
MASE denominator = per-BA train-set seasonal-naive (m=24).

## Status

Pre-release. API works locally, model checkpoints published separately via
Hugging Face Hub. See [roadmap](#roadmap).

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

Research and reference use only. **Not for trading, regulated bidding, or
bankability-graded decisions.** No SLA. Accuracy numbers are measured on a
specific 2025 hold-out and may not generalise to future extreme events.

## Roadmap

- [x] Phase 0: scaffold, data library (7 BAs load + weather), parquet store
- [x] Phase 1: Chronos-2 fine-tune, benchmark vs classical + FM baselines
- [x] Phase 1: FastAPI inference service
- [ ] Phase 2: all 67 BAs, LMP forecasting task, Hugging Face model + dataset release
- [ ] Phase 2: always-on hosted demo
- [ ] Phase 3: scenario simulator (surge-sim)
