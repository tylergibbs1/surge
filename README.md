# surge

**Open, probabilistic day-ahead load forecasts for the US power grid.**

A fine-tuned Chronos-2 foundation model + FastAPI service covering every US
balancing authority that publishes a demand series to EIA-930 — 53 BAs
spanning the Eastern, Western, and Texas interconnections. Public data
only, one-command deploy, permissive license.

![Day-ahead forecast vs. reality for US grids](docs/plots/hero_forecast.png)

*Chronos-2 fine-tuned on 7 years of EIA-930 load + ASOS temperature + calendar
features. Dashed line = median forecast, shaded band = 80% probability
interval, solid = actual.*

## What it is

- `surge` — Python library for pulling and harmonising US grid data
  (EIA-930 load, ASOS temperature, wind/solar generation, CAISO OASIS,
  ERCOT public reports, NOAA storm events). Central BA registry in
  `surge.bas` tracks all 67 EIA-930 balancing authorities (53 with a
  demand series, 14 gen-/transmission-only).
- `surge-fm-v3` — Chronos-2 fine-tuned on 7 years × 53 BAs of load with
  temperature + calendar covariates. **Test MASE 0.636** on 2025 hold-out
  (macro over 53 BAs); **0.518** on the original 7 RTOs (PJM/CAISO/ERCOT/
  MISO/NYISO/ISO-NE/SPP). Beats seasonal-naive-24 by 36% overall, 48% on
  the RTO subset.
- `surge-fm-v2` — Previous generation, 7-BA RTO-only model. **Test MASE
  0.49** on 2025 hold-out. Still available via `SURGE_MODEL_PATH` for
  users who want the narrower-but-sharper 7-BA variant.
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
python -m surge.ingest --days 90    # populate data store (all 53 BAs by default)
# checkpoint auto-downloads from https://huggingface.co/Tylerbry1/surge-fm-v3
uvicorn surge.api.main:app --port 8000

# 24-hour probabilistic forecast for PJM
curl 'http://localhost:8000/forecast/PJM?horizon=24'

# Streaming NDJSON for every supported BA
curl -N 'http://localhost:8000/forecast/stream?horizon=24'

# Full BA registry (codes, names, stations, peak MW)
curl 'http://localhost:8000/bas'
```

Response:

```json
{
  "ba": "PJM",
  "model": "surge-fm-v3",
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

![Surge vs. every classical and foundation baseline](docs/plots/leaderboard.png)

### 7-RTO subset (same benchmark as v1/v2)

| Model | Test MASE | vs. seasonal-naive-24 | Cost |
|---|---:|---:|---|
| seasonal-naive-24 (baseline) | 1.044 | — | — |
| Prophet (with temp regressor) | 2.023 | +94% worse | — |
| XGBoost hourly-binned (Roy '25) | 0.901 | −14% | — |
| N-BEATS (Pelekis '23) | 0.714 | −32% | — |
| Chronos-Bolt zero-shot | 0.688 | −34% | — |
| Chronos-2 zero-shot + covariates | 0.567 | −46% | — |
| **surge-fm-v2 (7-BA specialist)** | **0.492** | **−53%** | free |
| **surge-fm-v3 (53-BA generalist) — RTO subset** | **0.518** | **−50%** | free |

### All 53 BAs (v3 only — v2 doesn't cover most of these)

| Slice | Test MASE | n BAs | MAE (MW macro) |
|---|---:|---:|---:|
| All demand-reporting BAs | 0.636 | 53 | 272 |
| 7 RTO/ISOs | 0.518 | 7 | 889 |
| 46 non-RTO utilities | 0.653 | 46 | 178 |

All numbers: 2025 hold-out, rolling 24h-ahead windows at step=24, MASE
denominator = per-BA train-set seasonal-naive (m=24). Weather covariates
are ground-truth ASOS temperature where available (7 RTO BAs ingested from
day one; the 46 non-RTO BAs use a zero-filled fallback pending a backfill
from the new weather-station mapping in `surge.bas`).

## How far ahead can it forecast?

![Horizon degradation curve](docs/plots/horizon_curve.png)

Chronos-2 beats seasonal-naive-24 out to **~70 hours (≈ 2.9 days)** ahead.
Beyond that, weekly patterns win. Useful-horizon ceiling without adding
weather forecasts is therefore roughly 3 days; extending it further requires
HRRR/GFS weather as future covariates.

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
- [x] Phase 2 (this release): all EIA-930 BAs (53 demand-reporting, 67 total
      registered), surge-fm-v3 checkpoint, BA registry in `surge.bas`,
      dynamic `/bas` metadata endpoint, playground map extended to every
      BA footprint
- [ ] Phase 2: ASOS backfill for the 46 new BAs (Iowa Mesonet rate-limit
      cleanup — currently zero-filled), LMP forecasting task, Hugging Face
      dataset release
- [ ] Phase 2: always-on hosted demo
- [ ] Phase 3: scenario simulator (surge-sim)
