# surge

**Open, probabilistic day-ahead load forecasts for the US power grid.**

A fine-tuned Chronos-2 foundation model + FastAPI service + hosted
playground covering every US balancing authority that publishes a demand
series to EIA-930 — 53 BAs spanning the Eastern, Western, and Texas
interconnections. Public data only, one-command deploy, permissive license.

- **Live demo** — [surgeforecast.com](https://surgeforecast.com)
- **Code** — [github.com/tylergibbs1/surge](https://github.com/tylergibbs1/surge)
- **Model** — [huggingface.co/Tylerbry1/surge-fm-v3](https://huggingface.co/Tylerbry1/surge-fm-v3)

![Day-ahead forecast vs. reality for US grids](docs/plots/hero_forecast.png)

*Chronos-2 fine-tuned on 7 years of EIA-930 load + ASOS temperature + calendar
features. Dashed line = median forecast, shaded band = 80% probability
interval, solid = actual. Forecasts use only information available at
forecast time — no future weather.*

## What it is

- `surge` — Python library for pulling and harmonising US grid data
  (EIA-930 load, ASOS temperature, wind/solar generation, CAISO OASIS,
  ERCOT public reports, NOAA storm events). Central BA registry in
  `surge.bas` tracks all 67 EIA-930 balancing authorities (53 with a
  demand series, 14 gen-/transmission-only).
- `surge-fm-v3` — Chronos-2 fine-tuned on 7 years × 53 BAs of load with
  temperature + calendar covariates. This is the **published** checkpoint.
  Given real archived day-ahead weather forecasts it reaches **Test MASE
  0.540** on the 2025 hold-out (macro over 53 BAs) and **0.536** on the
  original 7 RTOs (PJM/CAISO/ERCOT/MISO/NYISO/ISO-NE/SPP) — beating
  seasonal-naive-24 by 44% and 49% respectively, using only information
  available at forecast time. Without future weather the same checkpoint
  scores 0.627 / 0.594.
  **How you feed it matters more than which checkpoint you use**: see
  [Accuracy](#accuracy-vs-the-status-quo), and note that the flat-temperature
  covariate the API shipped for months is *worse than sending none at all*.
- `surge-fm-v2` — Previous generation, 7-BA RTO-only model. Its published
  figure was measured under the superseded protocol and has not been
  re-scored; treat it as unverified. Still available via
  `SURGE_MODEL_PATH`.
- `surge.api` — FastAPI inference service with NDJSON streaming and OpenAPI docs.
- `web/playground` — Next.js playground at **surgeforecast.com**. Four
  coordinated views over the same data:
    * **Map** — MapLibre US choropleth, colour-coded by interconnect,
      pins sized by current load. Click any BA to drill into its chart.
    * **Grid** (`/grid`) — 53 BA cards sortable by % of all-time peak,
      peak GW, or name. Filters for interconnection (Eastern/Western/
      Texas) and size tier (RTO / major utility / small). Each card
      shows a 24 h sparkline and a traffic-light status dot.
    * **Live hero** — [Liveline](https://github.com/benjitaylor/liveline)
      canvas chart of rolling 24 h US aggregate demand (~165 GW
      overnight, ~240 GW on a summer afternoon), polling
      `/api/live-load` every 60 s with a visibility-gated interval and
      keep-last-good fallback on transient 502s.
    * **Now indicator** on each BA's forecast chart — a dashed vertical
      line + pulsing SVG dot at the current hour, sliding rightward
      through the 24 h window once a minute.
- Daily "bake" — `/api/bake` regenerates the full forecast set at
  06:15 UTC and writes `forecasts/{BA}.json` + `forecasts/all.json`
  to Vercel Blob. The read-side tries the blob first (~300 ms edge-
  cached) and falls through to live Modal inference on miss or
  `?force=1` (~3 s cold).

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

### What counts as a fair number

A day-ahead forecaster knows the calendar and its own load history. It does
**not** know tomorrow's weather or tomorrow's wind and solar output. Every
covariate here therefore carries an explicit policy, and the benchmark
harness *proves* causality rather than trusting the label: it perturbs all
values at and after the forecast origin and requires the future covariates
not to move (`experiments/causal_guard.py`).

| policy | meaning |
|---|---|
| `known` | calendar features — deterministic from the timestamp |
| `past_only` | supplied as history, never over the horizon |
| `persistence` | frozen at the last observed value |
| `oracle` | realized future values — perfect foresight, **not a forecast** |

Earlier versions of this table declared observed ASOS temperature and EIA
*actual* wind/solar generation as known-future inputs, which leaked the
answer into the input. Those figures (RTO 0.518, 53-BA 0.636) were
perfect-foresight upper bounds, not forecasts, and they were also computed
before two data defects were fixed — a store that was scanned without
deduplication (up to 4 rows per hour) and a test window with no upper bound
that silently grew past 2025. Full accounting in [#1](../../issues/1).

### 7-RTO subset (2025 hold-out, causal covariates)

![Surge vs. classical and foundation baselines, causal covariates](docs/plots/leaderboard.png)

| Model | Test MASE | vs. seasonal-naive-24 |
|---|---:|---:|
| seasonal-naive-24 (baseline) | 1.044 | — |
| XGBoost hourly-binned (Roy '25) | 1.019 | −2% |
| Chronos-2 zero-shot, no future weather | 0.620 | −41% |
| surge-fm-v3, no future weather | 0.594 | −43% |
| Chronos-2 zero-shot **+ forecast weather** | 0.564 | −46% |
| **surge-fm-v3 + forecast weather** | **0.536** | **−49%** |

Two things this table is built to show:

1. **The covariate matters more than the checkpoint.** Real day-ahead forecast
   weather moves surge-fm-v3 from 0.594 to 0.536 — a bigger jump than any
   amount of fine-tuning produced.
2. **The fine-tune only earns its keep once the weather is good.** Given *no*
   future weather, surge-fm-v3 (0.594) is barely ahead of stock Chronos-2
   (0.620) and the confidence intervals overlap. Given the *same* forecast
   covariates it wins clearly, 0.536 vs 0.564, with non-overlapping intervals
   on validation. The reason is mechanical: the checkpoint was fine-tuned on
   exact temperature, so it trusts the covariate — which is a liability when
   the input is crude and an advantage when it is accurate (~1.3 °C MAE).

Prophet, N-BEATS, Chronos-Bolt zero-shot and surge-fm-v2 have not been
re-scored under this protocol, so their old figures are omitted rather than
carried over — mixing them with the numbers above would repeat the
apples-to-oranges comparison this table exists to fix.

### All 53 BAs

Test MASE, 2025 hold-out. "no wx" = no future temperature; "forecast" = real
archived day-ahead forecast.

| Slice | v3, no wx | v3 **+ forecast** | retrained (unreleased) | Chronos-2 zero-shot + forecast |
|---|---:|---:|---:|---:|
| All 53 demand-reporting BAs | 0.627 | **0.540** | 0.532 | 0.569 |
| 7 RTO/ISOs | 0.594 | **0.536** | 0.532 | 0.564 |
| seasonal-naive-24, all 53 | 0.956 | — | — | — |

Against stock Chronos-2 given identical forecast covariates, published v3 wins
by 5.1% over all 53 BAs (0.540 vs 0.569) with non-overlapping 95% intervals
— [0.534, 0.548] against [0.563, 0.577]. That is the first result in this
repo where fine-tuning is demonstrably worth something rather than being
within noise of the base model.

Macro MAE with forecast weather is 250 MW over all 53 BAs and 1,073 MW over
the 7 RTOs; those differ by an order of magnitude simply because RTOs are far
larger, which is why MASE rather than MAE is the headline metric.

**You do not need a new checkpoint.** Retraining on the forecast channel buys
only ~1%, so the published `surge-fm-v3` captures essentially the whole gain
provided you feed it a real forecast. What to send is documented on the
[model card](https://huggingface.co/Tylerbry1/surge-fm-v3).

All numbers: 2025 hold-out (exactly 8,760 h, pinned), rolling 24 h-ahead
windows at step=24, MASE denominator = per-BA train-set seasonal-naive
(m=24), store deduplicated on `(ts_utc, ba)`.

### Where the weather data comes from

`surge.scrapers.openmeteo` pulls **archived forecasts** from Open-Meteo's
Previous Runs API — specifically `temperature_2m_previous_day1`, the value
that was forecast for hour *t* roughly 24 h before *t*. That is genuinely
knowable at day-ahead forecast time, unlike the observed ASOS series this
repo used to pass over the horizon.

Both temperature channels come from the same Open-Meteo grid point. Taking
history from an ASOS station and the future from a model centroid left a
~5 °C discontinuity exactly at the forecast boundary for PJM, whose station
(DCA) sits ~200 km from its load centroid. Measured day-ahead skill at the
seven RTO centroids over 2025: **MAE 1.06–1.55 °C, correlation 0.983–0.990**,
bias −0.40 to +0.20 °C.

This also supersedes the ASOS backfill for the 46 non-RTO BAs. Open-Meteo
serves any lat/lon, so those BAs went from a zero-filled weather channel to
real history *and* forecasts; the past-channel switch alone was worth 2.0%
before any forecast was added.

    python scripts/backfill_weather_forecast.py --bas all --start 2021-03-01

### Perfect-foresight ceiling, for reference only

Replacing the day-ahead forecast with **perfect foresight of the same
temperature channel** scores **0.488** on the RTO subset and **0.446** over
all 53 BAs. That is the true ceiling for this configuration, and it says how
much a better weather forecast could still be worth:

| | no future wx | real forecast | perfect foresight |
|---|---:|---:|---:|
| 7 RTOs | 0.594 | **0.536** | 0.488 |
| All 53 BAs | 0.627 | **0.540** | 0.446 |

So a real GFS forecast captures a little over half the available weather
headroom on the RTOs, and roughly half across all 53. The remaining ~9%
(RTOs) is reachable only with a better forecast — a sharper NWP model,
ensemble means, or more variables — not with a better load model. That keeps
weather, not architecture, as the top lever.

A caution about earlier versions of this section: they quoted a ceiling of
0.468 (RTO) / 0.610 (all 53) measured against the *ASOS* channel, which is
zero-filled for 46 of 53 BAs. Those were not valid upper bounds. The all-53
figure was actually **worse** than what the honest configuration now
achieves, because perfect foresight of a mostly-empty channel is worth less
than a real forecast of a populated one. Any "oracle" number is only a
ceiling for the exact channel it was measured on.

Perfect-foresight figures are reported here for calibration of expectations
only. They are standard practice in the load-forecasting literature as
explicitly-labelled upper bounds, and must never be compared against a real
forecaster.

Two further caveats worth stating plainly:

- **Calibration is imperfect.** The nominal 80% interval covers ~76% in
  practice. The model is mildly overconfident.
- **Five BAs have inflated MASE denominators.** The outlier filter clips
  only `load_mw > 200_000`, an absolute threshold, so a 70 GW spike in a
  1.8 GW BA survives and inflates that BA's denominator (BANC by ~60×).
  BANC, SPA, LGEE, SEC and TEPC are affected; excluding them the 53-BA
  macro is ~0.673 rather than 0.597. Fixing this is pending.

### vs. the grid operators' own forecasts

**Surge beats EIA's day-ahead demand forecast on 4 of 7 major RTOs.**

Every RTO/ISO submits a day-ahead load forecast to EIA each morning — that's
the *production* forecast used to schedule generation. We pull the operator
submissions (`type=DF` on EIA's Grid Monitor endpoint) and score them against
actuals for the exact same 2025 window, same 24h horizon, same per-BA MASE
denominator surge uses — 8,760 hours per BA, ~61,000 hours total.

This comparison is only meaningful if both sides face the same uncertainty.
The operators had to *forecast* weather and renewable output when they
submitted; surge is therefore scored with causal covariates only. An earlier
version of this table claimed 6 of 7 wins by giving surge realized weather
and realized wind/solar — a handicap the operators did not get.

| Region | Surge MAE | Operator MAE | Ratio | Surge MASE | Operator MASE |
|---|---:|---:|---:|---:|---:|
| PJM | 2,187 MW | 3,297 MW | 1.51× | 0.45 | 0.68 |
| CAISO | 626 MW | 2,098 MW | **3.35×** | 0.50 | 1.66 |
| ERCOT | 1,357 MW | 1,366 MW | 1.01× | 0.56 | 0.56 |
| MISO | 1,326 MW | 1,786 MW | 1.35× | 0.41 | 0.55 |
| NYISO | 537 MW | 560 MW | 1.04× | 0.57 | 0.60 |
| **ISO-NE** | **592 MW** | **306 MW** | **0.52×** | **0.65** | **0.34** |
| SPP | 885 MW | 2,590 MW | **2.93×** | 0.60 | 1.77 |
| **macro (7 RTOs)** | **1,073 MW** | **1,715 MW** | **1.60×** | **0.536** | **0.880** |

Surge figures are the **published** `surge-fm-v3` supplied with day-ahead
forecast weather — reproducible with the checkpoint on Hugging Face, not an
unreleased model.

Macro MAE is ~37% lower than the operators' own submissions and macro MASE
~39% lower. **ISO-NE is the sole loss**, and their team is genuinely elite —
MASE 0.34 at a 24-hour horizon is excellent. ERCOT and NYISO are effectively
ties (1.01× and 1.04×). Two operator submissions, **CAISO (1.66) and SPP
(1.77)**, still did worse than a "same as yesterday" baseline over 2025.

An earlier version of this table claimed 6 of 7 by handing surge *realized*
weather and *realized* wind and solar — a handicap the operators never got.
Stripping that took it to 4 of 7. Real day-ahead forecast weather brings it
back to 6 of 7 legitimately. The original claim happened to be about right;
the evidence for it was not.

Reproduce: `python scripts/compare_eia_df.py --start 2025-01-01 --end 2026-01-01`
for the operator side, then `python -m experiments.run_c2 rto7
'{"on":"test","future_mode":"none"}'` for surge's.

## How far ahead can it forecast?

![Horizon degradation curve](docs/plots/horizon_curve.png)

With day-ahead forecast weather, per-step skill crosses the seasonal-naive-24
line at **67 hours (≈2.8 days)**. Matched-horizon MASE:

| horizon | 1 h | 6 h | 24 h | 72 h | 168 h |
|---|---:|---:|---:|---:|---:|
| with forecast wx | 0.210 | 0.299 | **0.532** | 0.688 | 0.806 |
| no future wx | 0.213 | 0.305 | 0.572 | 0.901 | 1.202 |

Weather is what buys horizon. Without it the crossover is 41 h and the model
is *worse* than "same as last week" at one week out (1.202); with it the
crossover roughly doubles to 67 h and a week-ahead forecast still beats naive
(0.806).

There is some history in this number worth recording. The original claim here
was "~70 hours", produced under the leaky protocol. Stripping the leak took it
to 41 h. Adding a real forecast brings it back to 67 h. The original figure
was approximately correct and the reasoning behind it was not — it happened to
approximate what a good forecast delivers by instead assuming a perfect one.

Note the flat-temperature covariate the API shipped for months is measurably
*harmful*: RTO MASE 0.740 with it against 0.594 with no future temperature at
all. A constant 24 h temperature implies no diurnal cycle, and the checkpoint
was fine-tuned to trust the covariate. Crude causal substitutes
(same-hour-yesterday, month × hour climatology) also lost to withholding
temperature. Only a genuine forecast helps.

Closing the 0.572 → 0.468 perfect-foresight gap therefore needs a *genuine*
weather forecast — real HRRR/GFS archives (`surge.scrapers.hrrr` exists but
is not wired into ingest) used at both training and inference time, so the
model learns how much to trust an imperfect forecast.

## Status

Pre-release, hosted demo live. The API runs locally from a one-line
`uvicorn`, the 53-BA checkpoint auto-downloads from Hugging Face on
first request, and the playground at [surgeforecast.com](https://surgeforecast.com)
is open to anyone. See [roadmap](#roadmap) for what's next.

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

Research and reference use only. **Not for trading, regulated bidding, or
bankability-graded decisions.** No SLA. Accuracy numbers are measured on a
specific 2025 hold-out and may not generalise to future extreme events.
Every figure states the covariate regime it was measured under; numbers
taken under different regimes are not comparable, and the perfect-foresight
figures are upper bounds rather than achievable forecasts.

## Roadmap

- [x] Phase 0: scaffold, data library (7 BAs load + weather), parquet store
- [x] Phase 1: Chronos-2 fine-tune, benchmark vs classical + FM baselines
- [x] Phase 1: FastAPI inference service
- [x] Phase 2: all EIA-930 BAs (53 demand-reporting, 67 total registered),
      surge-fm-v3 checkpoint, BA registry in `surge.bas`, dynamic `/bas`
      metadata endpoint, playground map extended to every BA footprint
- [x] Phase 2: always-on hosted demo at [surgeforecast.com](https://surgeforecast.com)
      with map + grid + live US-demand hero + now-indicator, daily bake
      to Vercel Blob, Modal fallback for on-demand inference
- [x] Real archived day-ahead weather forecasts as a causal future covariate
      (`surge.scrapers.openmeteo`, Open-Meteo Previous Runs API). Worth
      −9.4% on the 53-BA macro — the single largest gain measured so far
- [x] Weather coverage for all 53 BAs. This supersedes the ASOS backfill:
      Open-Meteo serves any lat/lon, so the Iowa Mesonet rate-limit blocker
      no longer gates weather coverage. The 46 previously zero-filled BAs now
      have real temperature history *and* forecasts
- [ ] Forecast renewables (`shortwave_radiation`, `wind_speed_100m` at the
      same lead time) — realized wind/solar was worth a further 0.043 MASE
- [ ] Conformal calibration. The nominal 80% band covers ~68-70% and gets
      *worse* with every fine-tuning step, while zero-shot Chronos-2 is at
      0.795. This is the clearest outstanding defect
- [ ] Publish the forecast-trained checkpoint as surge-fm-v4 (only ~1% better
      than v3 given the same covariates, so low priority)
- [ ] Per-BA robust outlier filter (the absolute 200 GW cutoff inflates the
      MASE denominator for BANC, SPA, LGEE, SEC, TEPC)
- [ ] Stop sending flat persistence temperature from the live API; it scores
      worse than sending no future temperature
- [ ] Conformal calibration — the nominal 80% band covers ~76%
- [ ] Phase 2: LMP forecasting task, Hugging Face dataset release
- [ ] Phase 3: scenario simulator (surge-sim)
