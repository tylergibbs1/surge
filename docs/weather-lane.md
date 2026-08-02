# The weather lane

Surge serves calendar-only forecasts. This document records what adding weather
would cost, what it would buy, and the rail that has to exist first.

## Why this is written before any weather data exists

Surge published an accuracy claim built on realized future temperature once
already, and the retraction is in `docs/accuracy-restatement.md`. Nothing in the
code made that mistake impossible, so it survived until a reviewer noticed.

`src/surge/weather_vintage.py` is the missing rail. A forecast value carries the
moment it became knowable, a backtest refuses values published after the origin,
and archives that cannot answer "what was known at time T" cannot be constructed
at all.

## The trap, by name

Several archives are named as though they retain issue time and do not:

| Source | Status |
|---|---|
| Open-Meteo **Historical Forecast API** | Stitches the first hours of successive runs into one series. That is an analysis. **No issue time.** |
| Open-Meteo Historical Weather API, ERA5 | Reanalysis |
| NSRDB, GOES-derived GHI, SURFRAD | Satellite or ground observation |
| NYISO **P-70A** behind-the-meter | Estimated actuals. **P-70B** is the forecast-vintage sibling |
| ISO-NE realized BTM series | Observation |

Each would score well and mean nothing. They belong in the diagnostic lane, where
they answer "what is the ceiling", never in a published number.

## What is actually available

Open-Meteo's **Previous Runs** API returns each hour's value as forecast about 24
hours earlier, back to January 2024. Roughly 20 MB covers seven RTOs for the
whole period. It is a lead-time offset rather than an exact issue time, and it
errs older, which is conservative in the honest direction. Any report must name
that convention rather than imply exact origin alignment.

NOAA's NBM is the license-safe fallback, public domain, back to May 2020, at
around 9 GB. Building an HRRR GRIB pipeline is not worth it: about 200 GB and a
week or two for the same signal.

NYISO publishes a genuine day-ahead behind-the-meter forecast (P-70B) back to
November 2020. For NYISO that is better than irradiance, because it is already
in MW.

ISO-NE's behind-the-meter forecast has **no archive**. Every day nobody snapshots
it is a day that cannot be recovered.

## What it would buy

Less than first estimated. The earlier 1.5-point figure for ISO-NE was closer to
an oracle ceiling than a realizable gain.

A reasonable projection is **0.3 to 0.8 points** of all-hours MAPE on ISO-NE,
concentrated in the middle of the day. Roughly 5 GW of behind-the-meter capacity
gives about 3.5 to 4 GW at midday. A competent day-ahead regional PV forecast is
wrong by 8 to 15% of capacity, which is 300 to 600 MW against a 13 GW system.
That is the whole addressable error, and no model recovers the part where the
cloud field itself is wrong.

For scale: perfectly observed sky-imager data, at 10 to 30 minute horizons where
clouds are most predictable, bought 0.75 to 1.5% relative in the published work.
A day-ahead result far larger than that is a leak, not a result.

So weather would narrow the gap to the operators. It would not close it.

## How both lanes get reported

One code path, two feature configurations, both scored on every backtest and
both published in every report.

The calendar-only lane stays the headline claim. The weather lane is an
explicitly labelled variant that names its source and its lead-time convention.
Neither is published without the other, because the gap between them is the
interesting result: it measures what weather forecasts are worth, which is an
ablation the published literature does not contain.

## Before any of this

1. Measure the ceiling with observed data in the diagnostic lane. If a perfect
   behind-the-meter signal does not remove most of the midday error, stop.
2. Then measure the realizable gain with vintage-correct forecasts.
3. The difference between those two numbers is the irreducible cloud error. It
   is the honest headroom, and no pipeline can buy it.
