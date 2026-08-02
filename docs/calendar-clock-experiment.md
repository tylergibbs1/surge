# Does the calendar clock matter?

Validation-only. The 2025+ lane was not read.

## The defect

`calendar_covariates` derived hour, weekday, weekend and holiday from the UTC
stamp for every BA. Cyclic hour and weekday survive that — for a single series
the offset is constant and a model can learn it — but the binary flags cannot.
A weekend and a US holiday are local-calendar spans, so a UTC flag is
misaligned by the BA's offset at both ends.

Measured over 2024, hours whose `is_weekend` flag disagrees with the BA's own
wall clock:

| BA | Mislabeled hours | Of which 17:00–21:00 local |
|---|---:|---:|
| CISO | 772 | 525 |
| ERCO, MISO, SWPP | 562 | 352 |
| PJM, NYIS, ISNE | 457 | 247 |

For CISO that is every Friday evening reading as weekend and every Sunday
evening reading as a weekday — landing squarely on the evening peak.

## The measurement

`load-v3-core` changes the clock and nothing else, so an A/B isolates it.
Identical model, revision, snapshot, origins and horizon; 2024 validation,
seven RTOs.

| Contract | Macro MAE (MW) | Coverage |
|---|---:|---:|
| `load-v2-core` (UTC flags) | 1252.4 | 0.7586 |
| `load-v3-core` (BA-local flags) | 1257.4 | 0.7574 |

**Correcting the clock did not help.** Macro MAE is 0.4% *worse*, and coverage
is unchanged. Per RTO, only CISO improves (672.9 → 671.7 MW) — the BA with by
far the worst misalignment — while PJM (2705 → 2722), ERCO, MISO, NYIS and
SWPP each get marginally worse.

Both differences are small enough that they are almost certainly noise. Each
target hour is covered by 24 overlapping forecasts, so a naive interval on
these means is roughly five times too narrow; nothing here is separable from
zero without a moving-block bootstrap at block length ≥ 96 hours. The honest
statement is **no detectable effect**, not "UTC is better".

## Why, most likely

Chronos-2 receives 2,048 hours of context. The weekly and daily rhythm is
directly visible in that history, so the model has little need of a binary flag
to tell it that Sunday evening is Sunday evening. A misaligned flag is
apparently closer to a weak, slightly noisy feature than to a misleading one.

## What was done about it

`load-v3-core` is declared, tested and **not adopted**. `load-v2-core` remains
active with its hash unchanged.

The defect is real as a matter of data correctness, and the fix is kept ready
because it will matter more where context is short, where a model is fine-tuned
on these features, or for any downstream consumer that reads the flags directly.
But it should not be described as a forecast-accuracy improvement, because on
the evidence it is not one.
