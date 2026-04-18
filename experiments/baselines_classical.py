"""Classical / per-hour regression baselines for day-ahead load forecast.

Models:
    xgb_hourly: XGBoost, 24 separate models (one per hour-of-day),
                following Roy et al. 2505.11390's "hourly-binned" recipe.
    prophet:    Prophet, one model per BA, with temperature regressor.
    nbeats:     N-BEATS from neuralforecast, one model fit per BA.

All share the eval protocol in `experiments.eval_c2.rolling_eval_c2`: rolling
24h forecasts at step=24 across val or test, MASE denominator = per-BA train
seasonal-naive (m=24).
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass

import numpy as np

from experiments.features import BAData, load_multi_ba

warnings.filterwarnings("ignore")


# --- Feature builder (shared across baselines) ----------------------------
def features_for_hour(bd: BAData, target_idx: int, lags: tuple[int, ...] = (24, 48, 168)) -> np.ndarray:
    """Returns a feature vector at absolute index `target_idx`.

    Features:
        y_{t-24}, y_{t-48}, y_{t-168}   (lag load)
        temp_t                           (assumed-known-at-forecast-time)
        hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, is_holiday
        month (1..12)
    """
    feats = []
    for lag in lags:
        feats.append(bd.target[target_idx - lag] if target_idx - lag >= 0 else 0.0)
    feats.append(bd.covariates["temp_c"][target_idx])
    for k in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday"):
        feats.append(bd.covariates[k][target_idx])
    # Month from timestamp
    ts = bd.ts_utc[target_idx].astype("datetime64[M]").astype(int) % 12 + 1
    feats.append(float(ts))
    return np.array(feats, dtype=np.float32)


def build_xy(bd: BAData, start: int, end: int, lags=(24, 48, 168)):
    """Build (X, y, hour_of_day) for [start, end). Skips indices where lags unavailable."""
    rows = []
    ys = []
    hours = []
    for i in range(max(start, max(lags)), end):
        if np.isnan(bd.target[i]):
            continue
        rows.append(features_for_hour(bd, i, lags))
        ys.append(bd.target[i])
        # Hour of day from ts
        ts_h = bd.ts_utc[i].astype("datetime64[h]").astype(int) % 24
        hours.append(ts_h)
    return np.stack(rows), np.array(ys, dtype=np.float32), np.array(hours, dtype=np.int32)


# --- XGBoost hourly-binned -------------------------------------------------
def xgb_hourly_eval(bd: BAData, on: str, horizon: int = 24, step: int = 24,
                    context: int = 168):
    import xgboost as xgb

    # Train one booster per hour-of-day using all train rows with that hour.
    X_tr, y_tr, h_tr = build_xy(bd, 0, bd.train_end)

    models = {}
    for h in range(24):
        mask = h_tr == h
        if mask.sum() < 100:
            continue
        m = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            tree_method="hist", device="cuda",
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0,
            verbosity=0,
        )
        m.fit(X_tr[mask], y_tr[mask])
        models[h] = m

    eval_start = bd.train_end if on == "val" else bd.val_end
    eval_end = bd.val_end if on == "val" else len(bd.target)
    abs_errs = []
    win_abs = []

    for origin in range(eval_start, eval_end - horizon + 1, step):
        preds, trues = [], []
        for h in range(horizon):
            i = origin + h
            if i < 24:
                continue
            x = features_for_hour(bd, i).reshape(1, -1)
            # For multi-step, use actual lags (still in the past relative to origin).
            hour_of_day = int(bd.ts_utc[i].astype("datetime64[h]").astype(int) % 24)
            model = models.get(hour_of_day)
            if model is None:
                pred = float(bd.target[i - 24])
            else:
                pred = float(model.predict(x)[0])
            truth = float(bd.target[i])
            preds.append(pred); trues.append(truth)

        preds = np.array(preds); trues = np.array(trues)
        err = np.abs(trues - preds)
        abs_errs.append(err)
        win_abs.append(err.mean())

    abs_errs = np.concatenate(abs_errs)
    mae = float(abs_errs.mean())
    return {"mae": mae, "mase": mae / bd.denom_mae, "win_mase": np.array(win_abs) / bd.denom_mae}


# --- Prophet ---------------------------------------------------------------
def prophet_eval(bd: BAData, on: str, horizon: int = 24, step: int = 24):
    import pandas as pd
    from prophet import Prophet

    ts_train = pd.to_datetime(bd.ts_utc[:bd.train_end])
    temp_train = bd.covariates["temp_c"][:bd.train_end]
    y_train = bd.target[:bd.train_end]
    df_tr = pd.DataFrame({"ds": ts_train, "y": y_train, "temp": temp_train})

    m = Prophet(changepoint_prior_scale=0.05, daily_seasonality=True,
                weekly_seasonality=True, yearly_seasonality=True,
                mcmc_samples=0, uncertainty_samples=0)
    m.add_regressor("temp")
    m.add_country_holidays(country_name="US")
    m.fit(df_tr)

    eval_start = bd.train_end if on == "val" else bd.val_end
    eval_end = bd.val_end if on == "val" else len(bd.target)
    abs_errs = []
    win_abs = []

    # Predict the whole eval range once, faster than per-origin.
    ts_ev = pd.to_datetime(bd.ts_utc[eval_start:eval_end])
    temp_ev = bd.covariates["temp_c"][eval_start:eval_end]
    df_ev = pd.DataFrame({"ds": ts_ev, "temp": temp_ev})
    fc = m.predict(df_ev)
    pred_full = fc["yhat"].to_numpy()

    for origin in range(eval_start, eval_end - horizon + 1, step):
        j = origin - eval_start
        preds = pred_full[j:j + horizon]
        trues = bd.target[origin:origin + horizon]
        err = np.abs(trues - preds)
        abs_errs.append(err)
        win_abs.append(err.mean())

    abs_errs = np.concatenate(abs_errs)
    mae = float(abs_errs.mean())
    return {"mae": mae, "mase": mae / bd.denom_mae, "win_mase": np.array(win_abs) / bd.denom_mae}


# --- Dispatcher + bootstrap CI --------------------------------------------
def nbeats_eval(bd: BAData, on: str, horizon: int = 24, step: int = 24,
                context: int = 168):
    """N-BEATS via neuralforecast. One model per BA, univariate."""
    import pandas as pd
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS

    ts = pd.to_datetime(bd.ts_utc[:bd.train_end])
    y = bd.target[:bd.train_end]
    df = pd.DataFrame({"unique_id": "0", "ds": ts, "y": y})

    model = NBEATS(
        input_size=context,
        h=horizon,
        max_steps=2000,
        batch_size=64,
        random_seed=42,
        enable_progress_bar=False,
        enable_checkpointing=False,
        stack_types=["identity", "trend", "seasonality"],
        n_blocks=[1, 1, 1],
    )
    nf = NeuralForecast(models=[model], freq="h")
    nf.fit(df, val_size=24 * 30)

    eval_start = bd.train_end if on == "val" else bd.val_end
    eval_end = bd.val_end if on == "val" else len(bd.target)
    abs_errs = []
    win_abs = []

    # Rolling forecasts: insert_newdata then predict step-by-step.
    for origin in range(eval_start, eval_end - horizon + 1, step):
        ts_hist = pd.to_datetime(bd.ts_utc[origin - context:origin])
        y_hist = bd.target[origin - context:origin]
        hist = pd.DataFrame({"unique_id": "0", "ds": ts_hist, "y": y_hist})
        fc = nf.predict(df=hist)
        pred_col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        preds = fc[pred_col].to_numpy()[:horizon]
        trues = bd.target[origin:origin + horizon]
        err = np.abs(trues - preds)
        abs_errs.append(err)
        win_abs.append(err.mean())

    abs_errs = np.concatenate(abs_errs)
    mae = float(abs_errs.mean())
    return {"mae": mae, "mase": mae / bd.denom_mae, "win_mase": np.array(win_abs) / bd.denom_mae}


RUNNERS = {
    "xgb_hourly": xgb_hourly_eval,
    "prophet": prophet_eval,
    "nbeats": nbeats_eval,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(RUNNERS))
    ap.add_argument("--bas", nargs="+",
                    default=["PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP"])
    ap.add_argument("--on", default="val")
    ap.add_argument("--with-gen", action="store_true")
    args = ap.parse_args()

    bas = load_multi_ba(args.bas, with_gen=args.with_gen)
    per_ba = {}
    all_win_mase = []
    for ba, bd in bas.items():
        t0 = time.time()
        m = RUNNERS[args.model](bd, args.on)
        per_ba[ba] = {"mase": round(m["mase"], 4), "mae": round(m["mae"], 1),
                      "time_s": round(time.time() - t0, 1)}
        all_win_mase.append(m["win_mase"])
        print(f"[{args.model}] {ba}: MASE {m['mase']:.4f}  MAE {m['mae']:.1f}  "
              f"({time.time() - t0:.1f}s)", flush=True)

    pooled = np.concatenate(all_win_mase)
    rng = np.random.default_rng(42)
    boots = np.array([pooled[rng.integers(0, len(pooled), len(pooled))].mean() for _ in range(2000)])
    macro_mase = float(np.mean([v["mase"] for v in per_ba.values()]))

    print("METRIC:", json.dumps({
        "model": args.model, "on": args.on,
        "mase_macro": round(macro_mase, 4),
        "mase_ci_low": round(float(np.quantile(boots, 0.025)), 4),
        "mase_ci_high": round(float(np.quantile(boots, 0.975)), 4),
        "per_ba": per_ba,
    }))


if __name__ == "__main__":
    main()
