"""Forecaster registry. Each returns a callable (ctx, horizon) -> (quantiles, mean)."""
from __future__ import annotations

from typing import Callable

import numpy as np


def persistence() -> Callable:
    """ŷ_{t+h} = y_t for all h."""
    def fn(ctx: np.ndarray, horizon: int):
        val = ctx[-1]
        mean = np.full(horizon, val)
        # Degenerate quantiles (no uncertainty).
        quants = np.repeat(mean[:, None], 3, axis=1)
        return quants, mean
    return fn


def seasonal_naive(period: int) -> Callable:
    """ŷ_{t+h} = y_{t+h-period}. Requires context >= period."""
    def fn(ctx: np.ndarray, horizon: int):
        assert len(ctx) >= period, f"context shorter than period {period}"
        mean = np.array([ctx[-period + (h % period)] for h in range(horizon)])
        quants = np.repeat(mean[:, None], 3, axis=1)
        return quants, mean
    return fn


def chronos_bolt(model_name: str, device: str = "cuda", dtype: str = "bfloat16") -> Callable:
    """Batched Chronos-Bolt forecaster. Accepts (N, C) arrays."""
    import torch
    from chronos import BaseChronosPipeline
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    pipe = BaseChronosPipeline.from_pretrained(model_name, device_map=device, torch_dtype=dt)

    def fn(ctx: np.ndarray, horizon: int):
        t = torch.tensor(ctx, dtype=torch.float32)  # (N, C) or (C,)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        quants, mean = pipe.predict_quantiles(t, prediction_length=horizon,
                                              quantile_levels=[0.1, 0.5, 0.9])
        q = quants.float().cpu().numpy()
        m = mean.float().cpu().numpy()
        if ctx.ndim == 1:
            return q[0], m[0]
        return q, m  # (N, H, Q), (N, H)

    fn.batched = True  # type: ignore[attr-defined]
    return fn


def chronos_t5(model_name: str, device: str = "cuda", dtype: str = "bfloat16",
               num_samples: int = 20) -> Callable:
    """Original Chronos (T5-based). Sample-based probabilistic forecast."""
    import torch
    from chronos import ChronosPipeline
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    pipe = ChronosPipeline.from_pretrained(model_name, device_map=device, torch_dtype=dt)

    def fn(ctx: np.ndarray, horizon: int):
        t = torch.tensor(ctx, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        samples = pipe.predict(t, prediction_length=horizon, num_samples=num_samples)
        samples = samples.float().cpu().numpy()   # (N, S, H)
        mean = samples.mean(axis=1)               # (N, H)
        quants = np.quantile(samples, [0.1, 0.5, 0.9], axis=1).transpose(1, 2, 0)  # (N, H, Q)
        if ctx.ndim == 1:
            return quants[0], mean[0]
        return quants, mean

    fn.batched = True  # type: ignore[attr-defined]
    return fn


def moirai_zs(model_id: str = "Salesforce/moirai-1.1-R-large",
              context_length: int = 2048, num_samples: int = 20,
              device: str = "cuda") -> Callable:
    """Moirai zero-shot forecaster. Univariate."""
    import torch as _t
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    module = MoiraiModule.from_pretrained(model_id)
    model = MoiraiForecast(
        module=module,
        prediction_length=24,      # overridden per-call via .hparams
        context_length=context_length,
        patch_size="auto",
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    model = model.to(device).to(_t.bfloat16)
    model.eval()

    def fn(ctx: np.ndarray, horizon: int):
        model.hparams.prediction_length = horizon
        t = _t.tensor(ctx, dtype=_t.float32, device=device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        B, C = t.shape
        with _t.no_grad():
            past = t.to(_t.bfloat16).view(B, 1, C)              # (B, target_dim=1, C)
            past_mask = _t.ones_like(past, dtype=_t.bool)
            feat = _t.zeros(B, 1, C, 0, device=device, dtype=_t.bfloat16)
            samples = model(
                past_target=past,
                past_observed_target=past_mask,
                past_is_pad=_t.zeros(B, C, dtype=_t.bool, device=device),
                feat_dynamic_real=feat,
                observed_feat_dynamic_real=_t.ones(B, 1, C + horizon, 0, device=device, dtype=_t.bool),
                past_feat_dynamic_real=feat,
                past_observed_feat_dynamic_real=_t.ones_like(feat, dtype=_t.bool),
                num_samples=num_samples,
            )  # (B, num_samples, prediction_length)
        samples_np = samples.float().cpu().numpy()
        mean = samples_np.mean(axis=1)                          # (B, H)
        q = np.quantile(samples_np, [0.1, 0.5, 0.9], axis=1).transpose(1, 2, 0)  # (B, H, Q)
        if ctx.ndim == 1:
            return q[0], mean[0]
        return q, mean

    fn.batched = True  # type: ignore[attr-defined]
    return fn


def tirex_zs(device: str = "cuda") -> Callable:
    """TiRex zero-shot forecaster (xLSTM-based SOTA on HF benchmarks)."""
    from tirex import load_model
    m = load_model("NX-AI/TiRex", device=device)

    def fn(ctx: np.ndarray, horizon: int):
        import torch as _t
        t = _t.tensor(ctx, dtype=_t.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        # TiRex outputs 9 quantiles by default. We subselect 0.1/0.5/0.9.
        quants, mean = m.forecast(t, prediction_length=horizon, output_type="torch")
        q = quants.float().cpu().numpy()   # (N, H, Q)
        mn = mean.float().cpu().numpy()    # (N, H)
        # Subselect quantiles at 0.1, 0.5, 0.9. TiRex default is
        # [0.1, 0.2, ..., 0.9] — indices 0, 4, 8.
        q3 = q[..., [0, 4, 8]]
        if ctx.ndim == 1:
            return q3[0], mn[0]
        return q3, mn

    fn.batched = True  # type: ignore[attr-defined]
    return fn


MODELS = {
    "persistence": lambda cfg: persistence(),
    "seasonal_naive_24": lambda cfg: seasonal_naive(24),
    "seasonal_naive_168": lambda cfg: seasonal_naive(168),
    "chronos_bolt_tiny":  lambda cfg: chronos_bolt("amazon/chronos-bolt-tiny"),
    "chronos_bolt_mini":  lambda cfg: chronos_bolt("amazon/chronos-bolt-mini"),
    "chronos_bolt_small": lambda cfg: chronos_bolt("amazon/chronos-bolt-small"),
    "chronos_bolt_base":  lambda cfg: chronos_bolt("amazon/chronos-bolt-base"),
    "chronos_t5_small":   lambda cfg: chronos_t5("amazon/chronos-t5-small"),
    "chronos_t5_base":    lambda cfg: chronos_t5("amazon/chronos-t5-base"),
    "chronos_t5_large":   lambda cfg: chronos_t5("amazon/chronos-t5-large"),
    "tirex_zs":           lambda cfg: tirex_zs(),
    "moirai_zs":          lambda cfg: moirai_zs(
                              model_id=cfg.get("model_id", "Salesforce/moirai-1.1-R-large"),
                              context_length=cfg.get("context", 2048)),
}


def get(name: str, cfg: dict | None = None):
    cfg = cfg or {}
    # Allow arbitrary HF checkpoint path via cfg["path"].
    if name == "chronos_bolt_ft" and "path" in cfg:
        return chronos_bolt(cfg["path"])
    if name == "chronos_t5_ft" and "path" in cfg:
        return chronos_t5(cfg["path"])
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. known: {sorted(MODELS)}")
    return MODELS[name](cfg)
