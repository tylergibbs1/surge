"""Fine-tune Chronos-2 with covariates on multi-BA load.

Uses Chronos-2's official `.fit()` API. For each BA we build one training
task (one dict) containing:
  target            full train series
  past_covariates   temp + calendar, full train series
  future_covariates calendar keys only, declared with empty arrays
    (the trainer slices deterministic future-calendar windows internally).

For checkpoint validation, one input is built for each of 90 shared complete
rolling 2024 origins immediately preceding the 90-origin promotion cohort.
Chronos 2.3.1 scores only each input's trailing 24-hour label window, so
checkpoint selection cannot be dominated by training targets, one unusually
favorable final day, or the promotion windows themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.eval_c2 import (
    rolling_eval_c2,
    select_shared_complete_origin_schedule,
)
from experiments.features import load_multi_ba
from experiments.overfit import (
    FROZEN_CHRONOS_FIT_RUNTIME,
    POLICY_VERSION,
    TRUST_RTO_BAS,
    build_overfit_audit,
    configure_reproducible_runtime,
    rejected_overfit_audit,
    reproducibility_environment_versions,
    reproducibility_runtime_identity,
    revision_for_model_load,
    summarize_training_history,
    validate_frozen_data_snapshot,
    validate_h100_runtime,
    validate_promotion_inputs,
    validate_release_lineage,
    validate_reproducibility_runtime,
    validation_only_view,
    verify_code_checkout,
    verify_data_snapshot_manifest,
)
from surge.features import (
    LOAD_V2_CORE,
    AvailabilityMode,
    build_training_task,
)
from surge.model_loader import artifact_sha256, load_chronos2


def _task(bd, start: int, end: int, *, prediction_length: int = 24) -> dict:
    return build_training_task(
        bd,
        start=start,
        end=end,
        prediction_length=prediction_length,
        spec=LOAD_V2_CORE,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _trainer_audit_callback() -> Any:
    """Create a lightweight Transformers callback without a module-level dependency."""
    from transformers import TrainerCallback

    class AuditCallback(TrainerCallback):
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []
            self.final_state: dict[str, Any] = {}

        def on_log(self, args, state, control, logs=None, **kwargs):
            del args, control, kwargs
            values = logs or {}
            event: dict[str, Any] = {"event": "log", "step": state.global_step}
            for source, target in (
                ("loss", "train_loss"),
                ("eval_loss", "eval_loss"),
                ("learning_rate", "learning_rate"),
                ("epoch", "epoch"),
            ):
                if source in values:
                    event[target] = values[source]
            if len(event) > 2:
                self.events.append(event)

        def on_save(self, args, state, control, **kwargs):
            del args, control, kwargs
            self.events.append({"event": "checkpoint", "step": state.global_step})

        def on_train_end(self, args, state, control, **kwargs):
            del args, control, kwargs
            self.final_state = {
                "global_step": state.global_step,
                "best_metric": state.best_metric,
                "best_model_checkpoint": state.best_model_checkpoint,
            }

    return AuditCallback()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="amazon/chronos-2")
    ap.add_argument(
        "--base-model-id",
        default=os.environ.get("SURGE_BASE_MODEL_ID", "amazon/chronos-2"),
        help="portable upstream model ID recorded in a LoRA adapter",
    )
    ap.add_argument(
        "--base-revision",
        default=os.environ.get("SURGE_BASE_MODEL_REVISION", "unknown"),
    )
    ap.add_argument("--bas", nargs="+", default=list(TRUST_RTO_BAS))
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--mode", choices=["full", "lora"], default="lora")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num-steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument(
        "--diagnostic-origins",
        type=int,
        default=90,
        help=(
            "shared complete daily origins per promotion split; the same count "
            "of earlier 2024 origins is reserved for checkpoint selection"
        ),
    )
    ap.add_argument("--diagnostic-step", type=int, default=24)
    ap.add_argument("--diagnostic-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--with-generation",
        action="store_true",
        help="opt in to observed wind/solar as past-only covariates",
    )
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--code-revision", default=os.environ.get("SURGE_CODE_REVISION", "unknown"))
    ap.add_argument(
        "--data-snapshot-sha256",
        default=os.environ.get("SURGE_DATA_SNAPSHOT_SHA256", "unknown"),
    )
    args = ap.parse_args()
    print(f"[args] {vars(args)}", flush=True)
    if args.diagnostic_origins < 1:
        raise ValueError("diagnostic-origins must be positive")
    if args.diagnostic_step < 1 or args.diagnostic_batch < 1:
        raise ValueError("diagnostic-step and diagnostic-batch must be positive")
    args.bas = validate_promotion_inputs(
        args.bas,
        base_revision=args.base_revision,
        code_revision=args.code_revision,
        data_snapshot_sha256=args.data_snapshot_sha256,
    )
    validate_release_lineage(
        args.base,
        base_model_id=args.base_model_id,
        base_revision=args.base_revision,
    )
    validate_frozen_data_snapshot(args.data_snapshot_sha256)
    verify_code_checkout(Path(__file__).resolve().parents[1], args.code_revision)
    data_root_value = os.environ.get("SURGE_DATA_DIR")
    if not data_root_value:
        raise ValueError("promotion training requires SURGE_DATA_DIR")
    verify_data_snapshot_manifest(Path(data_root_value), args.data_snapshot_sha256)
    configure_reproducible_runtime(args.seed)
    environment_versions = reproducibility_environment_versions()
    runtime_system = reproducibility_runtime_identity()
    validate_h100_runtime(runtime_system)
    runtime_identity = {
        "system": runtime_system,
        "chronos_fit": {
            **FROZEN_CHRONOS_FIT_RUNTIME,
            "seed": args.seed,
            "data_seed": args.seed,
        },
    }
    validate_reproducibility_runtime(runtime_identity)
    out_dir = Path(args.out)
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing pre-existing output directory {out_dir}; use a new candidate path"
        ) from exc
    selection_valid_before = datetime(2025, 1, 1, tzinfo=UTC)
    loaded_bas = load_multi_ba(
        args.bas,
        with_gen=args.with_generation,
        availability_mode=AvailabilityMode.RETROSPECTIVE_FINAL,
        valid_before=selection_valid_before,
    )
    bas = {ba: validation_only_view(data) for ba, data in loaded_bas.items()}
    print(f"[data] loaded BAs: {list(bas)}", flush=True)
    print(
        "[data] locked test rows materialized: 0 "
        f"(valid-time cutoff {selection_valid_before.isoformat()})",
        flush=True,
    )

    checkpoint_validation_schedule = select_shared_complete_origin_schedule(
        bas,
        on="val",
        context=args.context,
        horizon=args.horizon,
        step=args.diagnostic_step,
        origin_count=args.diagnostic_origins,
        exclude_latest_complete=args.diagnostic_origins,
    )
    promotion_train_schedule = select_shared_complete_origin_schedule(
        bas,
        on="train",
        context=args.context,
        horizon=args.horizon,
        step=args.diagnostic_step,
        origin_count=args.diagnostic_origins,
    )
    promotion_validation_schedule = select_shared_complete_origin_schedule(
        bas,
        on="val",
        context=args.context,
        horizon=args.horizon,
        step=args.diagnostic_step,
        origin_count=args.diagnostic_origins,
    )

    train_inputs = [
        _task(bd, 0, bd.train_end, prediction_length=args.horizon) for bd in bas.values()
    ]
    val_inputs = [
        _task(
            bd,
            origin - args.context,
            origin + args.horizon,
            prediction_length=args.horizon,
        )
        for bd in bas.values()
        for origin in checkpoint_validation_schedule.indices_for(bd)
    ]
    print(f"[data] train tasks: {len(train_inputs)} | val tasks: {len(val_inputs)}", flush=True)

    if args.mode == "lora":
        try:
            import peft  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("LoRA mode requires peft; install surge-grid[train]") from exc

    pipe = load_chronos2(
        args.base,
        revision=revision_for_model_load(args.base, args.base_revision),
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    print(
        f"[model] chronos-2 loaded, params: "
        f"{sum(p.numel() for p in pipe.model.parameters()) / 1e6:.1f}M",
        flush=True,
    )

    t0 = time.time()
    audit_callback = _trainer_audit_callback()
    candidate_pipe = pipe.fit(
        inputs=train_inputs,
        validation_inputs=val_inputs,
        prediction_length=args.horizon,
        context_length=args.context,
        finetune_mode=args.mode,
        learning_rate=args.lr,
        num_steps=args.num_steps,
        batch_size=args.batch,
        output_dir=str(out_dir),
        finetuned_ckpt_name="candidate-unpromoted",
        callbacks=[audit_callback],
        disable_data_parallel=True,
        bf16=True,
        tf32=True,
        full_determinism=True,
        seed=args.seed,
        data_seed=args.seed,
    )
    elapsed = time.time() - t0
    candidate = out_dir / "candidate-unpromoted"
    diagnostic_bas = {ba: bas[ba] for ba in TRUST_RTO_BAS}
    diagnostics_error: str | None = None
    try:
        if not candidate.is_dir() or not any(candidate.iterdir()):
            raise RuntimeError(
                "Chronos fit completed without a saved candidate checkpoint"
            )
        adapter_configs = list(candidate.rglob("adapter_config.json"))
        if args.mode == "lora" and len(adapter_configs) != 1:
            raise RuntimeError(
                "LoRA saved artifact must contain exactly one adapter_config.json"
            )
        if args.mode == "lora":
            adapter_config_path = adapter_configs[0]
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            adapter_config["base_model_name_or_path"] = args.base_model_id
            adapter_config["revision"] = args.base_revision
            _write_json(adapter_config_path, adapter_config)

        train_metrics = rolling_eval_c2(
            candidate_pipe,
            diagnostic_bas,
            on="train",
            context=args.context,
            horizon=args.horizon,
            step=args.diagnostic_step,
            batch_size=args.diagnostic_batch,
            require_complete_origins=True,
            origin_schedule=promotion_train_schedule,
        )
        validation_metrics = rolling_eval_c2(
            candidate_pipe,
            diagnostic_bas,
            on="val",
            context=args.context,
            horizon=args.horizon,
            step=args.diagnostic_step,
            batch_size=args.diagnostic_batch,
            require_complete_origins=True,
            origin_schedule=promotion_validation_schedule,
            emit_origin_metrics=True,
        )
        baseline_validation_metrics = rolling_eval_c2(
            pipe,
            diagnostic_bas,
            on="val",
            context=args.context,
            horizon=args.horizon,
            step=args.diagnostic_step,
            batch_size=args.diagnostic_batch,
            require_complete_origins=True,
            origin_schedule=promotion_validation_schedule,
            emit_origin_metrics=True,
        )
        audit = build_overfit_audit(
            train_metrics,
            validation_metrics,
            baseline_validation_metrics,
            training_events=audit_callback.events,
            final_state=audit_callback.final_state,
            expected_steps=args.num_steps,
        )
    except Exception as exc:
        diagnostics_error = f"{type(exc).__name__}: {exc}"
        audit = rejected_overfit_audit(
            diagnostics_error,
            expected_steps=args.num_steps,
        )
        audit["training_behavior"] = summarize_training_history(
            audit_callback.events,
            final_state=audit_callback.final_state,
            expected_steps=args.num_steps,
        )
    audit["diagnostic_config"] = {
        "bas": list(TRUST_RTO_BAS),
        "context": args.context,
        "horizon": args.horizon,
        "step": args.diagnostic_step,
        "max_origins_per_split": args.diagnostic_origins,
        "batch_size": args.diagnostic_batch,
        "complete_target_origins_only": True,
        "shared_origin_schedule": True,
        "origin_filter_timing": "before trailing-origin limit and model inference",
        "train_origin_policy": (
            "latest complete-target origins strictly before 2024-01-01"
        ),
        "validation_origin_policy": "latest complete-target origins within 2024",
        "baseline": {
            "model": args.base_model_id,
            "revision": args.base_revision,
            "split": "same 2024 validation origins as candidate",
        },
        "test_opened": False,
    }
    audit["checkpoint_selection_schedule"] = (
        checkpoint_validation_schedule.as_dict()
    )
    audit["promotion_train_schedule"] = promotion_train_schedule.as_dict()
    audit["promotion_validation_schedule"] = (
        promotion_validation_schedule.as_dict()
    )
    audit["identity"] = {
        "base_model": args.base_model_id,
        "base_revision": args.base_revision,
        "code_revision": args.code_revision,
        "data_snapshot_sha256": args.data_snapshot_sha256,
        "feature_spec_version": LOAD_V2_CORE.version,
        "feature_spec_sha256": LOAD_V2_CORE.sha256,
        "bas": list(TRUST_RTO_BAS),
    }
    audit["environment"] = environment_versions
    audit["runtime"] = runtime_identity
    audit_path = out_dir / "surge-overfit-audit.json"
    _write_json(audit_path, audit)

    promoted = audit["promotion_eligible"] is True
    best = out_dir / "best"
    checkpoint = candidate
    if promoted:
        candidate.replace(best)
        checkpoint = best

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "code_revision": args.code_revision,
        "data_snapshot_sha256": args.data_snapshot_sha256,
        "base_model": args.base_model_id,
        "base_model_source": args.base,
        "base_model_revision": args.base_revision,
        "feature_spec_version": LOAD_V2_CORE.version,
        "feature_spec_sha256": LOAD_V2_CORE.sha256,
        "availability_mode": AvailabilityMode.RETROSPECTIVE_FINAL.value,
        "point_in_time_replay": False,
        "locked_test_opened": False,
        "selection_valid_before_utc": selection_valid_before.isoformat(),
        "bas": list(bas),
        "config": {
            "context": args.context,
            "horizon": args.horizon,
            "mode": args.mode,
            "learning_rate": args.lr,
            "num_steps": args.num_steps,
            "batch_size": args.batch,
            "diagnostic_origins": args.diagnostic_origins,
            "diagnostic_step": args.diagnostic_step,
            "diagnostic_batch_size": args.diagnostic_batch,
            "complete_target_origins_only": True,
            "shared_origin_schedule": True,
            "checkpoint_selection_disjoint_from_promotion": True,
            "seed": args.seed,
            "with_generation": args.with_generation,
        },
        "versions": environment_versions,
        "runtime": runtime_identity,
        "data": {
            ba: {
                "rows": len(data.target),
                "train_end": data.train_end,
                "validation_end": data.val_end,
                "locked_test_rows_materialized": 0,
                "selection_valid_before_utc": selection_valid_before.isoformat(),
                "start_utc": str(data.ts_utc[0]),
                "selection_end_utc": str(data.ts_utc[-1]),
                "missing_target": int(np.isnan(data.target).sum()),
                "missing_temperature": int(np.isnan(data.covariates["temp_c"]).sum()),
                "provenance": data.provenance,
            }
            for ba, data in bas.items()
        },
        "wall_seconds": round(elapsed, 3),
        "selection": {
            "policy_version": POLICY_VERSION,
            "promotion_eligible": promoted,
            "overfit_audit": audit_path.name,
            "overfit_audit_sha256": _sha256(audit_path),
            "checkpoint_state": "promoted" if promoted else "candidate-rejected",
            "diagnostics_error": diagnostics_error,
        },
        "artifact_files": _artifact_files(checkpoint),
        "model_artifact_hash_algorithm": "sha256-tree-v1",
        "model_artifact_sha256": artifact_sha256(checkpoint),
    }
    manifest_path = out_dir / "surge-training-manifest.json"
    _write_json(manifest_path, manifest)

    if not promoted:
        print(
            "FINETUNE_REJECTED:",
            json.dumps(
                {
                    "out": str(out_dir),
                    "candidate": str(candidate),
                    "manifest": str(manifest_path),
                    "audit": str(audit_path),
                    "rejection_reasons": audit["rejection_reasons"],
                    "test_opened": False,
                }
            ),
            flush=True,
        )
        raise RuntimeError("candidate failed the v0.2 overfitting promotion gate")

    promotion = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "promotion_eligible": True,
        "checkpoint": best.name,
        "manifest": manifest_path.name,
        "manifest_sha256": _sha256(manifest_path),
        "overfit_audit": audit_path.name,
        "overfit_audit_sha256": _sha256(audit_path),
        "model_artifact_hash_algorithm": manifest[
            "model_artifact_hash_algorithm"
        ],
        "model_artifact_sha256": manifest["model_artifact_sha256"],
        "test_opened": False,
    }
    promotion_path = out_dir / "surge-promotion.json"
    _write_json(promotion_path, promotion)

    print(
        "FINETUNE_DONE:",
        json.dumps(
            {
                "wall_s": round(elapsed, 1),
                "out": str(out_dir),
                "best": str(best),
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "overfit_audit": str(audit_path),
                "overfit_audit_sha256": _sha256(audit_path),
                "promotion": str(promotion_path),
                "promotion_sha256": _sha256(promotion_path),
                "test_opened": False,
                "feature_spec_version": LOAD_V2_CORE.version,
                "feature_spec_sha256": LOAD_V2_CORE.sha256,
                "availability_mode": AvailabilityMode.RETROSPECTIVE_FINAL.value,
            }
        ),
    )


if __name__ == "__main__":
    main()
