"""Evaluation runner for Chronos-2 with covariates.

    python -m experiments.run_c2 <exp_name> <config_json>

Config keys:
    base          str   HF model id or local path
    bas           list[str]  BAs to evaluate on
    context       int   2048
    horizon       int   24
    on            "train" | "val" | "test"
    step          int   origin stride in hours (default 24)
    max_origins   int | null  keep only the latest N origins in the split
    test_protocol "locked-once" for test only
    promotion_artifact str  checksummed eligible marker required for test
    selection_artifact str  frozen two-candidate winner required for test
    base_revision str  immutable upstream commit; omitted from local model load
    code_revision str  must match the promoted manifest for test
    data_snapshot_sha256 str  must match the promoted manifest for test
    bootstrap     int   0 for none, else n resamples
    batch_size    int   16

The locked test atomically creates ``surge-locked-test-receipt.json`` next to
the promotion marker before any 2025 rows are loaded. A catchable exception
after reservation records a terminal failure in both receipt and registry; an
abrupt process loss leaves ``started``. Either state consumes the run.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.eval_c2 import rolling_eval_c2
from experiments.features import load_multi_ba
from experiments.overfit import (
    complete_locked_test_run,
    configure_reproducible_runtime,
    fail_locked_test_run,
    reproducibility_environment_versions,
    reproducibility_runtime_identity,
    reserve_locked_test_run,
    revision_for_model_load,
    validate_h100_runtime,
    verify_code_checkout,
    verify_data_snapshot_manifest,
    verify_promotion_artifact,
    verify_selection_artifact,
)
from scripts.rebuild_data_snapshot import verify_snapshot
from surge import bas as _bas
from surge.features import (
    LOAD_V2_CORE,
    POINT_ESTIMATE_KIND,
    POINT_ESTIMATE_LABEL,
    AvailabilityMode,
)
from surge.model_loader import load_chronos2


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _record_and_emit_result(
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    *,
    load_s: float,
    eval_s: float,
    locked_test_receipt: Path | None,
) -> None:
    """Persist raw locked-test values, then print a rounded display copy."""
    raw_result = {
        **metadata,
        "load_s": load_s,
        "eval_s": eval_s,
        **metrics,
    }
    if locked_test_receipt is not None:
        complete_locked_test_run(locked_test_receipt, raw_result)

    display_result = {
        k: round(v, 4)
        if isinstance(v, (float, int)) and not isinstance(v, bool)
        else v
        for k, v in raw_result.items()
        if k != "per_ba"
    }
    display_result["load_s"] = round(load_s, 2)
    display_result["eval_s"] = round(eval_s, 2)
    if "per_ba" in raw_result:
        display_result["per_ba"] = {
            ba: {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in stats.items()
            }
            for ba, stats in raw_result["per_ba"].items()
        }
    print("METRIC:", json.dumps(display_result, allow_nan=False), flush=True)


def _main(on_locked_test_reserved: Callable[[Path], None]) -> None:
    import torch

    exp_name = sys.argv[1]
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

    base = cfg.get("base", "amazon/chronos-2")
    if not isinstance(base, str) or not base:
        raise ValueError("base must be a model ID or local path string")
    bas_list = cfg.get("bas", list(_bas.demand_codes()))
    if not isinstance(bas_list, list) or not all(
        isinstance(ba, str) for ba in bas_list
    ):
        raise ValueError("bas must be a JSON list of RTO strings")
    on = cfg.get("on", "val")
    if on not in {"train", "val", "test"}:
        raise ValueError("on must be 'train', 'val', or 'test'")
    context = _positive_integer("context", cfg.get("context", 2048))
    horizon = _positive_integer("horizon", cfg.get("horizon", 24))
    batch_size = _positive_integer("batch_size", cfg.get("batch_size", 16))
    step = _positive_integer("step", cfg.get("step", 24))
    bootstrap = cfg.get("bootstrap", 2_000 if on == "test" else 0)
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap must be a non-negative integer")
    seed = cfg.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    max_origins = cfg.get("max_origins")
    if max_origins is not None:
        max_origins = _positive_integer("max_origins", max_origins)
    with_generation = cfg.get("with_gen", False)
    if not isinstance(with_generation, bool):
        raise ValueError("with_gen must be a boolean")
    per_step = cfg.get("per_step", on == "test")
    if not isinstance(per_step, bool):
        raise ValueError("per_step must be a boolean")
    base_revision = cfg.get(
        "base_revision", os.environ.get("SURGE_BASE_MODEL_REVISION", "unknown")
    )
    code_revision = cfg.get(
        "code_revision", os.environ.get("SURGE_CODE_REVISION", "unknown")
    )
    data_snapshot_sha256 = cfg.get(
        "data_snapshot_sha256",
        os.environ.get("SURGE_DATA_SNAPSHOT_SHA256", "unknown"),
    )
    configure_reproducible_runtime(seed)
    load_revision = revision_for_model_load(base, base_revision)
    current_versions = reproducibility_environment_versions()
    current_runtime_system = reproducibility_runtime_identity()
    if on == "test" and cfg.get("test_protocol") != "locked-once":
        raise ValueError("test evaluation requires test_protocol='locked-once'")
    promotion_artifact: dict | None = None
    selection_artifact: dict | None = None
    locked_test_receipt: Path | None = None
    locked_data_root: Path | None = None
    if on == "test":
        if (
            context != 2_048
            or horizon != 24
            or step != 24
            or max_origins is not None
            or bootstrap != 2_000
            or seed != 42
            or with_generation is not False
            or per_step is not True
        ):
            raise ValueError(
                "locked test requires context=2048, horizon=24, step=24, "
                "max_origins=null, bootstrap=2000, seed=42, with_gen=false, "
                "and per_step=true"
            )
        promotion_path = cfg.get("promotion_artifact")
        if not isinstance(promotion_path, str):
            raise ValueError("locked-once test requires promotion_artifact")
        selection_path = cfg.get("selection_artifact")
        if not isinstance(selection_path, str):
            raise ValueError("locked-once test requires selection_artifact")
        promotion_artifact = verify_promotion_artifact(
            Path(promotion_path),
            model_path=Path(base),
        )
        selection_artifact = verify_selection_artifact(
            Path(selection_path),
            promotion_path=Path(promotion_path),
            promotion=promotion_artifact,
        )
        identity = promotion_artifact["training_identity"]
        if current_versions != identity.get("versions"):
            raise ValueError(
                "locked-test dependency versions do not match promoted training"
            )
        validate_h100_runtime(current_runtime_system)
        training_runtime = identity.get("runtime")
        if (
            not isinstance(training_runtime, dict)
            or current_runtime_system != training_runtime.get("system")
        ):
            raise ValueError(
                "locked-test system/accelerator runtime does not match promoted training"
            )
        if list(bas_list) != identity["bas"]:
            raise ValueError(
                "locked-test RTOs must exactly match the canonical promoted RTO list"
            )
        for label, actual, expected in (
            ("base_revision", base_revision, identity["base_revision"]),
            ("code_revision", code_revision, identity["code_revision"]),
            (
                "data_snapshot_sha256",
                data_snapshot_sha256,
                identity["data_snapshot_sha256"],
            ),
        ):
            if actual != expected:
                raise ValueError(
                    f"locked-test {label} does not match the promoted training manifest"
                )
        if (
            identity.get("feature_spec_version") != LOAD_V2_CORE.version
            or identity.get("feature_spec_sha256") != LOAD_V2_CORE.sha256
        ):
            raise ValueError("locked-test feature spec does not match promoted training")
        training_config = identity.get("training_config")
        if not isinstance(training_config, dict) or any(
            actual != training_config.get(field)
            for field, actual in (
                ("context", context),
                ("horizon", horizon),
                ("with_generation", with_generation),
            )
        ):
            raise ValueError("locked-test model configuration does not match promoted training")
        if step != 24 or max_origins is not None:
            raise ValueError(
                "locked test requires the frozen full-partition daily-origin protocol"
            )
        verify_code_checkout(Path(__file__).resolve().parents[1], code_revision)
        data_root_value = os.environ.get("SURGE_DATA_DIR")
        if not data_root_value:
            raise ValueError("locked test requires SURGE_DATA_DIR")
        locked_data_root = verify_data_snapshot_manifest(
            Path(data_root_value), data_snapshot_sha256
        )
        registry_value = os.environ.get("SURGE_LOCKED_TEST_REGISTRY")
        if not registry_value:
            raise ValueError("locked test requires SURGE_LOCKED_TEST_REGISTRY")
        locked_test_receipt = reserve_locked_test_run(
            Path(selection_path),
            experiment=exp_name,
            training_identity=identity,
            selection_sha256=selection_artifact["selection_sha256"],
            selection_decision_sha256=selection_artifact[
                "selection_decision_sha256"
            ],
            experiment_protocol_sha256=selection_artifact[
                "experiment_protocol_sha256"
            ],
            promotion_path=Path(promotion_path),
            marker_sha256=promotion_artifact["marker_sha256"],
            checkpoint_inventory_sha256=promotion_artifact[
                "checkpoint_inventory_sha256"
            ],
            model_artifact_sha256=promotion_artifact["model_artifact_sha256"],
            registry_root=Path(registry_value),
        )
        on_locked_test_reserved(locked_test_receipt)
    if locked_data_root is not None:
        # Byte identity is checked before reservation; semantic verification is
        # intentionally after it because this step materializes locked rows.
        verify_snapshot(locked_data_root)
    availability_mode = AvailabilityMode.RETROSPECTIVE_FINAL
    selection_valid_before = (
        datetime(2025, 1, 1, tzinfo=UTC) if on in {"train", "val"} else None
    )
    bas = load_multi_ba(
        bas_list,
        with_gen=with_generation,
        availability_mode=availability_mode,
        valid_before=selection_valid_before,
    )

    t0 = time.time()
    pipe = load_chronos2(
        base,
        revision=load_revision,
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    load_s = time.time() - t0
    if on == "test":
        post_load_promotion = verify_promotion_artifact(
            Path(cfg["promotion_artifact"]),
            model_path=Path(base),
        )
        if (
            promotion_artifact is None
            or post_load_promotion["marker_sha256"]
            != promotion_artifact["marker_sha256"]
            or post_load_promotion["checkpoint_inventory_sha256"]
            != promotion_artifact["checkpoint_inventory_sha256"]
        ):
            raise RuntimeError("promoted checkpoint changed while the model was loading")

    t0 = time.time()
    m = rolling_eval_c2(
        pipe,
        bas,
        on=on,
        context=context,
        horizon=horizon,
        step=step,
        batch_size=batch_size,
        bootstrap=bootstrap,
        seed=seed,
        per_step=per_step,
        max_origins=max_origins,
    )
    eval_s = time.time() - t0

    metadata = {
        "exp": exp_name,
        "base": base,
        "bas": bas_list,
        "on": on,
        "context": context,
        "horizon": horizon,
        "step": step,
        "max_origins": max_origins,
        "batch_size": batch_size,
        "bootstrap": bootstrap,
        "seed": seed,
        "per_step": per_step,
        "test_protocol": cfg.get("test_protocol") if on == "test" else None,
        "locked_test_opened": on == "test",
        "selection_valid_before_utc": (
            selection_valid_before.isoformat()
            if selection_valid_before is not None
            else None
        ),
        "promotion_policy_version": (
            promotion_artifact.get("policy_version")
            if promotion_artifact is not None
            else None
        ),
        "promotion_marker_sha256": (
            promotion_artifact.get("marker_sha256")
            if promotion_artifact is not None
            else None
        ),
        "model_artifact_hash_algorithm": (
            promotion_artifact.get("model_artifact_hash_algorithm")
            if promotion_artifact is not None
            else None
        ),
        "model_artifact_sha256": (
            promotion_artifact.get("model_artifact_sha256")
            if promotion_artifact is not None
            else None
        ),
        "selection_policy_version": (
            selection_artifact.get("policy_version")
            if selection_artifact is not None
            else None
        ),
        "selection_artifact_sha256": (
            selection_artifact.get("selection_sha256")
            if selection_artifact is not None
            else None
        ),
        "selection_decision_sha256": (
            selection_artifact.get("selection_decision_sha256")
            if selection_artifact is not None
            else None
        ),
        "experiment_protocol_sha256": (
            selection_artifact.get("experiment_protocol_sha256")
            if selection_artifact is not None
            else None
        ),
        "locked_test_receipt": (
            locked_test_receipt.name if locked_test_receipt is not None else None
        ),
        "with_generation": with_generation,
        "base_revision": base_revision,
        "code_revision": code_revision,
        "data_snapshot_sha256": data_snapshot_sha256,
        "versions": current_versions,
        "runtime": {
            "system": current_runtime_system,
            "inference": {"device_map": "cuda", "dtype": "bfloat16"},
        },
        "feature_spec_version": LOAD_V2_CORE.version,
        "feature_spec_sha256": LOAD_V2_CORE.sha256,
        "availability_mode": availability_mode.value,
        "point_estimate_kind": POINT_ESTIMATE_KIND,
        "point_estimate_quantile": POINT_ESTIMATE_LABEL,
    }
    _record_and_emit_result(
        metadata,
        m,
        load_s=load_s,
        eval_s=eval_s,
        locked_test_receipt=locked_test_receipt,
    )


def main() -> None:
    """Run evaluation, terminally recording any post-reservation failure."""
    locked_test_receipt: Path | None = None

    def remember_locked_test_receipt(path: Path) -> None:
        nonlocal locked_test_receipt
        locked_test_receipt = path

    try:
        _main(remember_locked_test_receipt)
    except BaseException as exc:
        if locked_test_receipt is not None:
            try:
                fail_locked_test_run(locked_test_receipt, exc)
            except Exception as recording_exc:
                exc.add_note(
                    "locked-test failure recording also failed: "
                    f"{type(recording_exc).__name__}"
                )
        raise


if __name__ == "__main__":
    main()
