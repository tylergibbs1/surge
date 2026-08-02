"""Auditable, fail-closed overfitting controls for v0.2 model selection.

This module is intentionally independent of Torch and Transformers so the
promotion policy can be unit-tested without loading a forecasting model.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from experiments.features import BAData
from surge.features import (
    POINT_ESTIMATE_KIND,
    POINT_ESTIMATE_LABEL,
    POINT_ESTIMATE_QUANTILE,
)
from surge.model_loader import artifact_sha256

POLICY_VERSION = "surge-v0.2-overfit-gate-v1"
TRUST_RTO_BAS = ("PJM", "CISO", "ERCO", "MISO", "NYIS", "ISNE", "SWPP")
RELEASE_BASE_MODEL_ID = "amazon/chronos-2"
RELEASE_BASE_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
FROZEN_DATA_SNAPSHOT_SHA256 = (
    "77d80d4031e2391808103ef29bb182b3ee2469cec1c24ae00569d217bd48a4c0"
)
SELECTION_POLICY_VERSION = "surge-v0.2-frozen-h100-selection-v1"
SELECTION_TRAINING_STEPS = (1_000, 2_000)
DIAGNOSTIC_HORIZON = 24
FROZEN_CHRONOS_FIT_RUNTIME = {
    "bf16": True,
    "tf32": True,
    "full_determinism": True,
    "seed": 42,
    "data_seed": 42,
    "disable_data_parallel": True,
}
PAIRED_VALIDATION_BOOTSTRAP_SAMPLES = 2_000
PAIRED_VALIDATION_BOOTSTRAP_SEED = 42
PAIRED_VALIDATION_BOOTSTRAP_BLOCK_ORIGINS = 7
_EPSILON = 1e-12


@dataclass(frozen=True)
class PromotionThresholds:
    """Conservative v0.2 governance thresholds; changing them changes policy."""

    required_diagnostic_windows_per_ba: int = 90
    max_validation_macro_mase: float = 1.0
    max_macro_mase_generalization_ratio: float = 1.75
    max_macro_wis_generalization_ratio: float = 1.75
    max_worst_rto_validation_mase: float = 1.25
    max_worst_rto_mase_generalization_ratio: float = 2.25
    max_validation_mase_cv: float = 0.40
    max_validation_mase_vs_baseline_ratio: float = 1.0
    max_validation_wis_vs_baseline_ratio: float = 1.0
    max_worst_rto_mase_vs_baseline_ratio: float = 1.10
    max_paired_bootstrap_mase_ratio_upper_95: float = 1.05
    min_train_loss_logs: int = 2
    min_eval_checkpoints: int = 2
    max_eval_loss_rebound_ratio: float = 1.25


DEFAULT_THRESHOLDS = PromotionThresholds()
ELIGIBLE_GATE_NAMES = {
    "diagnostic_windows_per_ba",
    "validation_macro_mase",
    "macro_mase_generalization_ratio",
    "macro_wis_generalization_ratio",
    "worst_rto_validation_mase",
    "worst_rto_mase_generalization_ratio",
    "validation_mase_cv",
    "validation_mase_vs_baseline_ratio",
    "validation_wis_vs_baseline_ratio",
    "worst_rto_mase_vs_baseline_ratio",
    "paired_bootstrap_mase_ratio_upper_95",
    "training_completed_steps",
    "train_loss_log_count",
    "eval_checkpoint_count",
    "eval_loss_rebound_ratio",
    "reported_best_matches_observed",
    "reported_best_checkpoint_was_saved",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    """Publish complete JSON with an atomic no-replace hard link."""
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_json_atomically(path: Path, value: dict[str, Any]) -> None:
    """Replace one JSON record atomically after fsyncing its complete payload."""
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                default=str,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_locked_test_terminal_extension(
    started: dict[str, Any],
    terminal: dict[str, Any],
) -> None:
    """Validate an authoritative terminal record before repairing its receipt."""
    for key, value in started.items():
        if key != "status" and (key not in terminal or terminal[key] != value):
            raise ValueError("locked-test terminal record changed its reservation identity")
    status = terminal.get("status")
    if status == "completed":
        expected_extra = {"completed_at_utc", "result_sha256", "result"}
        result = terminal.get("result")
        if not isinstance(result, dict):
            raise ValueError("locked-test result payload is malformed")
        encoded_payload = json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
            allow_nan=False,
        ).encode()
        payload_sha256 = terminal.get("result_sha256")
        timestamp_key = "completed_at_utc"
    elif status == "failed":
        expected_extra = {"failed_at_utc", "failure_sha256", "failure"}
        failure = terminal.get("failure")
        if (
            not isinstance(failure, dict)
            or set(failure) != {"exception_type", "message_omitted"}
            or not isinstance(failure.get("exception_type"), str)
            or not failure["exception_type"]
            or len(failure["exception_type"]) > 128
            or re.fullmatch(r"[A-Za-z0-9_.-]+", failure["exception_type"]) is None
            or failure.get("message_omitted") is not True
        ):
            raise ValueError("locked-test failure payload is malformed")
        encoded_payload = json.dumps(
            failure,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        payload_sha256 = terminal.get("failure_sha256")
        timestamp_key = "failed_at_utc"
    else:
        raise ValueError("locked-test authoritative record has an invalid terminal state")
    if set(terminal) - set(started) != expected_extra:
        raise ValueError("locked-test terminal record has unexpected fields")
    timestamp = terminal.get(timestamp_key)
    if not isinstance(timestamp, str) or datetime.fromisoformat(timestamp).tzinfo is None:
        raise ValueError("locked-test terminal record has an invalid timestamp")
    if payload_sha256 != hashlib.sha256(encoded_payload).hexdigest():
        raise ValueError("locked-test terminal payload checksum does not match")


def _locked_test_records(receipt_path: Path) -> tuple[dict[str, Any], Path]:
    """Load records, repairing a lagging receipt from the authoritative registry."""
    path = receipt_path.expanduser().resolve()
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("locked-test receipt must contain a JSON object")
    reservation_value = receipt.get("registry_reservation")
    if not isinstance(reservation_value, str):
        raise ValueError("locked-test receipt is missing its registry reservation")
    reservation_path = Path(reservation_value).expanduser().resolve()
    if not reservation_path.is_file():
        raise ValueError("locked-test registry reservation is missing")
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    if not isinstance(reservation, dict):
        raise ValueError("locked-test registry reservation must contain a JSON object")
    if reservation == receipt:
        return receipt, reservation_path
    if receipt.get("status") != "started":
        raise ValueError("locked-test receipt and registry reservation do not match")
    _validate_locked_test_terminal_extension(receipt, reservation)
    _replace_json_atomically(path, reservation)
    repaired = json.loads(path.read_text(encoding="utf-8"))
    if repaired != reservation:
        raise ValueError("locked-test receipt reconciliation did not persist")
    return reservation, reservation_path


def _terminalize_locked_test_run(
    receipt_path: Path,
    terminal_fields: dict[str, Any],
) -> None:
    """Publish the same immutable terminal state to both locked-test records."""
    path = receipt_path.expanduser().resolve()
    receipt, reservation_path = _locked_test_records(path)
    if receipt.get("status") != "started":
        raise ValueError("locked-test receipt is not in the started state")
    terminal = {**receipt, **terminal_fields}
    # The operator-controlled registry is authoritative for second-look
    # rejection, so transition it first. Each destination observes either the
    # complete started record or the complete terminal record, never partial JSON.
    _replace_json_atomically(reservation_path, terminal)
    try:
        _replace_json_atomically(path, terminal)
    except Exception:
        # A transient second-write failure is recoverable because the registry
        # is authoritative. The loader validates its terminal extension before
        # copying it over the still-started receipt.
        reconciled, _ = _locked_test_records(path)
        if reconciled != terminal:
            raise ValueError("locked-test terminal reconciliation disagrees") from None
    final, _ = _locked_test_records(path)
    if final != terminal:
        raise ValueError("locked-test terminal records do not match")


def _locked_test_failure_metadata(exc: BaseException) -> dict[str, Any]:
    """Return bounded failure metadata without persisting arbitrary exception text."""
    exception_type = re.sub(r"[^A-Za-z0-9_.-]", "_", type(exc).__name__)[:128]
    return {
        "exception_type": exception_type or "Exception",
        "message_omitted": True,
    }


def reproducibility_environment_versions() -> dict[str, str]:
    """Resolve the exact model/data runtime recorded by train and locked test."""
    return {
        distribution: importlib.metadata.version(distribution)
        for distribution in (
            "chronos-forecasting",
            "transformers",
            "torch",
            "peft",
            "numpy",
            "polars",
            "holidays",
            "pyarrow",
        )
    }


def configure_reproducible_runtime(seed: int) -> None:
    """Apply the deterministic Torch/NumPy/Python policy used by train and test."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    # Must be set before the first CUDA BLAS operation. The H100 runner starts
    # a fresh process, so doing this before model load is an enforceable gate.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def reproducibility_runtime_identity() -> dict[str, Any]:
    """Capture the execution/runtime identity relevant to H100 reproducibility."""
    import torch

    accelerators = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            accelerators.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_name": Path(sys.executable).name,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "accelerator_count": torch.cuda.device_count(),
        "accelerators": accelerators,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "default_dtype": str(torch.get_default_dtype()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def validate_h100_runtime(runtime: dict[str, Any]) -> None:
    accelerators = runtime.get("accelerators")
    if (
        runtime.get("cuda_available") is not True
        or runtime.get("accelerator_count") != 1
        or not isinstance(accelerators, list)
        or len(accelerators) != 1
        or not isinstance(accelerators[0], dict)
        or "H100" not in str(accelerators[0].get("name"))
    ):
        raise ValueError(
            "frozen v0.2 training/test requires exactly one visible NVIDIA H100"
        )


def validate_reproducibility_runtime(runtime: dict[str, Any]) -> None:
    """Require the frozen deterministic H100 policy, not merely recorded labels."""
    system = runtime.get("system")
    if not isinstance(system, dict):
        raise ValueError("frozen v0.2 runtime is missing its system identity")
    validate_h100_runtime(system)
    required_system = {
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": True,
        "float32_matmul_precision": "high",
        "cublas_workspace_config": ":4096:8",
    }
    if any(system.get(key) != expected for key, expected in required_system.items()):
        raise ValueError("frozen v0.2 runtime does not enforce deterministic H100 settings")
    if runtime.get("chronos_fit") != FROZEN_CHRONOS_FIT_RUNTIME:
        raise ValueError("frozen v0.2 runtime has unexpected Chronos trainer settings")


def verify_code_checkout(repo_root: Path, expected_revision: str) -> None:
    """Bind a declared code revision to the clean checkout executing training."""
    _immutable_digest("code-revision", expected_revision, (40, 64))
    root = repo_root.expanduser().resolve()
    try:
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=root,
            check=False,
        )
        untracked_source = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "*.py",
                "*.toml",
                "*.yml",
                "*.yaml",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("training requires a verifiable Git checkout") from exc
    if actual != expected_revision:
        raise ValueError("code-revision does not match the executing Git checkout")
    if tracked.returncode != 0 or untracked_source:
        raise ValueError("training requires a clean tracked source checkout")


def verify_data_snapshot_manifest(data_root: Path, expected_sha256: str) -> Path:
    """Bind declared data identity to the manifest and every Parquet byte."""
    validate_frozen_data_snapshot(expected_sha256)
    root = data_root.expanduser().resolve()
    manifest_path = root / "snapshot-manifest.json"
    if not manifest_path.is_file() or _sha256_file(manifest_path) != expected_sha256:
        raise ValueError(
            "data-snapshot-sha256 does not match SURGE_DATA_DIR/snapshot-manifest.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise ValueError("snapshot manifest has no artifact inventory")
    expected_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("snapshot manifest has an invalid artifact entry")
        value = entry["path"]
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or value in expected_paths:
            raise ValueError("snapshot manifest has an unsafe or duplicate artifact path")
        artifact = root / relative
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"snapshot artifact is missing regular file {value}")
        expected_digest = entry.get("sha256")
        _immutable_digest(f"snapshot.files[{value}].sha256", expected_digest, (64,))
        if _sha256_file(artifact) != expected_digest:
            raise ValueError(f"snapshot artifact checksum changed: {value}")
        expected_paths.add(value)
    actual_paths = {
        artifact.relative_to(root).as_posix()
        for artifact in root.rglob("*.parquet")
        if artifact.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("snapshot Parquet file set differs from its manifest")
    return root


def _immutable_digest(label: str, value: Any, lengths: tuple[int, ...]) -> str:
    if not isinstance(value, str) or len(value) not in lengths:
        expected = " or ".join(str(item) for item in lengths)
        raise ValueError(
            f"{label} must be a lowercase immutable hex digest of length {expected}"
        )
    if re.fullmatch(r"[0-9a-f]+", value) is None:
        expected = " or ".join(str(item) for item in lengths)
        raise ValueError(
            f"{label} must be a lowercase immutable hex digest of length {expected}"
        )
    return value


def revision_for_model_load(model: str | Path, revision: Any) -> str | None:
    """Pin remote model loads while leaving real local artifact paths revision-free."""
    immutable_revision = _immutable_digest("base-revision", revision, (40, 64))
    if Path(model).expanduser().exists():
        return None
    return immutable_revision


def validate_release_lineage(
    base_source: str | Path,
    *,
    base_model_id: str,
    base_revision: str,
) -> None:
    """Allow promotion only from the frozen, non-oracle v0.2 base lineage."""
    if (
        base_model_id != RELEASE_BASE_MODEL_ID
        or base_revision != RELEASE_BASE_REVISION
    ):
        raise ValueError(
            "v0.2 promotion requires release-safe base "
            f"{RELEASE_BASE_MODEL_ID}@{RELEASE_BASE_REVISION}; "
            "legacy and custom lineages are research-only"
        )
    if str(base_source) != RELEASE_BASE_MODEL_ID:
        raise ValueError(
            "v0.2 promotion must load the pinned release model ID directly; "
            "local and custom sources are research-only"
        )


def validate_frozen_data_snapshot(data_snapshot_sha256: str) -> None:
    if data_snapshot_sha256 != FROZEN_DATA_SNAPSHOT_SHA256:
        raise ValueError(
            "v0.2 H100 selection requires the predeclared frozen data snapshot "
            f"{FROZEN_DATA_SNAPSHOT_SHA256}"
        )


def _verify_checkpoint_inventory(
    manifest: dict[str, Any], checkpoint_path: Path
) -> str:
    """Verify every promoted checkpoint byte against the signed manifest."""
    inventory = manifest.get("artifact_files")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("training manifest has no checkpoint artifact inventory")

    expected_paths: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for raw in inventory:
        if not isinstance(raw, dict):
            raise ValueError("training manifest has an invalid checkpoint inventory row")
        value = raw.get("path")
        if not isinstance(value, str) or not value:
            raise ValueError("training manifest checkpoint path is missing")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or value in expected_paths:
            raise ValueError("training manifest has an unsafe or duplicate checkpoint path")
        artifact = checkpoint_path / relative
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"promoted checkpoint is missing regular file {value}")
        resolved = artifact.resolve()
        if checkpoint_path not in resolved.parents:
            raise ValueError(f"promoted checkpoint path escapes its directory: {value}")
        expected_bytes = raw.get("bytes")
        expected_sha = raw.get("sha256")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
            or artifact.stat().st_size != expected_bytes
        ):
            raise ValueError(f"promoted checkpoint byte count changed: {value}")
        _immutable_digest(f"artifact_files[{value}].sha256", expected_sha, (64,))
        if _sha256_file(artifact) != expected_sha:
            raise ValueError(f"promoted checkpoint checksum changed: {value}")
        expected_paths.add(value)
        normalized.append(
            {"path": value, "bytes": expected_bytes, "sha256": expected_sha}
        )

    actual_paths = {
        path.relative_to(checkpoint_path).as_posix()
        for path in checkpoint_path.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("promoted checkpoint file set differs from training manifest")
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verified_schedule_origins(
    value: Any,
    *,
    label: str,
    split: str,
) -> tuple[int, ...]:
    if (
        not isinstance(value, dict)
        or value.get("split") != split
        or value.get("shared_across_bas") is not True
        or value.get("complete_target_origins_only") is not True
        or value.get("origin_count")
        != DEFAULT_THRESHOLDS.required_diagnostic_windows_per_ba
        or value.get("requested_origin_count")
        != DEFAULT_THRESHOLDS.required_diagnostic_windows_per_ba
        or value.get("step_hours") != 24
    ):
        raise ValueError(f"{label} is not a frozen shared origin schedule")
    origins = value.get("origins_utc")
    if (
        not isinstance(origins, list)
        or len(origins) != DEFAULT_THRESHOLDS.required_diagnostic_windows_per_ba
        or not all(isinstance(origin, str) and origin for origin in origins)
    ):
        raise ValueError(f"{label} has an invalid origin inventory")
    try:
        keys = tuple(
            int(np.datetime64(origin).astype("datetime64[us]").astype(np.int64))
            for origin in origins
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has an invalid origin timestamp") from exc
    encoded = np.asarray(keys, dtype="<i8")
    fingerprint = _immutable_digest(
        f"{label}.origin_sha256", value.get("origin_sha256"), (64,)
    )
    deltas = np.diff(encoded)
    if (
        len(set(keys)) != len(keys)
        or np.any(deltas <= 0)
        or np.any(deltas % (24 * 3_600_000_000) != 0)
        or value.get("origin_start_utc") != origins[0]
        or value.get("origin_end_utc") != origins[-1]
        or hashlib.sha256(encoded.tobytes()).hexdigest() != fingerprint
    ):
        raise ValueError(f"{label} origin inventory changed")
    return keys


def verify_promotion_artifact(path: Path, *, model_path: Path) -> dict[str, Any]:
    """Verify that a locked-test model has a complete eligible promotion chain."""
    marker_path = path.expanduser().resolve()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        raise ValueError("promotion artifact must contain a JSON object")
    if (
        marker.get("policy_version") != POLICY_VERSION
        or marker.get("promotion_eligible") is not True
        or marker.get("test_opened") is not False
    ):
        raise ValueError("promotion artifact is not an eligible unopened v0.2 decision")

    def referenced_file(field: str) -> Path:
        value = marker.get(field)
        if not isinstance(value, str):
            raise ValueError(f"promotion artifact is missing {field}")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"promotion artifact has unsafe {field}")
        resolved = (marker_path.parent / relative).resolve()
        if resolved.parent != marker_path.parent or not resolved.is_file():
            raise ValueError(f"promotion artifact has missing {field}")
        expected = marker.get(f"{field}_sha256")
        if not isinstance(expected, str) or _sha256_file(resolved) != expected:
            raise ValueError(f"promotion artifact has invalid {field} checksum")
        return resolved

    manifest_path = referenced_file("manifest")
    audit_path = referenced_file("overfit_audit")
    checkpoint_name = marker.get("checkpoint")
    if not isinstance(checkpoint_name, str):
        raise ValueError("promotion artifact is missing checkpoint")
    checkpoint = Path(checkpoint_name)
    if checkpoint.is_absolute() or ".." in checkpoint.parts:
        raise ValueError("promotion artifact has unsafe checkpoint")
    checkpoint_path = (marker_path.parent / checkpoint).resolve()
    if checkpoint_path != model_path.expanduser().resolve() or not checkpoint_path.is_dir():
        raise ValueError("locked-test model path does not match promoted checkpoint")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selection = manifest.get("selection") if isinstance(manifest, dict) else None
    split_contract = audit.get("split_contract") if isinstance(audit, dict) else None
    if (
        not isinstance(selection, dict)
        or selection.get("promotion_eligible") is not True
        or selection.get("policy_version") != POLICY_VERSION
        or selection.get("checkpoint_state") != "promoted"
        or selection.get("diagnostics_error") is not None
        or manifest.get("locked_test_opened") is not False
    ):
        raise ValueError("training manifest is not an eligible unopened decision")
    if (
        not isinstance(audit, dict)
        or audit.get("promotion_eligible") is not True
        or audit.get("policy_version") != POLICY_VERSION
        or not isinstance(split_contract, dict)
        or split_contract.get("test_opened") is not False
    ):
        raise ValueError("overfit audit is not an eligible unopened decision")
    gates = audit.get("gates")
    behavior = audit.get("training_behavior")
    metrics = audit.get("metrics")
    gate_names = {
        gate.get("name") for gate in gates if isinstance(gate, dict)
    } if isinstance(gates, list) else set()
    if (
        audit.get("thresholds") != asdict(DEFAULT_THRESHOLDS)
        or audit.get("required_bas") != list(TRUST_RTO_BAS)
        or not isinstance(gates, list)
        or len(gates) != len(ELIGIBLE_GATE_NAMES)
        or gate_names != ELIGIBLE_GATE_NAMES
        or any(not isinstance(gate, dict) or gate.get("passed") is not True for gate in gates)
        or audit.get("rejection_reasons") != []
        or not isinstance(behavior, dict)
        or behavior.get("completed_steps") != behavior.get("expected_steps")
        or not isinstance(behavior.get("events"), list)
        or not behavior["events"]
        or not isinstance(metrics, dict)
        or not isinstance(metrics.get("per_rto"), dict)
        or set(metrics["per_rto"]) != set(TRUST_RTO_BAS)
    ):
        raise ValueError("eligible overfit audit is missing its complete passing evidence")
    checkpoint_origins = _verified_schedule_origins(
        audit.get("checkpoint_selection_schedule"),
        label="checkpoint selection schedule",
        split="val",
    )
    promotion_origins = _verified_schedule_origins(
        audit.get("promotion_validation_schedule"),
        label="promotion validation schedule",
        split="val",
    )
    _verified_schedule_origins(
        audit.get("promotion_train_schedule"),
        label="promotion train schedule",
        split="train",
    )
    if (
        set(checkpoint_origins) & set(promotion_origins)
        or max(checkpoint_origins) >= min(promotion_origins)
    ):
        raise ValueError("checkpoint and promotion validation schedules are not disjoint")
    promotion_origin_sha256 = hashlib.sha256(
        np.asarray(promotion_origins, dtype="<i8").tobytes()
    ).hexdigest()
    paired_validation = metrics.get("paired_validation")
    if (
        not isinstance(paired_validation, dict)
        or paired_validation.get("origin_sha256") != promotion_origin_sha256
        or any(
            metrics["per_rto"][ba]["validation"]["origins"]["sha256"]
            != promotion_origin_sha256
            for ba in TRUST_RTO_BAS
        )
    ):
        raise ValueError("promotion metrics do not bind the reserved origin schedule")
    paired_ratio_high = _verify_paired_validation_evidence(
        paired_validation,
        promotion_origins=promotion_origins,
    )
    paired_gate = next(
        (
            gate
            for gate in gates
            if isinstance(gate, dict)
            and gate.get("name") == "paired_bootstrap_mase_ratio_upper_95"
        ),
        None,
    )
    paired_gate_actual = (
        _finite_number(paired_gate.get("actual"))
        if isinstance(paired_gate, dict)
        else None
    )
    if (
        not isinstance(paired_gate, dict)
        or paired_gate_actual is None
        or paired_gate.get("operator") != "<="
        or paired_gate.get("threshold")
        != DEFAULT_THRESHOLDS.max_paired_bootstrap_mase_ratio_upper_95
        or not math.isclose(
            paired_gate_actual,
            paired_ratio_high,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("paired validation gate does not match its evidence")
    if selection.get("overfit_audit_sha256") != marker.get("overfit_audit_sha256"):
        raise ValueError("manifest and promotion marker disagree on overfit audit")
    if (
        selection.get("overfit_audit") != marker.get("overfit_audit")
        or marker.get("manifest") != manifest_path.name
    ):
        raise ValueError("promotion chain filenames disagree")

    manifest_bas = manifest.get("bas")
    if not isinstance(manifest_bas, list):
        raise ValueError("training manifest is missing RTO identities")
    canonical_bas = validate_promotion_inputs(
        manifest_bas,
        base_revision=manifest.get("base_model_revision"),
        code_revision=manifest.get("code_revision"),
        data_snapshot_sha256=manifest.get("data_snapshot_sha256"),
    )
    validate_release_lineage(
        manifest.get("base_model_source"),
        base_model_id=manifest.get("base_model"),
        base_revision=manifest.get("base_model_revision"),
    )
    validate_frozen_data_snapshot(manifest.get("data_snapshot_sha256"))
    required_versions = {
        "chronos-forecasting",
        "transformers",
        "torch",
        "peft",
        "numpy",
        "polars",
        "holidays",
        "pyarrow",
    }
    versions = manifest.get("versions")
    audit_environment = audit.get("environment")
    runtime = manifest.get("runtime")
    if (
        not isinstance(versions, dict)
        or not required_versions.issubset(versions)
        or not isinstance(audit_environment, dict)
        or not required_versions.issubset(audit_environment)
        or versions != audit_environment
        or not isinstance(runtime, dict)
        or audit.get("runtime") != runtime
    ):
        raise ValueError("promotion chain has missing or inconsistent runtime identity")
    validate_reproducibility_runtime(runtime)
    feature_spec_sha256 = _immutable_digest(
        "feature-spec-sha256", manifest.get("feature_spec_sha256"), (64,)
    )
    expected_identity = {
        "base_model": manifest.get("base_model"),
        "base_revision": manifest["base_model_revision"],
        "code_revision": manifest["code_revision"],
        "data_snapshot_sha256": manifest["data_snapshot_sha256"],
        "feature_spec_version": manifest.get("feature_spec_version"),
        "feature_spec_sha256": feature_spec_sha256,
        "bas": canonical_bas,
    }
    if audit.get("identity") != expected_identity:
        raise ValueError("overfit audit identity disagrees with training manifest")
    training_config = manifest.get("config")
    if (
        not isinstance(training_config, dict)
        or training_config.get("shared_origin_schedule") is not True
        or training_config.get("checkpoint_selection_disjoint_from_promotion")
        is not True
        or training_config.get("complete_target_origins_only") is not True
    ):
        raise ValueError("training manifest is missing its immutable configuration")
    inventory_sha256 = _verify_checkpoint_inventory(manifest, checkpoint_path)
    model_artifact_hash_algorithm = manifest.get("model_artifact_hash_algorithm")
    model_artifact_sha256 = _immutable_digest(
        "model-artifact-sha256", manifest.get("model_artifact_sha256"), (64,)
    )
    if model_artifact_hash_algorithm != "sha256-tree-v1":
        raise ValueError("training manifest has an unsupported model artifact hash")
    if (
        marker.get("model_artifact_hash_algorithm")
        != model_artifact_hash_algorithm
        or marker.get("model_artifact_sha256") != model_artifact_sha256
    ):
        raise ValueError("promotion marker and manifest disagree on model artifact hash")
    if artifact_sha256(checkpoint_path) != model_artifact_sha256:
        raise ValueError("promoted checkpoint model artifact SHA-256 changed")

    return {
        **marker,
        "marker_sha256": _sha256_file(marker_path),
        "checkpoint_inventory_sha256": inventory_sha256,
        "model_artifact_hash_algorithm": model_artifact_hash_algorithm,
        "model_artifact_sha256": model_artifact_sha256,
        "training_identity": {
            **expected_identity,
            "training_config": training_config,
            "versions": versions,
            "runtime": runtime,
        },
    }


def verify_selection_artifact(
    path: Path,
    *,
    promotion_path: Path,
    promotion: dict[str, Any],
) -> dict[str, Any]:
    """Verify the frozen two-candidate decision and its selected marker."""
    selection_path = path.expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise ValueError("selection artifact must contain a JSON object")
    if (
        selection.get("policy_version") != SELECTION_POLICY_VERSION
        or selection.get("locked_test_opened") is not False
    ):
        raise ValueError("selection artifact is not a frozen unopened v0.2 decision")
    candidates = selection.get("candidates")
    winner = selection.get("winner")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("selection artifact must contain exactly two candidates")
    if not isinstance(winner, dict) or winner.get("promotion_eligible") is not True:
        raise ValueError("selection artifact has no eligible winner")
    if any(not isinstance(record, dict) for record in candidates):
        raise ValueError("selection artifact has an invalid candidate record")
    selected_marker_path = promotion_path.expanduser().resolve()
    selected_root = selected_marker_path.parent
    public_identity = dict(promotion["training_identity"])
    public_identity.pop("training_config", None)
    if selection.get("training_identity") != public_identity:
        raise ValueError("selection identity does not match promoted winner")

    actual_records: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    frozen_config = {
        "context": 2_048,
        "horizon": 24,
        "mode": "lora",
        "learning_rate": 1e-5,
        "batch_size": 32,
        "diagnostic_origins": 90,
        "diagnostic_step": 24,
        "diagnostic_batch_size": 16,
        "complete_target_origins_only": True,
        "shared_origin_schedule": True,
        "checkpoint_selection_disjoint_from_promotion": True,
        "seed": 42,
        "with_generation": False,
    }
    for claimed in candidates:
        candidate_name = claimed.get("candidate")
        if (
            not isinstance(candidate_name, str)
            or not candidate_name
            or Path(candidate_name).name != candidate_name
            or candidate_name in {".", ".."}
            or candidate_name in seen_candidates
        ):
            raise ValueError("selection candidates must be distinct safe sibling names")
        seen_candidates.add(candidate_name)
        root = (selection_path.parent / candidate_name).resolve()
        if root.parent != selection_path.parent or not root.is_dir():
            raise ValueError("selection candidate is not a sibling artifact directory")
        manifest_path = root / "surge-training-manifest.json"
        audit_path = root / "surge-overfit-audit.json"
        if not manifest_path.is_file() or not audit_path.is_file():
            raise ValueError("selection candidate is missing its manifest or audit")
        manifest_sha256 = _sha256_file(manifest_path)
        audit_sha256 = _sha256_file(audit_path)
        if claimed.get("manifest_sha256") != manifest_sha256 or claimed.get(
            "overfit_audit_sha256"
        ) != audit_sha256:
            raise ValueError("selection candidate artifact hashes changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(audit, dict):
            raise ValueError("selection candidate manifest and audit must be objects")
        manifest_selection = manifest.get("selection")
        config = manifest.get("config")
        split_contract = audit.get("split_contract")
        if (
            not isinstance(manifest_selection, dict)
            or not isinstance(config, dict)
            or audit.get("policy_version") != POLICY_VERSION
            or not isinstance(split_contract, dict)
            or split_contract.get("test_opened") is not False
            or manifest.get("locked_test_opened") is not False
            or manifest_selection.get("policy_version") != POLICY_VERSION
            or manifest_selection.get("overfit_audit") != audit_path.name
            or manifest_selection.get("overfit_audit_sha256") != audit_sha256
        ):
            raise ValueError("selection candidate has an invalid audit chain")
        for field, expected in frozen_config.items():
            if config.get(field) != expected:
                raise ValueError(f"selection candidate has unexpected {field}")
        num_steps = config.get("num_steps")
        if isinstance(num_steps, bool) or num_steps not in SELECTION_TRAINING_STEPS:
            raise ValueError("selection candidate has an undeclared training duration")

        bas = validate_promotion_inputs(
            manifest.get("bas"),
            base_revision=manifest.get("base_model_revision"),
            code_revision=manifest.get("code_revision"),
            data_snapshot_sha256=manifest.get("data_snapshot_sha256"),
        )
        validate_release_lineage(
            manifest.get("base_model_source"),
            base_model_id=manifest.get("base_model"),
            base_revision=manifest.get("base_model_revision"),
        )
        candidate_identity = {
            "base_model": manifest.get("base_model"),
            "base_revision": manifest.get("base_model_revision"),
            "code_revision": manifest.get("code_revision"),
            "data_snapshot_sha256": manifest.get("data_snapshot_sha256"),
            "feature_spec_version": manifest.get("feature_spec_version"),
            "feature_spec_sha256": manifest.get("feature_spec_sha256"),
            "bas": bas,
            "versions": manifest.get("versions"),
            "runtime": manifest.get("runtime"),
        }
        if candidate_identity != public_identity:
            raise ValueError("selection candidates do not share the frozen identity")
        audit_identity = dict(candidate_identity)
        audit_identity.pop("versions")
        audit_identity.pop("runtime")
        if audit.get("identity") != audit_identity:
            raise ValueError("selection candidate audit identity disagrees with manifest")
        if (
            audit.get("environment") != manifest.get("versions")
            or not isinstance(manifest.get("runtime"), dict)
            or audit.get("runtime") != manifest.get("runtime")
        ):
            raise ValueError("selection candidate runtime identity disagrees")

        manifest_eligible = manifest_selection.get("promotion_eligible")
        audit_eligible = audit.get("promotion_eligible")
        if not isinstance(manifest_eligible, bool) or manifest_eligible is not audit_eligible:
            raise ValueError("selection candidate promotion decisions disagree")
        marker_sha256: str | None = None
        score: float | None = None
        mase_ratio: float | None = None
        wis_ratio: float | None = None
        metrics = audit.get("metrics")
        generalization = metrics.get("generalization") if isinstance(metrics, dict) else None
        if isinstance(generalization, dict):
            mase_ratio = _finite_number(
                generalization.get("macro_mase_vs_baseline_ratio")
            )
            wis_ratio = _finite_number(
                generalization.get("macro_wis_scaled_vs_baseline_ratio")
            )
        if manifest_eligible:
            marker_path = root / "surge-promotion.json"
            if root == selected_root:
                verified = promotion
            else:
                verified = verify_promotion_artifact(marker_path, model_path=root / "best")
            verified_identity = dict(verified["training_identity"])
            verified_identity.pop("training_config", None)
            if verified_identity != public_identity:
                raise ValueError("eligible selection candidate chain changed")
            marker_sha256 = verified["marker_sha256"]
            if not isinstance(generalization, dict):
                raise ValueError("eligible selection candidate is missing metrics")
            if mase_ratio is None or mase_ratio < 0 or wis_ratio is None or wis_ratio < 0:
                raise ValueError("eligible selection candidate has invalid score metrics")
            score = 0.5 * mase_ratio + 0.5 * wis_ratio
        else:
            if (root / "surge-promotion.json").exists():
                raise ValueError("rejected selection candidate unexpectedly has a marker")
            gates = audit.get("gates")
            rejection_reasons = audit.get("rejection_reasons")
            diagnostics_error = manifest_selection.get("diagnostics_error")
            if (
                manifest_selection.get("checkpoint_state") != "candidate-rejected"
                or audit.get("thresholds") != asdict(DEFAULT_THRESHOLDS)
                or audit.get("required_bas") != list(TRUST_RTO_BAS)
                or not isinstance(gates, list)
                or not gates
                or any(
                    not isinstance(gate, dict)
                    or not isinstance(gate.get("passed"), bool)
                    for gate in gates
                )
                or all(gate["passed"] for gate in gates)
                or not isinstance(rejection_reasons, list)
                or not rejection_reasons
                or any(not isinstance(reason, str) or not reason for reason in rejection_reasons)
                or (
                    diagnostics_error is not None
                    and (not isinstance(diagnostics_error, str) or not diagnostics_error)
                )
            ):
                raise ValueError("rejected selection candidate has incomplete failure evidence")
            metrics = audit.get("metrics")
            if diagnostics_error is None and (
                not isinstance(metrics, dict)
                or not isinstance(metrics.get("per_rto"), dict)
                or set(metrics["per_rto"]) != set(TRUST_RTO_BAS)
            ):
                raise ValueError("rejected selection candidate is missing gate metrics")
            _verify_checkpoint_inventory(
                manifest,
                root / "candidate-unpromoted",
            )

        actual = {
            "candidate": candidate_name,
            "num_steps": num_steps,
            "promotion_eligible": manifest_eligible,
            "selection_score": score,
            "validation_mase_vs_base_ratio": mase_ratio,
            "validation_scaled_wis_vs_base_ratio": wis_ratio,
            "rejection_reasons": audit.get("rejection_reasons", []),
            "manifest_sha256": manifest_sha256,
            "overfit_audit_sha256": audit_sha256,
            "promotion_marker_sha256": marker_sha256,
        }
        if claimed != actual:
            raise ValueError("selection candidate record disagrees with source artifacts")
        actual_records.append(actual)

    if {record["num_steps"] for record in actual_records} != set(SELECTION_TRAINING_STEPS):
        raise ValueError("selection artifact must contain one candidate per training duration")
    eligible_records = [
        record for record in actual_records if record["promotion_eligible"] is True
    ]
    if not eligible_records:
        raise ValueError("selection artifact has no eligible candidates")
    expected_winner = min(
        eligible_records,
        key=lambda record: (
            round(float(record["selection_score"]), 6),
            record["num_steps"],
        ),
    )
    if winner != expected_winner:
        raise ValueError("selection artifact winner violates the frozen selection rule")
    if selected_root != (selection_path.parent / winner["candidate"]).resolve():
        raise ValueError("selection winner does not match the supplied promotion marker")
    if winner.get("promotion_marker_sha256") != promotion.get("marker_sha256"):
        raise ValueError("selection winner marker checksum does not match promotion")

    semantic_candidates = [
        {key: value for key, value in record.items() if key != "candidate"}
        for record in sorted(actual_records, key=lambda record: record["num_steps"])
    ]
    semantic_winner = {
        key: value for key, value in expected_winner.items() if key != "candidate"
    }
    semantic_decision = {
        "policy_version": SELECTION_POLICY_VERSION,
        "training_identity": public_identity,
        "candidates": semantic_candidates,
        "winner": semantic_winner,
    }
    decision_encoded = json.dumps(
        semantic_decision,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    experiment_protocol = {
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "overfit_policy_version": POLICY_VERSION,
        "frozen_identity": {
            key: public_identity[key]
            for key in (
                "base_model",
                "base_revision",
                "data_snapshot_sha256",
                "feature_spec_version",
                "feature_spec_sha256",
                "bas",
            )
        },
        "candidate_protocols": [
            {**frozen_config, "num_steps": steps}
            for steps in SELECTION_TRAINING_STEPS
        ],
        "locked_test_protocol": {
            "bas": list(TRUST_RTO_BAS),
            "context": 2_048,
            "horizon": 24,
            "step": 24,
            "max_origins": None,
            "bootstrap": 2_000,
            "seed": 42,
            "with_generation": False,
            "per_step": True,
        },
    }
    protocol_encoded = json.dumps(
        experiment_protocol,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()

    return {
        **selection,
        "selection_sha256": _sha256_file(selection_path),
        "selection_decision_sha256": hashlib.sha256(decision_encoded).hexdigest(),
        "experiment_protocol_sha256": hashlib.sha256(protocol_encoded).hexdigest(),
        "selection_path": str(selection_path),
    }


def reserve_locked_test_run(
    selection_path: Path,
    *,
    experiment: str,
    training_identity: dict[str, Any],
    selection_sha256: str,
    selection_decision_sha256: str,
    experiment_protocol_sha256: str,
    promotion_path: Path,
    marker_sha256: str,
    checkpoint_inventory_sha256: str,
    model_artifact_sha256: str,
    registry_root: Path,
) -> Path:
    """Atomically consume the one allowed test run before test rows are loaded.

    A started receipt remains consuming if the process crashes. This is deliberate:
    an interrupted look at the locked test cannot be safely treated as unseen.
    """
    frozen_selection_path = selection_path.expanduser().resolve()
    marker_path = promotion_path.expanduser().resolve()
    receipt_path = frozen_selection_path.with_name("surge-locked-test-receipt.json")
    _immutable_digest("selection-sha256", selection_sha256, (64,))
    _immutable_digest(
        "selection-decision-sha256", selection_decision_sha256, (64,)
    )
    _immutable_digest(
        "experiment-protocol-sha256", experiment_protocol_sha256, (64,)
    )
    _immutable_digest("model-artifact-sha256", model_artifact_sha256, (64,))
    registry = registry_root.expanduser().resolve()
    if registry.exists() and (registry.is_symlink() or not registry.is_dir()):
        raise ValueError("locked-test registry must be a regular directory")
    registry.mkdir(parents=True, exist_ok=True)
    reservation_path = registry / f"{experiment_protocol_sha256}.json"
    receipt = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "status": "started",
        "test_opened": True,
        "started_at_utc": datetime.now(tz=UTC).isoformat(),
        "experiment": experiment,
        "selection_artifact": frozen_selection_path.name,
        "selection_artifact_sha256": selection_sha256,
        "selection_decision_sha256": selection_decision_sha256,
        "experiment_protocol_sha256": experiment_protocol_sha256,
        "registry_reservation": str(reservation_path),
        "promotion_marker": marker_path.name,
        "promotion_marker_sha256": marker_sha256,
        "checkpoint_inventory_sha256": checkpoint_inventory_sha256,
        "model_artifact_hash_algorithm": "sha256-tree-v1",
        "model_artifact_sha256": model_artifact_sha256,
        "training_identity": training_identity,
    }
    try:
        _link_exclusive_json(reservation_path, receipt)
    except FileExistsError as exc:
        raise RuntimeError(
            "locked test was already consumed for this selection; "
            f"registry reservation exists at {reservation_path}"
        ) from exc
    try:
        _link_exclusive_json(receipt_path, receipt)
    except FileExistsError as exc:
        raise RuntimeError(
            f"locked test was already consumed; receipt exists at {receipt_path}"
        ) from exc
    return receipt_path


def complete_locked_test_run(receipt_path: Path, result: dict[str, Any]) -> None:
    """Complete a previously reserved receipt with the immutable metric payload."""
    encoded_result = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode()
    _terminalize_locked_test_run(
        receipt_path,
        {
            "status": "completed",
            "completed_at_utc": datetime.now(tz=UTC).isoformat(),
            "result_sha256": hashlib.sha256(encoded_result).hexdigest(),
            "result": result,
        },
    )


def fail_locked_test_run(receipt_path: Path, exc: BaseException) -> None:
    """Record a post-reservation exception as an immutable terminal failure."""
    failure = _locked_test_failure_metadata(exc)
    encoded_failure = json.dumps(
        failure,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    _terminalize_locked_test_run(
        receipt_path,
        {
            "status": "failed",
            "failed_at_utc": datetime.now(tz=UTC).isoformat(),
            "failure_sha256": hashlib.sha256(encoded_failure).hexdigest(),
            "failure": failure,
        },
    )


def validate_promotion_inputs(
    bas: list[str],
    *,
    base_revision: str,
    code_revision: str,
    data_snapshot_sha256: str,
) -> list[str]:
    """Validate immutable selection identity and return canonical RTO order."""
    if len(bas) != len(TRUST_RTO_BAS) or set(bas) != set(TRUST_RTO_BAS):
        raise ValueError(
            "v0.2 promotion training requires exactly one of each trust RTO: "
            + ",".join(TRUST_RTO_BAS)
        )

    for label, value, lengths in (
        ("base-revision", base_revision, (40, 64)),
        ("code-revision", code_revision, (40, 64)),
        ("data-snapshot-sha256", data_snapshot_sha256, (64,)),
    ):
        _immutable_digest(label, value, lengths)
    return list(TRUST_RTO_BAS)


def validation_only_view(data: BAData) -> BAData:
    """Return a copy whose arrays end before the locked 2025 test partition."""
    if data.train_end <= 0 or data.val_end <= data.train_end:
        raise ValueError(f"{data.ba} has no non-empty train/validation split")
    if data.val_end > len(data.target) or data.val_end > len(data.ts_utc):
        raise ValueError(f"{data.ba} split boundaries exceed available rows")
    end = data.val_end
    return replace(
        data,
        ts_utc=data.ts_utc[:end].copy(),
        target=data.target[:end].copy(),
        covariates={key: values[:end].copy() for key, values in data.covariates.items()},
        val_end=end,
        warnings=tuple(
            dict.fromkeys(
                (*data.warnings, "locked 2025 test rows excluded from model selection")
            )
        ),
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _step(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _checkpoint_step(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"checkpoint-(\d+)$", Path(value).name)
    return int(match.group(1)) if match else None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= _EPSILON:
        return 1.0 if numerator <= _EPSILON else None
    ratio = numerator / denominator
    return ratio if math.isfinite(ratio) else None


def summarize_training_history(
    events: Iterable[dict[str, Any]],
    *,
    final_state: dict[str, Any] | None,
    expected_steps: int,
) -> dict[str, Any]:
    """Summarize Trainer logs without hiding checkpoint regressions."""
    normalized: list[dict[str, Any]] = []
    train_by_step: dict[int, float] = {}
    eval_by_step: dict[int, float] = {}
    checkpoint_steps: set[int] = set()

    for raw in events:
        if not isinstance(raw, dict):
            continue
        step = _step(raw.get("step"))
        if step is None:
            continue
        event: dict[str, Any] = {"step": step, "event": str(raw.get("event", "log"))}
        for source, target in (
            ("train_loss", "train_loss"),
            ("eval_loss", "eval_loss"),
            ("learning_rate", "learning_rate"),
            ("epoch", "epoch"),
        ):
            value = _finite_number(raw.get(source))
            if value is not None:
                event[target] = value
        if "train_loss" in event:
            train_by_step[step] = event["train_loss"]
        if "eval_loss" in event:
            eval_by_step[step] = event["eval_loss"]
        if event["event"] == "checkpoint":
            checkpoint_steps.add(step)
        normalized.append(event)

    train_points = sorted(train_by_step.items())
    eval_points = sorted(eval_by_step.items())
    best_eval_step: int | None = None
    best_eval_loss: float | None = None
    final_eval_step: int | None = None
    final_eval_loss: float | None = None
    if eval_points:
        best_eval_step, best_eval_loss = min(eval_points, key=lambda item: item[1])
        final_eval_step, final_eval_loss = eval_points[-1]

    state = final_state or {}
    completed_steps = _step(state.get("global_step"))
    reported_best_metric = _finite_number(state.get("best_metric"))
    reported_checkpoint = state.get("best_model_checkpoint")
    reported_best_step = _checkpoint_step(reported_checkpoint)
    rebound = (
        _safe_ratio(final_eval_loss, best_eval_loss)
        if final_eval_loss is not None and best_eval_loss is not None
        else None
    )
    reported_matches_observed = (
        reported_best_metric is not None
        and best_eval_loss is not None
        and math.isclose(reported_best_metric, best_eval_loss, rel_tol=1e-6, abs_tol=1e-9)
        and reported_best_step == best_eval_step
    )
    reported_checkpoint_was_saved = (
        reported_best_step is not None and reported_best_step in checkpoint_steps
    )

    return {
        "expected_steps": expected_steps,
        "completed_steps": completed_steps,
        "train_loss_log_count": len(train_points),
        "eval_checkpoint_count": len(eval_points),
        "checkpoint_save_steps": sorted(checkpoint_steps),
        "first_train_loss": train_points[0][1] if train_points else None,
        "final_train_loss": train_points[-1][1] if train_points else None,
        "minimum_train_loss": min((value for _, value in train_points), default=None),
        "best_eval_step": best_eval_step,
        "best_eval_loss": best_eval_loss,
        "final_eval_step": final_eval_step,
        "final_eval_loss": final_eval_loss,
        "eval_loss_rebound_ratio": rebound,
        "reported_best_checkpoint": (
            Path(reported_checkpoint).name if isinstance(reported_checkpoint, str) else None
        ),
        "reported_best_step": reported_best_step,
        "reported_best_metric": reported_best_metric,
        "reported_best_matches_observed": reported_matches_observed,
        "reported_best_checkpoint_was_saved": reported_checkpoint_was_saved,
        "events": sorted(normalized, key=lambda item: (item["step"], item["event"])),
    }


def _gate(
    name: str,
    *,
    actual: Any,
    operator: str,
    threshold: Any,
    passed: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "actual": actual,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _metric(per_ba: dict[str, Any], ba: str, key: str) -> float:
    row = per_ba.get(ba)
    if not isinstance(row, dict):
        raise ValueError(f"missing metrics for {ba}")
    value = _finite_number(row.get(key))
    if value is None or value < 0:
        raise ValueError(f"{ba} has invalid {key}")
    return value


def _windows(per_ba: dict[str, Any], ba: str) -> int:
    row = per_ba.get(ba)
    value = _step(row.get("n_windows") if isinstance(row, dict) else None)
    if value is None:
        raise ValueError(f"{ba} has invalid n_windows")
    point_count = _step(row.get("n_points") if isinstance(row, dict) else None)
    expected_points = value * DIAGNOSTIC_HORIZON
    if point_count != expected_points:
        raise ValueError(
            f"{ba} diagnostic windows are incomplete: "
            f"expected {expected_points} finite points, got {point_count}"
        )
    return value


def _origin_contract(per_ba: dict[str, Any], ba: str) -> dict[str, Any]:
    row = per_ba.get(ba)
    if not isinstance(row, dict):
        raise ValueError(f"missing metrics for {ba}")
    start = row.get("origin_start_utc")
    end = row.get("origin_end_utc")
    step_hours = _step(row.get("origin_step_hours"))
    fingerprint = row.get("origin_sha256")
    if (
        not isinstance(start, str)
        or not start
        or not isinstance(end, str)
        or not end
        or step_hours is None
        or step_hours < 1
    ):
        raise ValueError(f"{ba} has an invalid origin contract")
    _immutable_digest(f"{ba}.origin_sha256", fingerprint, (64,))
    return {
        "start_utc": start,
        "end_utc": end,
        "step_hours": step_hours,
        "sha256": fingerprint,
    }


def _validate_evaluation_contract(
    metrics: dict[str, Any],
    *,
    label: str,
    split: str,
    require_origin_metrics: bool,
) -> None:
    expected = {
        "split": split,
        "aggregation": "equal_ba_macro",
        "point_estimate_kind": POINT_ESTIMATE_KIND,
        "point_estimate_quantile": POINT_ESTIMATE_LABEL,
        "point_estimate_quantile_value": POINT_ESTIMATE_QUANTILE,
        "horizon": DIAGNOSTIC_HORIZON,
        "origin_step_hours": 24,
        "crps_approximation": "2x_mean_pinball",
        "crps_approx_quantile_levels": [0.1, 0.5, 0.9],
        "complete_target_origins_only": True,
        "shared_origin_schedule": True,
        "origin_metrics_emitted": require_origin_metrics,
    }
    for field, value in expected.items():
        if metrics.get(field) != value:
            raise ValueError(f"{label} metrics have unexpected {field}")
    schedule = metrics.get("origin_schedule")
    if (
        not isinstance(schedule, dict)
        or schedule.get("split") != split
        or schedule.get("shared_across_bas") is not True
        or schedule.get("complete_target_origins_only") is not True
        or schedule.get("origin_count")
        != DEFAULT_THRESHOLDS.required_diagnostic_windows_per_ba
        or schedule.get("requested_origin_count")
        != DEFAULT_THRESHOLDS.required_diagnostic_windows_per_ba
        or schedule.get("step_hours") != 24
    ):
        raise ValueError(f"{label} metrics have an invalid shared origin schedule")
    fingerprint = _immutable_digest(
        f"{label}.origin_schedule.origin_sha256",
        schedule.get("origin_sha256"),
        (64,),
    )
    origins = schedule.get("origins_utc")
    if (
        not isinstance(origins, list)
        or len(origins) != DEFAULT_THRESHOLDS.required_diagnostic_windows_per_ba
        or not all(isinstance(origin, str) and origin for origin in origins)
    ):
        raise ValueError(f"{label} metrics have an invalid origin inventory")
    try:
        origin_keys = np.asarray(
            [
                int(np.datetime64(origin).astype("datetime64[us]").astype(np.int64))
                for origin in origins
            ],
            dtype="<i8",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} metrics have an invalid origin timestamp") from exc
    deltas = np.diff(origin_keys)
    if (
        len(set(origin_keys.tolist())) != len(origin_keys)
        or np.any(deltas <= 0)
        or np.any(deltas % (24 * 3_600_000_000) != 0)
        or schedule.get("origin_start_utc") != origins[0]
        or schedule.get("origin_end_utc") != origins[-1]
        or hashlib.sha256(origin_keys.tobytes()).hexdigest() != fingerprint
    ):
        raise ValueError(f"{label} metrics have a changed origin inventory")


def _origin_mase_contract(per_ba: dict[str, Any], ba: str) -> dict[str, float]:
    row = per_ba.get(ba)
    raw = row.get("origin_mase") if isinstance(row, dict) else None
    if not isinstance(raw, list) or len(raw) != _windows(per_ba, ba):
        raise ValueError(f"{ba} has an invalid per-origin MASE trace")
    values: dict[str, float] = {}
    origin_keys: list[int] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{ba} has an invalid per-origin MASE row")
        origin = item.get("origin_utc")
        mase = _finite_number(item.get("mase"))
        if not isinstance(origin, str) or not origin or mase is None or mase < 0:
            raise ValueError(f"{ba} has an invalid per-origin MASE row")
        if origin in values:
            raise ValueError(f"{ba} has duplicate per-origin MASE rows")
        try:
            origin_key = int(
                np.datetime64(origin).astype("datetime64[us]").astype(np.int64)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{ba} has an invalid per-origin timestamp") from exc
        values[origin] = mase
        origin_keys.append(origin_key)
    expected_sha = _origin_contract(per_ba, ba)["sha256"]
    actual_sha = hashlib.sha256(
        np.asarray(sorted(origin_keys), dtype="<i8").tobytes()
    ).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"{ba} per-origin MASE trace changed origin schedule")
    aggregate_mase = _metric(per_ba, ba, "mase")
    if not math.isclose(
        float(np.mean(list(values.values()))),
        aggregate_mase,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise ValueError(f"{ba} per-origin MASE trace disagrees with aggregate MASE")
    return values


def _paired_moving_block_ratio_stats(
    candidate_macro_by_origin: np.ndarray,
    baseline_macro_by_origin: np.ndarray,
) -> tuple[float, float, float]:
    if (
        candidate_macro_by_origin.ndim != 1
        or baseline_macro_by_origin.shape != candidate_macro_by_origin.shape
        or len(candidate_macro_by_origin)
        != DEFAULT_THRESHOLDS.required_diagnostic_windows_per_ba
        or not np.isfinite(candidate_macro_by_origin).all()
        or not np.isfinite(baseline_macro_by_origin).all()
        or np.any(candidate_macro_by_origin < 0)
        or np.any(baseline_macro_by_origin < 0)
    ):
        raise ValueError("paired validation bootstrap has invalid macro origin rows")
    baseline_macro = float(baseline_macro_by_origin.mean())
    if baseline_macro <= _EPSILON:
        raise ValueError("paired validation baseline MASE must be positive")
    point_ratio = float(candidate_macro_by_origin.mean() / baseline_macro)
    rng = np.random.default_rng(PAIRED_VALIDATION_BOOTSTRAP_SEED)
    ratios = np.empty(PAIRED_VALIDATION_BOOTSTRAP_SAMPLES, dtype=np.float64)
    blocks_per_sample = math.ceil(
        len(candidate_macro_by_origin) / PAIRED_VALIDATION_BOOTSTRAP_BLOCK_ORIGINS
    )
    within_block = np.arange(PAIRED_VALIDATION_BOOTSTRAP_BLOCK_ORIGINS)
    for sample in range(PAIRED_VALIDATION_BOOTSTRAP_SAMPLES):
        starts = rng.integers(
            0,
            len(candidate_macro_by_origin),
            blocks_per_sample,
        )
        indices = (
            (starts[:, None] + within_block[None, :])
            % len(candidate_macro_by_origin)
        ).reshape(-1)[: len(candidate_macro_by_origin)]
        sampled_baseline = float(baseline_macro_by_origin[indices].mean())
        ratios[sample] = (
            float(candidate_macro_by_origin[indices].mean()) / sampled_baseline
            if sampled_baseline > _EPSILON
            else math.inf
        )
    if not np.isfinite(ratios).all():
        raise ValueError("paired validation bootstrap produced a non-finite ratio")
    return (
        point_ratio,
        float(np.quantile(ratios, 0.025)),
        float(np.quantile(ratios, 0.975)),
    )


def _paired_validation_bootstrap(
    candidate_per_ba: dict[str, Any],
    baseline_per_ba: dict[str, Any],
    *,
    noninferiority_ratio_margin: float,
) -> dict[str, Any]:
    candidate = {
        ba: _origin_mase_contract(candidate_per_ba, ba) for ba in TRUST_RTO_BAS
    }
    baseline = {
        ba: _origin_mase_contract(baseline_per_ba, ba) for ba in TRUST_RTO_BAS
    }
    origin_set = set(candidate[TRUST_RTO_BAS[0]])
    if not origin_set:
        raise ValueError("paired validation bootstrap has no origins")
    for ba in TRUST_RTO_BAS:
        if set(candidate[ba]) != origin_set or set(baseline[ba]) != origin_set:
            raise ValueError("paired validation bootstrap requires one shared origin set")
    origins = sorted(origin_set)
    candidate_blocks = np.asarray(
        [[candidate[ba][origin] for ba in TRUST_RTO_BAS] for origin in origins],
        dtype=np.float64,
    )
    baseline_blocks = np.asarray(
        [[baseline[ba][origin] for ba in TRUST_RTO_BAS] for origin in origins],
        dtype=np.float64,
    )
    candidate_macro_by_origin = candidate_blocks.mean(axis=1)
    baseline_macro_by_origin = baseline_blocks.mean(axis=1)
    point_ratio, ratio_low, ratio_high = _paired_moving_block_ratio_stats(
        candidate_macro_by_origin,
        baseline_macro_by_origin,
    )
    return {
        "method": "paired-circular-moving-block-bootstrap-equal-ba-macro",
        "samples": PAIRED_VALIDATION_BOOTSTRAP_SAMPLES,
        "seed": PAIRED_VALIDATION_BOOTSTRAP_SEED,
        "block_origins": PAIRED_VALIDATION_BOOTSTRAP_BLOCK_ORIGINS,
        "origin_count": len(origins),
        "origin_sha256": _origin_contract(
            candidate_per_ba, TRUST_RTO_BAS[0]
        )["sha256"],
        "candidate_minus_baseline_macro_mase": float(
            candidate_blocks.mean() - baseline_blocks.mean()
        ),
        "candidate_vs_baseline_mase_ratio": point_ratio,
        "mase_ratio_ci_low_95": ratio_low,
        "mase_ratio_ci_high_95": ratio_high,
        "noninferiority_ratio_margin": noninferiority_ratio_margin,
        "per_origin_macro_mase": [
            {
                "origin_utc": origin,
                "candidate": float(candidate_value),
                "baseline": float(baseline_value),
                "candidate_minus_baseline": float(candidate_value - baseline_value),
            }
            for origin, candidate_value, baseline_value in zip(
                origins,
                candidate_macro_by_origin,
                baseline_macro_by_origin,
                strict=True,
            )
        ],
    }


def _verify_paired_validation_evidence(
    value: Any,
    *,
    promotion_origins: tuple[int, ...],
) -> float:
    if (
        not isinstance(value, dict)
        or value.get("method")
        != "paired-circular-moving-block-bootstrap-equal-ba-macro"
        or value.get("samples") != PAIRED_VALIDATION_BOOTSTRAP_SAMPLES
        or value.get("seed") != PAIRED_VALIDATION_BOOTSTRAP_SEED
        or value.get("block_origins") != PAIRED_VALIDATION_BOOTSTRAP_BLOCK_ORIGINS
        or value.get("origin_count") != len(promotion_origins)
        or value.get("noninferiority_ratio_margin")
        != DEFAULT_THRESHOLDS.max_paired_bootstrap_mase_ratio_upper_95
    ):
        raise ValueError("paired validation evidence has an invalid protocol")
    expected_sha = hashlib.sha256(
        np.asarray(promotion_origins, dtype="<i8").tobytes()
    ).hexdigest()
    if value.get("origin_sha256") != expected_sha:
        raise ValueError("paired validation evidence changed origin schedule")
    rows = value.get("per_origin_macro_mase")
    if not isinstance(rows, list) or len(rows) != len(promotion_origins):
        raise ValueError("paired validation evidence has invalid origin rows")
    candidate_values: list[float] = []
    baseline_values: list[float] = []
    actual_origins: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("paired validation evidence has an invalid origin row")
        origin = row.get("origin_utc")
        candidate = _finite_number(row.get("candidate"))
        baseline = _finite_number(row.get("baseline"))
        delta = _finite_number(row.get("candidate_minus_baseline"))
        if (
            not isinstance(origin, str)
            or candidate is None
            or candidate < 0
            or baseline is None
            or baseline < 0
            or delta is None
            or not math.isclose(
                delta,
                candidate - baseline,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("paired validation evidence has an invalid origin row")
        try:
            actual_origins.append(
                int(np.datetime64(origin).astype("datetime64[us]").astype(np.int64))
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "paired validation evidence has an invalid timestamp"
            ) from exc
        candidate_values.append(candidate)
        baseline_values.append(baseline)
    if tuple(actual_origins) != promotion_origins:
        raise ValueError("paired validation evidence origin order changed")
    candidate_array = np.asarray(candidate_values, dtype=np.float64)
    baseline_array = np.asarray(baseline_values, dtype=np.float64)
    point_ratio, ratio_low, ratio_high = _paired_moving_block_ratio_stats(
        candidate_array,
        baseline_array,
    )
    expected_values = {
        "candidate_minus_baseline_macro_mase": float(
            candidate_array.mean() - baseline_array.mean()
        ),
        "candidate_vs_baseline_mase_ratio": point_ratio,
        "mase_ratio_ci_low_95": ratio_low,
        "mase_ratio_ci_high_95": ratio_high,
    }
    if any(
        (actual := _finite_number(value.get(field))) is None
        or not math.isclose(
            actual,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for field, expected in expected_values.items()
    ):
        raise ValueError("paired validation evidence does not recompute")
    return ratio_high


def rejected_overfit_audit(
    reason: str,
    *,
    expected_steps: int,
    thresholds: PromotionThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Build a serializable fail-closed audit when diagnostics cannot execute."""
    return {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "required_bas": list(TRUST_RTO_BAS),
        "thresholds": asdict(thresholds),
        "split_contract": {
            "train": "valid times before 2024-01-01T00:00:00Z",
            "validation": "2024 valid times only",
            "test": "2025+ locked and unopened",
            "test_opened": False,
        },
        "training_behavior": {
            "expected_steps": expected_steps,
            "completed_steps": None,
            "events": [],
        },
        "metrics": {},
        "gates": [
            _gate(
                "diagnostics_completed",
                actual=False,
                operator="==",
                threshold=True,
                passed=False,
            )
        ],
        "promotion_eligible": False,
        "rejection_reasons": [f"diagnostics_completed: {reason}"],
    }


def build_overfit_audit(
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    baseline_validation_metrics: dict[str, Any],
    *,
    training_events: Iterable[dict[str, Any]],
    final_state: dict[str, Any] | None,
    expected_steps: int,
    thresholds: PromotionThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Create an auditable promotion decision from train and validation only."""
    required_bas = TRUST_RTO_BAS
    behavior = summarize_training_history(
        training_events,
        final_state=final_state,
        expected_steps=expected_steps,
    )
    base: dict[str, Any] = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "required_bas": list(required_bas),
        "thresholds": asdict(thresholds),
        "split_contract": {
            "train": "valid times before 2024-01-01T00:00:00Z",
            "validation": "2024 valid times only",
            "test": "2025+ locked and unopened",
            "test_opened": False,
        },
        "training_behavior": behavior,
    }

    try:
        _validate_evaluation_contract(
            train_metrics,
            label="train",
            split="train",
            require_origin_metrics=False,
        )
        _validate_evaluation_contract(
            validation_metrics,
            label="validation",
            split="val",
            require_origin_metrics=True,
        )
        _validate_evaluation_contract(
            baseline_validation_metrics,
            label="baseline validation",
            split="val",
            require_origin_metrics=True,
        )
        train_per_ba = train_metrics.get("per_ba")
        validation_per_ba = validation_metrics.get("per_ba")
        baseline_per_ba = baseline_validation_metrics.get("per_ba")
        if not all(
            isinstance(value, dict)
            for value in (train_per_ba, validation_per_ba, baseline_per_ba)
        ):
            raise ValueError("train, validation, and baseline metrics require per_ba objects")
        if any(
            set(value) != set(required_bas)
            for value in (train_per_ba, validation_per_ba, baseline_per_ba)
        ):
            raise ValueError("diagnostics must contain exactly the seven trust RTOs")
        for label, metrics, per_ba in (
            ("train", train_metrics, train_per_ba),
            ("validation", validation_metrics, validation_per_ba),
            (
                "baseline validation",
                baseline_validation_metrics,
                baseline_per_ba,
            ),
        ):
            origin_hashes = {
                _origin_contract(per_ba, ba)["sha256"] for ba in required_bas
            }
            schedule = metrics["origin_schedule"]
            if len(origin_hashes) != 1 or schedule["origin_sha256"] not in origin_hashes:
                raise ValueError(f"{label} metrics do not share one origin schedule")

        per_rto: dict[str, dict[str, Any]] = {}
        for ba in required_bas:
            train_mase = _metric(train_per_ba, ba, "mase")
            validation_mase = _metric(validation_per_ba, ba, "mase")
            train_wis = _metric(train_per_ba, ba, "wis_scaled")
            validation_wis = _metric(validation_per_ba, ba, "wis_scaled")
            baseline_mase = _metric(baseline_per_ba, ba, "mase")
            baseline_wis = _metric(baseline_per_ba, ba, "wis_scaled")
            train_origins = _origin_contract(train_per_ba, ba)
            validation_origins = _origin_contract(validation_per_ba, ba)
            baseline_origins = _origin_contract(baseline_per_ba, ba)
            if validation_origins != baseline_origins:
                raise ValueError(
                    f"{ba} candidate and baseline validation origins are not identical"
                )
            per_rto[ba] = {
                "train": {
                    "mase": train_mase,
                    "wis_scaled": train_wis,
                    "n_windows": _windows(train_per_ba, ba),
                    "n_points": _step(train_per_ba[ba].get("n_points")),
                    "origins": train_origins,
                },
                "validation": {
                    "mase": validation_mase,
                    "wis_scaled": validation_wis,
                    "n_windows": _windows(validation_per_ba, ba),
                    "n_points": _step(validation_per_ba[ba].get("n_points")),
                    "origins": validation_origins,
                },
                "baseline_validation": {
                    "mase": baseline_mase,
                    "wis_scaled": baseline_wis,
                    "n_windows": _windows(baseline_per_ba, ba),
                    "n_points": _step(baseline_per_ba[ba].get("n_points")),
                    "origins": baseline_origins,
                },
                "generalization": {
                    "mase_absolute_gap": validation_mase - train_mase,
                    "mase_ratio": _safe_ratio(validation_mase, train_mase),
                    "wis_scaled_absolute_gap": validation_wis - train_wis,
                    "wis_scaled_ratio": _safe_ratio(validation_wis, train_wis),
                    "mase_vs_baseline_ratio": _safe_ratio(
                        validation_mase, baseline_mase
                    ),
                    "wis_scaled_vs_baseline_ratio": _safe_ratio(
                        validation_wis, baseline_wis
                    ),
                },
            }

        paired_validation = _paired_validation_bootstrap(
            validation_per_ba,
            baseline_per_ba,
            noninferiority_ratio_margin=(
                thresholds.max_paired_bootstrap_mase_ratio_upper_95
            ),
        )

        train_mase_values = np.asarray(
            [per_rto[ba]["train"]["mase"] for ba in required_bas], dtype=np.float64
        )
        validation_mase_values = np.asarray(
            [per_rto[ba]["validation"]["mase"] for ba in required_bas],
            dtype=np.float64,
        )
        train_wis_values = np.asarray(
            [per_rto[ba]["train"]["wis_scaled"] for ba in required_bas],
            dtype=np.float64,
        )
        validation_wis_values = np.asarray(
            [per_rto[ba]["validation"]["wis_scaled"] for ba in required_bas],
            dtype=np.float64,
        )
        baseline_mase_values = np.asarray(
            [per_rto[ba]["baseline_validation"]["mase"] for ba in required_bas],
            dtype=np.float64,
        )
        baseline_wis_values = np.asarray(
            [
                per_rto[ba]["baseline_validation"]["wis_scaled"]
                for ba in required_bas
            ],
            dtype=np.float64,
        )
        train_macro_mase = float(train_mase_values.mean())
        validation_macro_mase = float(validation_mase_values.mean())
        train_macro_wis = float(train_wis_values.mean())
        validation_macro_wis = float(validation_wis_values.mean())
        baseline_macro_mase = float(baseline_mase_values.mean())
        baseline_macro_wis = float(baseline_wis_values.mean())
        macro_mase_ratio = _safe_ratio(validation_macro_mase, train_macro_mase)
        macro_wis_ratio = _safe_ratio(validation_macro_wis, train_macro_wis)
        macro_mase_vs_baseline = _safe_ratio(
            validation_macro_mase, baseline_macro_mase
        )
        macro_wis_vs_baseline = _safe_ratio(validation_macro_wis, baseline_macro_wis)
        validation_mase_std = float(validation_mase_values.std())
        validation_mase_cv = _safe_ratio(validation_mase_std, validation_macro_mase)
        worst_validation_ba = max(
            required_bas, key=lambda ba: per_rto[ba]["validation"]["mase"]
        )
        valid_rto_ratios = {
            ba: per_rto[ba]["generalization"]["mase_ratio"] for ba in required_bas
        }
        if any(value is None for value in valid_rto_ratios.values()):
            worst_ratio_ba = None
            worst_ratio = None
        else:
            worst_ratio_ba = max(
                required_bas, key=lambda ba: float(valid_rto_ratios[ba])
            )
            worst_ratio = float(valid_rto_ratios[worst_ratio_ba])
        baseline_rto_ratios = {
            ba: per_rto[ba]["generalization"]["mase_vs_baseline_ratio"]
            for ba in required_bas
        }
        if any(value is None for value in baseline_rto_ratios.values()):
            worst_baseline_ratio_ba = None
            worst_baseline_ratio = None
        else:
            worst_baseline_ratio_ba = max(
                required_bas, key=lambda ba: float(baseline_rto_ratios[ba])
            )
            worst_baseline_ratio = float(
                baseline_rto_ratios[worst_baseline_ratio_ba]
            )

        metrics = {
            "train": {
                "macro_mase": train_macro_mase,
                "macro_wis_scaled": train_macro_wis,
            },
            "validation": {
                "macro_mase": validation_macro_mase,
                "macro_wis_scaled": validation_macro_wis,
                "mase_std": validation_mase_std,
                "mase_cv": validation_mase_cv,
                "worst_rto": worst_validation_ba,
                "worst_rto_mase": per_rto[worst_validation_ba]["validation"]["mase"],
            },
            "baseline_validation": {
                "macro_mase": baseline_macro_mase,
                "macro_wis_scaled": baseline_macro_wis,
            },
            "paired_validation": paired_validation,
            "generalization": {
                "macro_mase_absolute_gap": validation_macro_mase - train_macro_mase,
                "macro_mase_ratio": macro_mase_ratio,
                "macro_wis_scaled_absolute_gap": validation_macro_wis - train_macro_wis,
                "macro_wis_scaled_ratio": macro_wis_ratio,
                "worst_rto_ratio_ba": worst_ratio_ba,
                "worst_rto_mase_ratio": worst_ratio,
                "macro_mase_vs_baseline_ratio": macro_mase_vs_baseline,
                "macro_wis_scaled_vs_baseline_ratio": macro_wis_vs_baseline,
                "worst_rto_baseline_ratio_ba": worst_baseline_ratio_ba,
                "worst_rto_mase_vs_baseline_ratio": worst_baseline_ratio,
            },
            "per_rto": per_rto,
        }
    except (KeyError, TypeError, ValueError) as exc:
        base.update(
            {
                "metrics": {},
                "gates": [
                    _gate(
                        "metrics_contract_valid",
                        actual=False,
                        operator="==",
                        threshold=True,
                        passed=False,
                    )
                ],
                "promotion_eligible": False,
                "rejection_reasons": [f"metrics_contract_valid: {exc}"],
            }
        )
        return base

    minimum_windows = min(
        min(
            row[split]["n_windows"]
            for split in ("train", "validation", "baseline_validation")
        )
        for row in per_rto.values()
    )
    gates = [
        _gate(
            "diagnostic_windows_per_ba",
            actual=minimum_windows,
            operator="==",
            threshold=thresholds.required_diagnostic_windows_per_ba,
            passed=minimum_windows == thresholds.required_diagnostic_windows_per_ba,
        ),
        _gate(
            "validation_macro_mase",
            actual=validation_macro_mase,
            operator="<=",
            threshold=thresholds.max_validation_macro_mase,
            passed=validation_macro_mase <= thresholds.max_validation_macro_mase,
        ),
        _gate(
            "macro_mase_generalization_ratio",
            actual=macro_mase_ratio,
            operator="<=",
            threshold=thresholds.max_macro_mase_generalization_ratio,
            passed=(
                macro_mase_ratio is not None
                and macro_mase_ratio <= thresholds.max_macro_mase_generalization_ratio
            ),
        ),
        _gate(
            "macro_wis_generalization_ratio",
            actual=macro_wis_ratio,
            operator="<=",
            threshold=thresholds.max_macro_wis_generalization_ratio,
            passed=(
                macro_wis_ratio is not None
                and macro_wis_ratio <= thresholds.max_macro_wis_generalization_ratio
            ),
        ),
        _gate(
            "worst_rto_validation_mase",
            actual=per_rto[worst_validation_ba]["validation"]["mase"],
            operator="<=",
            threshold=thresholds.max_worst_rto_validation_mase,
            passed=(
                per_rto[worst_validation_ba]["validation"]["mase"]
                <= thresholds.max_worst_rto_validation_mase
            ),
        ),
        _gate(
            "worst_rto_mase_generalization_ratio",
            actual=worst_ratio,
            operator="<=",
            threshold=thresholds.max_worst_rto_mase_generalization_ratio,
            passed=(
                worst_ratio is not None
                and worst_ratio <= thresholds.max_worst_rto_mase_generalization_ratio
            ),
        ),
        _gate(
            "validation_mase_cv",
            actual=validation_mase_cv,
            operator="<=",
            threshold=thresholds.max_validation_mase_cv,
            passed=(
                validation_mase_cv is not None
                and validation_mase_cv <= thresholds.max_validation_mase_cv
            ),
        ),
        _gate(
            "validation_mase_vs_baseline_ratio",
            actual=macro_mase_vs_baseline,
            operator="<=",
            threshold=thresholds.max_validation_mase_vs_baseline_ratio,
            passed=(
                macro_mase_vs_baseline is not None
                and macro_mase_vs_baseline
                <= thresholds.max_validation_mase_vs_baseline_ratio
            ),
        ),
        _gate(
            "validation_wis_vs_baseline_ratio",
            actual=macro_wis_vs_baseline,
            operator="<=",
            threshold=thresholds.max_validation_wis_vs_baseline_ratio,
            passed=(
                macro_wis_vs_baseline is not None
                and macro_wis_vs_baseline
                <= thresholds.max_validation_wis_vs_baseline_ratio
            ),
        ),
        _gate(
            "worst_rto_mase_vs_baseline_ratio",
            actual=worst_baseline_ratio,
            operator="<=",
            threshold=thresholds.max_worst_rto_mase_vs_baseline_ratio,
            passed=(
                worst_baseline_ratio is not None
                and worst_baseline_ratio
                <= thresholds.max_worst_rto_mase_vs_baseline_ratio
            ),
        ),
        _gate(
            "paired_bootstrap_mase_ratio_upper_95",
            actual=paired_validation["mase_ratio_ci_high_95"],
            operator="<=",
            threshold=thresholds.max_paired_bootstrap_mase_ratio_upper_95,
            passed=(
                paired_validation["mase_ratio_ci_high_95"]
                <= thresholds.max_paired_bootstrap_mase_ratio_upper_95
            ),
        ),
        _gate(
            "training_completed_steps",
            actual=behavior["completed_steps"],
            operator="==",
            threshold=expected_steps,
            passed=behavior["completed_steps"] == expected_steps,
        ),
        _gate(
            "train_loss_log_count",
            actual=behavior["train_loss_log_count"],
            operator=">=",
            threshold=thresholds.min_train_loss_logs,
            passed=behavior["train_loss_log_count"] >= thresholds.min_train_loss_logs,
        ),
        _gate(
            "eval_checkpoint_count",
            actual=behavior["eval_checkpoint_count"],
            operator=">=",
            threshold=thresholds.min_eval_checkpoints,
            passed=behavior["eval_checkpoint_count"] >= thresholds.min_eval_checkpoints,
        ),
        _gate(
            "eval_loss_rebound_ratio",
            actual=behavior["eval_loss_rebound_ratio"],
            operator="<=",
            threshold=thresholds.max_eval_loss_rebound_ratio,
            passed=(
                behavior["eval_loss_rebound_ratio"] is not None
                and behavior["eval_loss_rebound_ratio"]
                <= thresholds.max_eval_loss_rebound_ratio
            ),
        ),
        _gate(
            "reported_best_matches_observed",
            actual=behavior["reported_best_matches_observed"],
            operator="==",
            threshold=True,
            passed=behavior["reported_best_matches_observed"] is True,
        ),
        _gate(
            "reported_best_checkpoint_was_saved",
            actual=behavior["reported_best_checkpoint_was_saved"],
            operator="==",
            threshold=True,
            passed=behavior["reported_best_checkpoint_was_saved"] is True,
        ),
    ]
    eligible = all(gate["passed"] for gate in gates)
    base.update(
        {
            "metrics": metrics,
            "gates": gates,
            "promotion_eligible": eligible,
            "rejection_reasons": [gate["name"] for gate in gates if not gate["passed"]],
        }
    )
    return base
