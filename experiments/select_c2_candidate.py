"""Freeze the winner of the predeclared v0.2 H100 candidate comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.overfit import (
    POLICY_VERSION,
    RELEASE_BASE_MODEL_ID,
    RELEASE_BASE_REVISION,
    SELECTION_POLICY_VERSION,
    SELECTION_TRAINING_STEPS,
    TRUST_RTO_BAS,
    validate_frozen_data_snapshot,
    validate_promotion_inputs,
    validate_release_lineage,
    verify_promotion_artifact,
)

BASE_MODEL = RELEASE_BASE_MODEL_ID
BASE_REVISION = RELEASE_BASE_REVISION
EXPECTED_STEPS = SELECTION_TRAINING_STEPS
SELECTION_POLICY = SELECTION_POLICY_VERSION


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite_ratio(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return number


def _selection_score(audit: dict[str, Any]) -> float:
    metrics = audit.get("metrics")
    generalization = metrics.get("generalization") if isinstance(metrics, dict) else None
    if not isinstance(generalization, dict):
        raise ValueError("eligible audit is missing generalization metrics")
    mase_ratio = _finite_ratio(
        generalization.get("macro_mase_vs_baseline_ratio"),
        label="macro MASE/base ratio",
    )
    wis_ratio = _finite_ratio(
        generalization.get("macro_wis_scaled_vs_baseline_ratio"),
        label="macro scaled-WIS/base ratio",
    )
    return 0.5 * mase_ratio + 0.5 * wis_ratio


def choose_candidate(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Apply the frozen score and six-decimal, fewer-steps tie break."""
    eligible = [record for record in records if record.get("promotion_eligible") is True]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda record: (
            round(_finite_ratio(record.get("selection_score"), label="selection score"), 6),
            int(record["num_steps"]),
        ),
    )


def _load_candidate(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.expanduser().resolve()
    manifest_path = root / "surge-training-manifest.json"
    audit_path = root / "surge-overfit-audit.json"
    manifest = _read_object(manifest_path)
    audit = _read_object(audit_path)
    selection = manifest.get("selection")
    split = audit.get("split_contract")
    config = manifest.get("config")
    if not isinstance(selection, dict) or not isinstance(config, dict):
        raise ValueError(f"{root} has an incomplete training manifest")
    if (
        audit.get("policy_version") != POLICY_VERSION
        or selection.get("policy_version") != POLICY_VERSION
        or not isinstance(split, dict)
        or split.get("test_opened") is not False
        or manifest.get("locked_test_opened") is not False
    ):
        raise ValueError(f"{root} is not an unopened v0.2 selection artifact")
    if selection.get("overfit_audit") != audit_path.name:
        raise ValueError(f"{root} manifest references a different overfit audit")
    if selection.get("overfit_audit_sha256") != _sha256_file(audit_path):
        raise ValueError(f"{root} overfit audit checksum does not match its manifest")

    manifest_bas = manifest.get("bas")
    base_revision = manifest.get("base_model_revision")
    code_revision = manifest.get("code_revision")
    data_snapshot_sha256 = manifest.get("data_snapshot_sha256")
    if (
        not isinstance(manifest_bas, list)
        or not all(isinstance(ba, str) for ba in manifest_bas)
        or not isinstance(base_revision, str)
        or not isinstance(code_revision, str)
        or not isinstance(data_snapshot_sha256, str)
    ):
        raise ValueError(f"{root} has malformed immutable training identity")
    bas = validate_promotion_inputs(
        manifest_bas,
        base_revision=base_revision,
        code_revision=code_revision,
        data_snapshot_sha256=data_snapshot_sha256,
    )
    if bas != list(TRUST_RTO_BAS):
        raise ValueError(f"{root} has a non-canonical RTO order")
    validate_frozen_data_snapshot(manifest.get("data_snapshot_sha256"))
    if (
        manifest.get("base_model") != BASE_MODEL
        or manifest.get("base_model_revision") != BASE_REVISION
    ):
        raise ValueError(f"{root} does not descend from the release-safe pinned base")
    validate_release_lineage(
        manifest.get("base_model_source"),
        base_model_id=manifest.get("base_model"),
        base_revision=manifest.get("base_model_revision"),
    )

    expected_config = {
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
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"{root} has unexpected {key}={config.get(key)!r}")
    num_steps = config.get("num_steps")
    if isinstance(num_steps, bool) or num_steps not in EXPECTED_STEPS:
        raise ValueError(f"{root} has an undeclared training duration")

    manifest_eligible = selection.get("promotion_eligible")
    audit_eligible = audit.get("promotion_eligible")
    if not isinstance(manifest_eligible, bool) or manifest_eligible is not audit_eligible:
        raise ValueError(f"{root} manifest and audit promotion decisions disagree")
    eligible = manifest_eligible
    promotion_path = root / "surge-promotion.json"
    marker_sha256: str | None = None
    score: float | None = None
    if eligible:
        verified = verify_promotion_artifact(
            promotion_path,
            model_path=root / "best",
        )
        marker_sha256 = verified["marker_sha256"]
        score = _selection_score(audit)
    elif promotion_path.exists():
        raise ValueError(f"{root} rejected candidate unexpectedly has a promotion marker")

    record = {
        "candidate": root.name,
        "num_steps": num_steps,
        "promotion_eligible": eligible,
        "selection_score": score,
        "validation_mase_vs_base_ratio": (
            audit.get("metrics", {})
            .get("generalization", {})
            .get("macro_mase_vs_baseline_ratio")
        ),
        "validation_scaled_wis_vs_base_ratio": (
            audit.get("metrics", {})
            .get("generalization", {})
            .get("macro_wis_scaled_vs_baseline_ratio")
        ),
        "rejection_reasons": audit.get("rejection_reasons", []),
        "manifest_sha256": _sha256_file(manifest_path),
        "overfit_audit_sha256": _sha256_file(audit_path),
        "promotion_marker_sha256": marker_sha256,
    }
    identity = {
        "base_model": manifest["base_model"],
        "base_revision": base_revision,
        "code_revision": code_revision,
        "data_snapshot_sha256": data_snapshot_sha256,
        "feature_spec_version": manifest.get("feature_spec_version"),
        "feature_spec_sha256": manifest.get("feature_spec_sha256"),
        "bas": bas,
        "versions": manifest.get("versions"),
        "runtime": manifest.get("runtime"),
    }
    return record, identity


def _write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to replace frozen selection {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate) != len(EXPECTED_STEPS):
        parser.error("the frozen v0.2 experiment requires exactly two candidates")
    selection_parent = args.out.expanduser().resolve().parent
    if any(path.expanduser().resolve().parent != selection_parent for path in args.candidate):
        parser.error("candidate directories must be siblings of the frozen selection artifact")

    loaded = [_load_candidate(path) for path in args.candidate]
    records = [record for record, _identity in loaded]
    identities = [identity for _record, identity in loaded]
    if {record["num_steps"] for record in records} != set(EXPECTED_STEPS):
        parser.error("candidates must contain the declared 1,000- and 2,000-step runs")
    if any(identity != identities[0] for identity in identities[1:]):
        parser.error("candidate code, data, model, feature, or RTO identities differ")

    winner = choose_candidate(records)
    output = {
        "schema_version": 1,
        "policy_version": SELECTION_POLICY,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "training_identity": identities[0],
        "selection_rule": (
            "minimum 0.5*(validation MASE/base MASE) + "
            "0.5*(validation scaled WIS/base scaled WIS); six-decimal tie chooses fewer steps"
        ),
        "candidates": records,
        "winner": winner,
        "locked_test_opened": False,
        "fallback": (
            None
            if winner is not None
            else "no candidate passed; keep pinned zero-shot serving default and do not open test"
        ),
    }
    _write_exclusive_json(args.out, output)
    print("SELECTION_RESULT:", json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
