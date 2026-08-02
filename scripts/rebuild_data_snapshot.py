"""Build or verify a compact, checksummed API recovery snapshot.

The builder reads ``SURGE_DATA_DIR``, canonicalizes overlapping load/weather
rows, and writes one deterministic parquet file per dataset/month. Immutable
ledger parquet files are copied byte-for-byte so issuance identities and their
content hashes survive recovery. It stages the complete output before replacing
``--output`` so a failed build never leaves a half-written recovery artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from surge import __version__, bas, store

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data_snapshot"
CANONICAL_DATASET_KEYS = {
    "load_hourly": ["ts_utc", "ba"],
    "weather_hourly": ["ts_utc", "ba"],
}
LEDGER_DATASETS = (
    "forecast_points",
    "forecast_issuances",
    "forecast_runs",
    "forecast_verifications",
)
SNAPSHOT_DATASETS = (*CANONICAL_DATASET_KEYS, *LEDGER_DATASETS)
DATASET_VALUE_COLUMNS = {
    "load_hourly": "load_mw",
    "weather_hourly": "temp_c",
}
DATASET_SOURCES = {
    "load_hourly": {
        "provider": "EIA-930",
        "url": "https://api.eia.gov/v2/electricity/rto/region-data/data/",
    },
    "weather_hourly": {
        "provider": "ASOS / Iowa Environmental Mesonet",
        "url": "https://mesonet.agron.iastate.edu/",
    },
}
FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = {1, FORMAT_VERSION}
MIN_INPUT_ROWS_PER_BA = 2_049
TRUST_LEDGER_BAS = tuple(bas.rto_codes())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _safe_output(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"refusing symlink snapshot output: {expanded}")
    resolved = expanded.resolve()
    if resolved == Path(resolved.anchor) or resolved == ROOT:
        raise ValueError(f"refusing unsafe snapshot output: {resolved}")
    return resolved


def _canonical_dataset(name: str) -> pl.DataFrame:
    dataset_root = store.dataset_path(name)
    if not dataset_root.exists() or not any(dataset_root.rglob("*.parquet")):
        return pl.DataFrame()
    df = store.scan(name, dedupe_on=CANONICAL_DATASET_KEYS[name]).collect()
    if df.is_empty():
        return df
    if "ts_utc" not in df.columns:
        raise ValueError(f"{name} has no ts_utc column")
    if "ba" not in df.columns:
        raise ValueError(f"{name} has no ba column")
    value_column = DATASET_VALUE_COLUMNS[name]
    if value_column not in df.columns:
        raise ValueError(f"{name} has no {value_column} column")
    if "year" not in df.columns or "month" not in df.columns:
        df = df.with_columns(
            pl.col("ts_utc").dt.year().alias("year"),
            pl.col("ts_utc").dt.month().alias("month"),
        )
    return df.sort(["ts_utc", "ba"])


def _validate_dataset(
    name: str,
    df: pl.DataFrame,
    *,
    max_age_hours: float | None,
    allow_incomplete: bool,
    require_all_demand_bas: bool,
) -> None:
    if df.is_empty():
        if name == "load_hourly" or not allow_incomplete:
            raise ValueError(f"{name} is empty; refusing to build a recovery snapshot")
        return

    value_column = DATASET_VALUE_COLUMNS[name]
    now = datetime.now(tz=UTC)
    usable = pl.col(value_column).is_not_null() & pl.col(value_column).is_finite()
    if "as_of" in df.columns:
        usable &= pl.col("as_of") <= now
    coverage = (
        df.group_by("ba")
        .agg(
            pl.len().alias("rows"),
            usable.sum().alias("usable_rows"),
            pl.col("ts_utc")
            .filter(usable)
            .max()
            .alias("latest_usable"),
        )
        .sort("ba")
    )
    expected = set(bas.demand_codes() if require_all_demand_bas else TRUST_LEDGER_BAS)
    observed = set(coverage["ba"].to_list())
    missing = sorted(expected - observed)
    short = sorted(
        row["ba"]
        for row in coverage.iter_rows(named=True)
        if row["ba"] in expected
        and int(row["usable_rows"]) < MIN_INPUT_ROWS_PER_BA
    )

    if "as_of" in df.columns:
        future_availability = df.filter(pl.col("as_of") > now + timedelta(hours=2))
        if not future_availability.is_empty():
            raise ValueError(f"{name}: contains future-dated availability timestamps")

    stale: list[str] = []
    if max_age_hours is not None:
        if max_age_hours <= 0:
            raise ValueError("max_input_age_hours must be positive")
        cutoff = now - timedelta(hours=max_age_hours)
        stale = sorted(
            row["ba"]
            for row in coverage.iter_rows(named=True)
            if row["ba"] in expected
            and row["latest_usable"] is not None
            and row["latest_usable"] < cutoff
        )

    future = sorted(
        row["ba"]
        for row in coverage.iter_rows(named=True)
        if row["ba"] in expected
        and row["latest_usable"] is not None
        and row["latest_usable"] > now + timedelta(hours=2)
    )
    if future:
        raise ValueError(f"{name}: future-dated BAs={','.join(future)}")

    if allow_incomplete:
        return
    problems = []
    if missing:
        problems.append(f"missing BAs={','.join(missing)}")
    if short:
        problems.append(
            f"BAs with fewer than {MIN_INPUT_ROWS_PER_BA} usable rows={','.join(short)}"
        )
    if stale:
        problems.append(f"stale BAs={','.join(stale)}")
    if problems:
        raise ValueError(f"{name}: " + "; ".join(problems))


def _dataset_summary(df: pl.DataFrame) -> dict[str, Any]:
    if df.is_empty():
        return {
            "rows": 0,
            "ba_count": 0,
            "start_utc": None,
            "end_utc": None,
            "max_available_at_utc": None,
            "sources": [],
        }
    result = {
        "rows": df.height,
        "ba_count": df["ba"].n_unique(),
        "start_utc": _iso(df["ts_utc"].min()),
        "end_utc": _iso(df["ts_utc"].max()),
        "max_available_at_utc": _iso(df["as_of"].max()) if "as_of" in df.columns else None,
        "sources": (
            sorted(str(value) for value in df["source"].drop_nulls().unique())
            if "source" in df.columns
            else []
        ),
    }
    return result


def _inventory_sha256(entries: list[dict[str, Any]]) -> str:
    inventory = [
        {
            "path": item["path"],
            "rows": int(item["rows"]),
            "bytes": int(item["bytes"]),
            "sha256": item["sha256"],
        }
        for item in sorted(entries, key=lambda value: value["path"])
    ]
    payload = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _copy_ledger_dataset(
    name: str,
    stage: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Copy an immutable ledger dataset without re-encoding its Parquet bytes."""
    source_root = store.dataset_path(name)
    entries: list[dict[str, Any]] = []
    if source_root.exists():
        if source_root.is_symlink() or not source_root.is_dir():
            raise ValueError(f"{name} must be a regular directory")
        resolved_root = source_root.resolve()
        for source in sorted(source_root.rglob("*.parquet")):
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"{name} contains a non-regular parquet file")
            resolved = source.resolve()
            if not resolved.is_relative_to(resolved_root):
                raise ValueError(f"{name} contains a parquet file outside its root")
            relative_inside = source.relative_to(source_root)
            destination = stage / name / relative_inside
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            rows = pl.read_parquet(source).height
            entries.append(
                {
                    "path": destination.relative_to(stage).as_posix(),
                    "rows": rows,
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                    "storage_mode": "verbatim",
                }
            )
    summary = {
        "rows": sum(int(item["rows"]) for item in entries),
        "file_count": len(entries),
        "bytes": sum(int(item["bytes"]) for item in entries),
        "storage_mode": "verbatim",
        "content_sha256": _inventory_sha256(entries),
    }
    return summary, entries


def build_snapshot(
    output: Path,
    *,
    force: bool,
    max_input_age_hours: float | None,
    allow_incomplete: bool,
    require_all_demand_bas: bool,
) -> None:
    output = _safe_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    if output.exists() and not output.is_dir():
        raise ValueError(f"snapshot output is not a directory: {output}")
    if (
        output.exists()
        and force
        and not (output / "snapshot-manifest.json").is_file()
    ):
        raise ValueError(
            f"refusing to replace non-snapshot directory: {output}; "
            "choose a new path or provide a verified snapshot directory"
        )

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "created_at_utc": datetime.now(tz=UTC).isoformat(),
            "surge_grid_version": __version__,
            "source_catalog": DATASET_SOURCES,
            "validation": {
                "expected_load_bas": (
                    list(bas.demand_codes())
                    if require_all_demand_bas
                    else list(TRUST_LEDGER_BAS)
                ),
                "minimum_input_rows_per_ba": MIN_INPUT_ROWS_PER_BA,
                "max_input_age_hours": max_input_age_hours,
                "allow_incomplete": allow_incomplete,
                "require_all_demand_bas": require_all_demand_bas,
            },
            "datasets": {},
            "files": [],
        }

        for dataset in CANONICAL_DATASET_KEYS:
            df = _canonical_dataset(dataset)
            _validate_dataset(
                dataset,
                df,
                max_age_hours=max_input_age_hours,
                allow_incomplete=allow_incomplete,
                require_all_demand_bas=require_all_demand_bas,
            )

            manifest["datasets"][dataset] = _dataset_summary(df)
            if df.is_empty():
                continue

            for (year, month), part in df.group_by(
                ["year", "month"], maintain_order=True
            ):
                dest = (
                    stage
                    / dataset
                    / f"year={int(year):04d}"
                    / f"month={int(month):02d}"
                    / "part.parquet"
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                part.drop(["year", "month"]).write_parquet(dest, compression="zstd")
                manifest["files"].append(
                    {
                        "path": dest.relative_to(stage).as_posix(),
                        "rows": part.height,
                        "bytes": dest.stat().st_size,
                        "sha256": _sha256(dest),
                        "storage_mode": "canonical",
                    }
                )

        for dataset in LEDGER_DATASETS:
            summary, entries = _copy_ledger_dataset(dataset, stage)
            manifest["datasets"][dataset] = summary
            manifest["files"].extend(entries)

        manifest["files"].sort(key=lambda item: item["path"])
        manifest["content_sha256"] = _inventory_sha256(manifest["files"])
        (stage / "snapshot-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        backup: Path | None = None
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}-previous-", dir=output.parent))
            backup.rmdir()
            output.replace(backup)
        try:
            stage.replace(output)
        except BaseException:
            if backup is not None and backup.exists() and not output.exists():
                backup.replace(output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        print(
            f"snapshot built: {output} "
            f"({len(manifest['files'])} parquet files, "
            f"{manifest['datasets']['load_hourly']['rows']:,} load rows)"
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify_snapshot(path: Path) -> None:
    root = path.expanduser().resolve()
    manifest_path = root / "snapshot-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    format_version = manifest.get("format_version")
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"unsupported snapshot format {manifest.get('format_version')!r}"
        )

    raw_entries = manifest.get("files", [])
    if not isinstance(raw_entries, list):
        raise ValueError("snapshot manifest files must be a list")
    entries: dict[str, dict[str, Any]] = {}
    for item in raw_entries:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("snapshot manifest contains an invalid file entry")
        relative = item["path"]
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe snapshot file path: {relative}")
        if relative in entries:
            raise ValueError(f"duplicate snapshot file entry: {relative}")
        entries[relative] = item
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*.parquet")
        if item.is_file()
    }
    expected = set(entries)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"snapshot file mismatch: missing={missing}, extra={extra}")

    for relative, entry in sorted(entries.items()):
        dataset = Path(relative).parts[0] if Path(relative).parts else ""
        allowed_datasets = (
            set(CANONICAL_DATASET_KEYS)
            if format_version == 1
            else set(SNAPSHOT_DATASETS)
        )
        if dataset not in allowed_datasets:
            raise ValueError(f"snapshot file is outside a supported dataset: {relative}")
        file_path = root / relative
        if file_path.is_symlink() or not file_path.is_file():
            raise ValueError(f"snapshot entry is not a regular file: {relative}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {relative}")
        if "bytes" in entry and file_path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"byte-count mismatch: {relative}")
        rows = pl.read_parquet(file_path).height
        if rows != int(entry["rows"]):
            raise ValueError(
                f"row-count mismatch: {relative} expected={entry['rows']} actual={rows}"
            )

    datasets = manifest.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError("snapshot manifest datasets must be an object")
    required_datasets = (
        tuple(CANONICAL_DATASET_KEYS) if format_version == 1 else SNAPSHOT_DATASETS
    )
    for dataset in required_datasets:
        summary = datasets.get(dataset)
        if not isinstance(summary, dict):
            raise ValueError(f"snapshot manifest has no {dataset} summary")
        declared_rows = int(summary.get("rows", -1))
        indexed_rows = sum(
            int(entry["rows"])
            for relative, entry in entries.items()
            if relative.startswith(f"{dataset}/")
        )
        if declared_rows != indexed_rows:
            raise ValueError(
                f"dataset row-count mismatch: {dataset} "
                f"declared={declared_rows} indexed={indexed_rows}"
            )

        if dataset in LEDGER_DATASETS and format_version == FORMAT_VERSION:
            dataset_entries = [
                entry
                for relative, entry in entries.items()
                if relative.startswith(f"{dataset}/")
            ]
            if summary.get("storage_mode") != "verbatim":
                raise ValueError(f"ledger dataset is not marked verbatim: {dataset}")
            if int(summary.get("file_count", -1)) != len(dataset_entries):
                raise ValueError(f"ledger dataset file-count mismatch: {dataset}")
            expected_content_sha = _inventory_sha256(dataset_entries)
            if summary.get("content_sha256") != expected_content_sha:
                raise ValueError(f"ledger dataset content checksum mismatch: {dataset}")

    if (
        format_version == FORMAT_VERSION
        and manifest.get("content_sha256")
        != _inventory_sha256(list(entries.values()))
    ):
        raise ValueError("snapshot content checksum mismatch")

    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("snapshot manifest has no validation policy")
    required_policy = {
        "max_input_age_hours",
        "allow_incomplete",
        "require_all_demand_bas",
    }
    if missing_policy := required_policy - set(validation):
        raise ValueError(f"snapshot validation policy missing: {sorted(missing_policy)}")
    max_input_age_hours = validation["max_input_age_hours"]
    if max_input_age_hours is not None and not isinstance(max_input_age_hours, (int, float)):
        raise ValueError("snapshot max_input_age_hours must be numeric or null")
    allow_incomplete = validation["allow_incomplete"]
    require_all_demand_bas = validation["require_all_demand_bas"]
    if not isinstance(allow_incomplete, bool) or not isinstance(require_all_demand_bas, bool):
        raise ValueError("snapshot validation flags must be booleans")

    for dataset in CANONICAL_DATASET_KEYS:
        paths = [root / relative for relative in sorted(entries) if relative.startswith(f"{dataset}/")]
        frame = (
            pl.concat([pl.read_parquet(file_path) for file_path in paths], how="diagonal_relaxed")
            if paths
            else pl.DataFrame()
        )
        _validate_dataset(
            dataset,
            frame,
            max_age_hours=float(max_input_age_hours) if max_input_age_hours is not None else None,
            allow_incomplete=allow_incomplete,
            require_all_demand_bas=require_all_demand_bas,
        )

    load_rows = int(datasets.get("load_hourly", {}).get("rows", 0))
    if load_rows <= 0:
        raise ValueError("snapshot manifest contains no load rows")
    readiness = "research-incomplete" if allow_incomplete else "recovery-ready"
    print(
        f"snapshot verified ({readiness}): {root} "
        f"({len(entries)} parquet files, {load_rows:,} load rows)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--max-input-age-hours",
        "--max-load-age-hours",
        dest="max_input_age_hours",
        type=float,
        default=None,
        help="Fail if any expected BA's newest load or weather input is older than this many hours.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow missing/short BA histories or missing weather (research only).",
    )
    parser.add_argument(
        "--require-all-demand-bas",
        action="store_true",
        help="Require recovery coverage for the legacy 53-BA explorer, not only the seven-RTO ledger.",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="Verify an existing snapshot instead of building one.",
    )
    args = parser.parse_args()
    if args.verify is not None:
        verify_snapshot(args.verify)
        return
    build_snapshot(
        args.output,
        force=args.force,
        max_input_age_hours=args.max_input_age_hours,
        allow_incomplete=args.allow_incomplete,
        require_all_demand_bas=args.require_all_demand_bas,
    )


if __name__ == "__main__":
    main()
