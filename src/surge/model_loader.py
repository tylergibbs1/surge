"""Pinned Chronos-2 loading for base models and portable LoRA adapters."""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from collections import OrderedDict
from os import PathLike
from pathlib import Path
from typing import Any

_FileState = tuple[str, int, int, int, int, int, int]
_RootState = tuple[int, int, int, int, int, int]
_ArtifactSnapshot = tuple[str, str, _RootState, tuple[_FileState, ...]]
_ARTIFACT_DIGEST_CACHE_SIZE = 8
_artifact_digest_cache: OrderedDict[_ArtifactSnapshot, str] = OrderedDict()
_artifact_digest_lock = threading.RLock()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root_state(metadata: os.stat_result) -> _RootState:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _file_state(root: Path, path: Path) -> _FileState:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"model artifact symlinks are not allowed: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"model artifact contains a non-regular file: {path}")
    return (
        path.relative_to(root).as_posix(),
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _artifact_snapshot(path: Path) -> _ArtifactSnapshot:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"model artifact symlinks are not allowed: {path}")
    root_state = _root_state(metadata)
    if stat.S_ISREG(metadata.st_mode):
        return (str(path), "file", root_state, (_file_state(path.parent, path),))
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"model artifact is not a file or directory: {path}")

    files: list[_FileState] = []
    for current, directories, filenames in os.walk(path, topdown=True, followlinks=False):
        directories.sort()
        filenames.sort()
        current_path = Path(current)
        for name in directories:
            candidate = current_path / name
            candidate_metadata = candidate.lstat()
            if stat.S_ISLNK(candidate_metadata.st_mode):
                raise ValueError(f"model artifact symlinks are not allowed: {candidate}")
            if not stat.S_ISDIR(candidate_metadata.st_mode):
                raise ValueError(
                    f"model artifact contains a non-directory entry: {candidate}"
                )
        files.extend(_file_state(path, current_path / name) for name in filenames)
    if not files:
        raise ValueError(f"model artifact directory is empty: {path}")
    return (str(path), "directory", root_state, tuple(sorted(files)))


def _digest_snapshot(snapshot: _ArtifactSnapshot) -> str:
    root_text, kind, _, files = snapshot
    root = Path(root_text)
    if kind == "file":
        return _sha256_file(root)

    digest = hashlib.sha256()
    for relative_text, *_ in files:
        relative = relative_text.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(root / relative_text)))
    return digest.hexdigest()


def clear_artifact_sha256_cache() -> None:
    """Explicitly invalidate cached local artifact digests."""
    with _artifact_digest_lock:
        _artifact_digest_cache.clear()


def artifact_sha256(model: str | PathLike[str]) -> str:
    """Return a cached, deterministic SHA-256 for a local model artifact.

    Directory identities bind both each relative file path and that file's
    content digest. This makes renames, additions, deletions, and byte-level
    changes visible while remaining stable across filesystem traversal order.
    Cached content digests are reused only when the complete metadata snapshot
    is unchanged. A post-hash snapshot prevents caching a concurrently-mutated
    tree, while symlink rejection keeps every hashed byte inside the artifact.
    """
    path = Path(os.path.abspath(Path(model).expanduser()))
    with _artifact_digest_lock:
        for _ in range(3):
            snapshot = _artifact_snapshot(path)
            cached = _artifact_digest_cache.get(snapshot)
            if cached is not None:
                _artifact_digest_cache.move_to_end(snapshot)
                return cached

            digest = _digest_snapshot(snapshot)
            if snapshot != _artifact_snapshot(path):
                continue
            _artifact_digest_cache[snapshot] = digest
            _artifact_digest_cache.move_to_end(snapshot)
            while len(_artifact_digest_cache) > _ARTIFACT_DIGEST_CACHE_SIZE:
                _artifact_digest_cache.popitem(last=False)
            return digest
    raise RuntimeError(f"model artifact changed repeatedly while hashing: {path}")


def load_chronos2(
    model: str | PathLike[str],
    *,
    revision: str | None = None,
    **kwargs: Any,
) -> Any:
    """Load a Chronos-2 base or adapter with the narrow PEFT import allowlist.

    PEFT 0.20 protects dynamic adapter imports. Surge adapters legitimately
    reference Chronos2Model, so allow only that exact module when an adapter
    config is present. Passing this option to a base model is invalid, hence
    the explicit preflight detection.
    """
    from chronos import Chronos2Pipeline
    from transformers.utils.peft_utils import find_adapter_config_file

    model_ref = str(model)
    is_adapter = find_adapter_config_file(model_ref, revision=revision) is not None
    if revision is not None:
        kwargs["revision"] = revision
    if is_adapter:
        kwargs["import_allowlist"] = ["chronos.chronos2.model"]
    return Chronos2Pipeline.from_pretrained(model_ref, **kwargs)
