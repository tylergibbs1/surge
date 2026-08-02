"""Append-only archive of what EIA-930 actually said, when it said it.

EIA-930 is preliminary on publication and is revised afterwards, with no
published closure date. Every score Surge publishes is therefore a claim about
a *vintage* of the truth, not about the truth. Without an archive of the bytes
EIA served at a given moment, "scored against the data available at +72 hours"
is a convention nobody can check -- including us.

This module stores the raw API response for one balancing authority and window,
exactly as received, under a content hash, and records an immutable index entry
naming the moment it was captured. Nothing is ever overwritten: a later capture
of the same window is a new vintage sitting beside the old one, which is what
makes revisions measurable.

This is the design used by real-time macroeconomic data sets (Croushore & Stark,
*A real-time data set for macroeconomists*, J. Econometrics 105(1), 2001), where
forecast evaluation against revised data is a known source of overstated skill.

A vintage that was not captured cannot be reconstructed later. Every day this
does not run is a day of evidence permanently lost.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path.home() / ".surge" / "vintage"
INDEX_NAME = "index.jsonl"
SCHEMA_VERSION = 1

# Anything matching these keys is stripped from the recorded request before it
# is written, so a credential can never reach the archive.
_SECRET_KEYS = frozenset({"api_key", "apikey", "token", "key", "password", "secret"})


@dataclass(frozen=True)
class CapturedVintage:
    """One immutable observation of what a source said at a moment in time."""

    dataset: str
    ba: str
    start: str
    end: str
    captured_at_utc: str
    payload_sha256: str
    payload_path: Path
    row_count: int
    already_present: bool

    def as_index_entry(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset": self.dataset,
            "ba": self.ba,
            "start": self.start,
            "end": self.end,
            "captured_at_utc": self.captured_at_utc,
            "payload_sha256": self.payload_sha256,
            "row_count": self.row_count,
        }


def archive_root() -> Path:
    return Path(os.environ.get("SURGE_VINTAGE_DIR", DEFAULT_ROOT)).expanduser()


def _redacted(request: dict[str, Any]) -> dict[str, Any]:
    return {
        key: ("<redacted>" if key.lower() in _SECRET_KEYS else value)
        for key, value in request.items()
    }


def capture(
    *,
    dataset: str,
    ba: str,
    start: str,
    end: str,
    rows: list[dict[str, Any]],
    request: dict[str, Any] | None = None,
    root: Path | None = None,
    captured_at: datetime | None = None,
) -> CapturedVintage:
    """Archive one response verbatim and return its immutable identity.

    Storage is content-addressed, so re-capturing an unchanged window costs one
    hash and writes nothing new; ``already_present`` reports that. A window whose
    values changed hashes differently and is stored alongside, never over, the
    earlier vintage.
    """
    if not dataset or not ba:
        raise ValueError("dataset and ba are required")
    captured_at_utc = (captured_at or datetime.now(tz=UTC)).astimezone(UTC).isoformat()
    base = archive_root() if root is None else Path(root).expanduser()

    document = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "ba": ba,
        "start": start,
        "end": end,
        "request": _redacted(request or {}),
        "rows": rows,
    }
    # The hash covers the payload only, never the capture time, so an unchanged
    # window is recognised as unchanged however often it is polled.
    payload = json.dumps(
        {key: value for key, value in document.items() if key != "captured_at_utc"},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()

    payload_path = base / dataset / ba / f"{digest}.json.gz"
    already_present = payload_path.is_file()
    if not already_present:
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = payload_path.with_name(f".{payload_path.name}.{os.getpid()}.tmp")
        with gzip.open(temporary, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, payload_path)

    captured = CapturedVintage(
        dataset=dataset,
        ba=ba,
        start=start,
        end=end,
        captured_at_utc=captured_at_utc,
        payload_sha256=digest,
        payload_path=payload_path,
        row_count=len(rows),
        already_present=already_present,
    )
    _append_index(base, captured)
    return captured


def _append_index(base: Path, captured: CapturedVintage) -> None:
    """Append one line to the index. Append-only: entries are never rewritten."""
    base.mkdir(parents=True, exist_ok=True)
    line = json.dumps(captured.as_index_entry(), sort_keys=True) + "\n"
    with open(base / INDEX_NAME, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_index(root: Path | None = None) -> list[dict[str, Any]]:
    base = archive_root() if root is None else Path(root).expanduser()
    index_path = base / INDEX_NAME
    if not index_path.is_file():
        return []
    entries = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def load_payload(payload_sha256: str, dataset: str, ba: str, root: Path | None = None) -> dict:
    """Read one archived response back, verifying it still hashes to its name."""
    base = archive_root() if root is None else Path(root).expanduser()
    path = base / dataset / ba / f"{payload_sha256}.json.gz"
    try:
        with gzip.open(path, "rb") as handle:
            raw = handle.read()
        actual = hashlib.sha256(raw).hexdigest()
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        # Unreadable and altered are the same failure to a verifier: the bytes
        # on disk are not the evidence this digest names.
        raise ValueError(
            f"archived vintage {payload_sha256} no longer matches its digest"
        ) from exc
    if actual != payload_sha256:
        raise ValueError(f"archived vintage {payload_sha256} no longer matches its digest")
    return json.loads(raw)


def distinct_vintages(dataset: str, ba: str, root: Path | None = None) -> list[dict[str, Any]]:
    """Index entries for one series, oldest capture first, one per distinct payload."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for entry in read_index(root):
        if entry.get("dataset") != dataset or entry.get("ba") != ba:
            continue
        digest = entry["payload_sha256"]
        if digest in seen:
            continue
        seen.add(digest)
        out.append(entry)
    return out
