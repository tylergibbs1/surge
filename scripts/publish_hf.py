"""Upload a fine-tuned Chronos-2 checkpoint to Hugging Face Hub.

Usage:
    HF_TOKEN=... python scripts/publish_hf.py                # latest (v3)
    HF_TOKEN=... python scripts/publish_hf.py --version v2   # republish an older release

Requires HF_TOKEN in env with write scope.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_OWNER = "Tylerbry1"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Version → (local checkpoint dir, commit message tag). Keep older entries
# so `--version v2` can still re-push the 7-BA specialist if ever needed.
VERSIONS: dict[str, tuple[str, str]] = {
    "v2": ("chronos2_full_v2", "initial surge-fm-v2 release"),
    "v3": ("chronos2_full_v3", "initial surge-fm-v3 release (53 BAs)"),
}
LATEST = "v3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=LATEST, choices=list(VERSIONS),
                    help=f"Which checkpoint to publish (default: {LATEST})")
    ap.add_argument("--message", default=None,
                    help="Override commit message")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN in env", file=sys.stderr)
        sys.exit(1)

    local_name, default_msg = VERSIONS[args.version]
    local = REPO_ROOT / "models" / local_name
    if not local.exists():
        print(f"ERROR: {local} not found", file=sys.stderr)
        sys.exit(1)

    repo = f"{REPO_OWNER}/surge-fm-{args.version}"
    msg = args.message or f"feat: {default_msg}"

    api = HfApi(token=token)
    print(f"creating repo {repo} (exist_ok=True)…", flush=True)
    create_repo(repo, token=token, exist_ok=True, repo_type="model")
    print(f"uploading {local} → {repo}…", flush=True)
    commit = api.upload_folder(
        folder_path=str(local),
        repo_id=repo,
        repo_type="model",
        commit_message=msg,
        # Skip bulky / irrelevant artefacts. eval_*.json is kept — useful
        # as a pinned record of the numbers that shipped with this commit.
        ignore_patterns=["*.pyc", "__pycache__/*", ".DS_Store"],
    )
    sha = getattr(commit, "oid", None) or getattr(commit, "commit_oid", None)
    print(f"done → https://huggingface.co/{repo}")
    if sha:
        # Pin this exact SHA in `surge.api.forecaster.MODEL_REVISION` so
        # loaders never drift from what we shipped.
        print(f"commit SHA (pin this in forecaster.MODEL_REVISION): {sha}")


if __name__ == "__main__":
    main()
