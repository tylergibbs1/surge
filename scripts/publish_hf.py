"""Upload the fine-tuned Chronos-2 checkpoint to Hugging Face Hub.

Requires HF_TOKEN in env with write scope.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO = "Tylerbry1/surge-fm-v2"
LOCAL = Path(__file__).resolve().parents[1] / "models" / "chronos2_full_v2"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN in env", file=sys.stderr)
        sys.exit(1)
    if not LOCAL.exists():
        print(f"ERROR: {LOCAL} not found", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    print(f"creating repo {REPO} (exist_ok=True)…", flush=True)
    create_repo(REPO, token=token, exist_ok=True, repo_type="model")
    print("uploading folder…", flush=True)
    api.upload_folder(
        folder_path=str(LOCAL),
        repo_id=REPO,
        repo_type="model",
        commit_message="feat: initial surge-fm-v2 release",
        ignore_patterns=["*.pyc", "__pycache__/*", ".DS_Store"],
    )
    print(f"done → https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
