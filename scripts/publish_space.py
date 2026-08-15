#!/usr/bin/env python3
"""Assemble and publish the Hugging Face Space demo.

HF Spaces are independent repos with a flat root layout, which doesn't match
this repo's structure (space/ + src/ + pyproject.toml at different levels) —
so this copies everything into a temp dir in the deployed layout before
uploading, rather than pushing space/ as-is.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]


def assemble(build_dir: Path) -> None:
    shutil.copy(REPO_ROOT / "pyproject.toml", build_dir / "pyproject.toml")
    shutil.copytree(REPO_ROOT / "src", build_dir / "src")
    shutil.copy(REPO_ROOT / "space" / "app.py", build_dir / "app.py")
    shutil.copy(REPO_ROOT / "space" / "Dockerfile", build_dir / "Dockerfile")
    shutil.copy(REPO_ROOT / "space" / "README.md", build_dir / "README.md")
    shutil.copytree(REPO_ROOT / "space" / "samples", build_dir / "samples")

    # Space's own __pycache__ from a local dev run shouldn't be uploaded.
    for cache_dir in build_dir.rglob("__pycache__"):
        shutil.rmtree(cache_dir)


def publish(repo_id: str) -> None:
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        build_dir = Path(tmp) / "space_build"
        build_dir.mkdir()
        assemble(build_dir)
        api.upload_folder(folder_path=str(build_dir), repo_id=repo_id, repo_type="space")

    print(f"Published to https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/adas-perception-demo")
    args = parser.parse_args()
    publish(args.repo_id)
