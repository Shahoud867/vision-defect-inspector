#!/usr/bin/env python3
"""Fetch the two Kaggle benchmark datasets into ``data/raw/``.

The pipeline runs fine *without* this (synthetic fallbacks with ground truth),
but this reproduces the assignment's exact evaluation imagery.

Prerequisites
-------------
1. ``pip install -e ".[data]"``  (installs ``kagglehub``)
2. A Kaggle API token at ``~/.kaggle/kaggle.json``
   (Kaggle -> Account -> Create New API Token). On Windows:
   ``%USERPROFILE%\\.kaggle\\kaggle.json``.

Usage
-----
    python scripts/download_data.py                # both datasets, 12 images each
    python scripts/download_data.py --per-class 30 # more samples
    python scripts/download_data.py --only pcb
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"

DATASETS = {
    "pcb": "akhatova/pcb-defects",
    "concrete": "arunrk7/surface-crack-detection",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _copy_sample(src_root: Path, dst: Path, limit: int) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    picked = 0
    for path in sorted(src_root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        shutil.copy2(path, dst / f"{picked:03d}_{path.name}")
        picked += 1
        if picked >= limit:
            break
    return picked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--per-class", type=int, default=12, help="images to keep per dataset")
    ap.add_argument("--only", choices=list(DATASETS), help="fetch a single dataset")
    args = ap.parse_args(argv)

    try:
        import kagglehub
    except ModuleNotFoundError:
        print('ERROR: kagglehub not installed. Run:  pip install -e ".[data]"', file=sys.stderr)
        return 2

    targets = {args.only: DATASETS[args.only]} if args.only else DATASETS
    for name, slug in targets.items():
        print(f"[{name}] downloading {slug} ...")
        try:
            cache_dir = Path(kagglehub.dataset_download(slug))
        except Exception as exc:
            print(
                f"  FAILED: {exc}\n  (check ~/.kaggle/kaggle.json). Synthetic fallback still works.",
                file=sys.stderr,
            )
            continue
        n = _copy_sample(cache_dir, RAW / name, args.per_class)
        print(f"  -> {n} images copied to {RAW / name}")

    print("\nDone. Re-run experiments on the real data with:")
    print("  cvlab all --data-dir data/raw --prefer real")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
