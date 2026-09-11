#!/usr/bin/env python3
"""Assemble the graded submission archive: ``23i-2515_Assignment01.zip``.

Layout required by the brief:
    src/            all implementation + experiment scripts
    output_images/  every generated figure
    report.pdf      the 4-page report

Extras included for completeness (harmless to graders, useful to reviewers):
    tests/, README.md, requirements.txt, results/*.json, SHA256SUMS.txt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

ROLL = "23i-2515"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED = ["src", "output_images", "report/report.pdf"]
EXTRAS = [
    "tests",
    "notebooks",
    "docs",
    "results",
    "report",  # LaTeX sources + both PDFs (report/build is excluded below)
    "README.md",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "LICENSE",
    "CHANGELOG.md",
]
EXCLUDE_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "build",
    ".egg-info",
    ".ipynb_checkpoints",
}


def _iter_files(rel: str):
    p = PROJECT_ROOT / rel
    if p.is_file():
        yield p
        return
    if not p.is_dir():
        return
    for f in p.rglob("*"):
        if f.is_file() and not any(
            part in EXCLUDE_PARTS or part.endswith(".egg-info") for part in f.parts
        ):
            yield f


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / f"{ROLL}_Assignment01.zip")
    ap.add_argument(
        "--allow-missing-report",
        action="store_true",
        help="package even if report.pdf has not been built yet",
    )
    args = ap.parse_args(argv)

    missing = [r for r in REQUIRED if not (PROJECT_ROOT / r).exists()]
    if missing:
        if missing == ["report/report.pdf"] and args.allow_missing_report:
            print("WARNING: report.pdf missing -- packaging without it (--allow-missing-report)")
        else:
            print(f"ERROR: required paths missing: {missing}", file=sys.stderr)
            print(
                "Build figures with `cvlab all` and the report with `cd report && make`.",
                file=sys.stderr,
            )
            return 1

    files: list[Path] = []
    for rel in [*REQUIRED, *EXTRAS]:
        files.extend(_iter_files(rel))
    files = sorted(set(files))

    sha_lines = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arc = Path(f"{ROLL}_Assignment01") / f.relative_to(PROJECT_ROOT)
            zf.write(f, arc)
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            sha_lines.append(f"{digest}  {f.relative_to(PROJECT_ROOT).as_posix()}")
        zf.writestr(f"{ROLL}_Assignment01/SHA256SUMS.txt", "\n".join(sha_lines) + "\n")

    size_mb = args.out.stat().st_size / 1e6
    print(f"  packed {len(files)} files -> {args.out.name}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
