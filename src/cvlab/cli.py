"""``cvlab`` command-line entry point.

Examples
--------
    cvlab all                      # run every task, write output_images/
    cvlab task3 --prefer synthetic # single task, force synthetic data
    cvlab all --data-dir data/raw --dpi 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cvlab import __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cvlab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--version", action="version", version=f"cvlab {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", type=Path, default=None, help="figure output directory")
    common.add_argument("--results-dir", type=Path, default=None, help="JSON metrics directory")
    common.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="folder with real dataset images (data/raw by default)",
    )
    common.add_argument("--seed", type=int, default=2515)
    common.add_argument("--dpi", type=int, default=150)
    common.add_argument("--image-size", type=int, default=256)
    common.add_argument("--prefer", choices=["auto", "real", "synthetic"], default="auto")

    for name in ("task1", "task2", "task3", "task4", "task5", "all"):
        sub.add_parser(name, parents=[common], help=f"run {name}")
    return p


def _config_from_args(args: argparse.Namespace):
    from experiments._common import DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR, ExperimentConfig

    return ExperimentConfig(
        out_dir=args.out_dir or DEFAULT_OUT_DIR,
        results_dir=args.results_dir or DEFAULT_RESULTS_DIR,
        data_dir=args.data_dir,
        seed=args.seed,
        dpi=args.dpi,
        image_size=args.image_size,
        prefer=args.prefer,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = _config_from_args(args)

    if args.command == "all":
        from experiments.run_all import run as run_all

        run_all(cfg)
        return 0

    runners = {
        "task1": "experiments.run_task1_convolution",
        "task2": "experiments.run_task2_enhancement",
        "task3": "experiments.run_task3_canny",
        "task4": "experiments.run_task4_ablations",
        "task5": "experiments.run_task5_analysis",
    }
    import importlib

    importlib.import_module(runners[args.command]).run(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
