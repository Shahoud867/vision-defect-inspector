"""Shared configuration and helpers for the experiment drivers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cvlab.datasets import PROJECT_ROOT

DEFAULT_OUT_DIR = PROJECT_ROOT / "output_images"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class ExperimentConfig:
    """Runtime knobs shared by every task driver."""

    out_dir: Path = DEFAULT_OUT_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR
    data_dir: Path | None = None
    seed: int = 2515  # last four digits of roll number 23i-2515
    dpi: int = 150
    image_size: int = 256
    prefer: str = "auto"  # 'auto' | 'real' | 'synthetic'
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.results_dir = Path(self.results_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if self.data_dir is not None:
            self.data_dir = Path(self.data_dir)

    def rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)


def dump_json(cfg: ExperimentConfig, name: str, payload: dict) -> Path:
    """Persist a metrics/summary dict to ``results/<name>.json``."""
    path = cfg.results_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return path


def _json_default(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)!r}")


class Timer:
    """Context manager returning elapsed wall-clock seconds."""

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._t0


def banner(title: str) -> None:
    line = "=" * max(12, len(title) + 4)
    print(f"\n{line}\n  {title}\n{line}")
