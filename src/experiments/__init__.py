"""Reproducible experiment drivers for Assignment 1 (Tasks 1-5).

Each ``run_taskN`` module exposes ``run(cfg: ExperimentConfig) -> list[Path]`` and
writes high-resolution figures into ``cfg.out_dir``.  ``run_all`` chains them.
"""

from __future__ import annotations

from ._common import ExperimentConfig

__all__ = ["ExperimentConfig"]
