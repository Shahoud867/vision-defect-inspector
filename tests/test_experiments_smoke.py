"""Fast end-to-end smoke test for the experiment drivers.

Runs each task once at a tiny resolution on synthetic data and asserts that the
promised figures and metrics land on disk.  Marked ``slow`` so the default unit
run stays sub-second; CI runs the full ``cvlab all`` separately.
"""

from __future__ import annotations

import pytest

from experiments._common import ExperimentConfig

pytestmark = pytest.mark.slow

TASKS = {
    "task1": ("experiments.run_task1_convolution", 3),
    "task2": ("experiments.run_task2_enhancement", 4),
    "task3": ("experiments.run_task3_canny", 3),
    "task4": ("experiments.run_task4_ablations", 3),
    "task5": ("experiments.run_task5_analysis", 5),
}


@pytest.mark.parametrize("name", list(TASKS))
def test_task_driver_writes_artifacts(tmp_path, name):
    import importlib

    module_name, min_files = TASKS[name]
    cfg = ExperimentConfig(
        out_dir=tmp_path / "figs",
        results_dir=tmp_path / "res",
        seed=1,
        dpi=60,
        image_size=96,
        prefer="synthetic",
    )
    artefacts = importlib.import_module(module_name).run(cfg)
    assert len(artefacts) >= min_files
    pngs = list((tmp_path / "figs").glob("*.png"))
    assert pngs, "no figures written"
    for p in pngs:
        assert p.stat().st_size > 1000
