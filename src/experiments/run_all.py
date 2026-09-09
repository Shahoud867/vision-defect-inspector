"""Run every task driver end-to-end and emit a manifest of generated figures."""

from __future__ import annotations

import json
from pathlib import Path

from . import run_task1_convolution as t1
from . import run_task2_enhancement as t2
from . import run_task3_canny as t3
from . import run_task4_ablations as t4
from . import run_task5_analysis as t5
from ._common import ExperimentConfig, Timer, banner

TASKS = {
    "task1": t1.run,
    "task2": t2.run,
    "task3": t3.run,
    "task4": t4.run,
    "task5": t5.run,
}


def run(cfg: ExperimentConfig | None = None, only: list[str] | None = None) -> dict:
    cfg = cfg or ExperimentConfig()
    selected = only or list(TASKS)
    manifest: dict[str, object] = {
        "config": {
            "seed": cfg.seed,
            "dpi": cfg.dpi,
            "image_size": cfg.image_size,
            "prefer": cfg.prefer,
            "out_dir": str(cfg.out_dir),
        },
        "tasks": {},
    }

    for name in selected:
        with Timer() as tm:
            artefacts = TASKS[name](cfg)
        manifest["tasks"][name] = {
            "seconds": round(tm.elapsed, 2),
            "artefacts": [
                str(Path(a).relative_to(cfg.out_dir.parent))
                if cfg.out_dir.parent in Path(a).parents
                else str(a)
                for a in artefacts
            ],
        }
        print(f"  -> {name} done in {tm.elapsed:.1f}s ({len(artefacts)} files)")

    manifest_path = cfg.results_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    banner("ALL TASKS COMPLETE")
    total = sum(len(v["artefacts"]) for v in manifest["tasks"].values())
    print(f"  {total} artefacts under {cfg.out_dir}")
    print(f"  manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":  # pragma: no cover
    run()
