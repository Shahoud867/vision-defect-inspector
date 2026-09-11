#!/usr/bin/env python3
"""Print every quantitative value cited in report/ straight from results/*.json.

Run after ``cvlab all`` to cross-check the hard-coded numbers in the report
prose and tables.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _load(name: str) -> dict:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"missing {p} -- run `cvlab all` first")
    return json.loads(p.read_text())


def main() -> None:
    t1 = _load("task1_shape_verification")
    t2 = _load("task2_metrics")
    t3 = _load("task3_metrics")
    t4 = _load("task4_metrics")
    t5 = _load("task5_metrics")

    print("== Task 1: conv2d shape formula ==")
    print(f"  all {len(t1['cases'])} geometries match Eq. (1): {t1['all_pass']}")

    print("\n== Task 2 ==")
    pg = t2["pooling"]
    print(
        f"  max-pool([[8,7],[12,9]]) = {pg['max_pool_output']}  ->  dOut/dIn = {pg['analytic_gradient']}"
    )
    print(f"  finite-difference check passes: {pg['gradients_match']}")
    for k, v in t2["unsharp"].items():
        print(
            f"  unsharp {k}: high-freq gain x{v['high_freq_gain']:.3f} "
            f"(std {v['input_std']:.2f} -> {v['sharpened_std']:.2f})"
        )

    print("\n== Task 3: full Canny ==")
    for k, v in t3.items():
        print(
            f"  {k:9s}: F1={v['f1']:.3f} P={v['precision']:.3f} R={v['recall']:.3f} "
            f"density={v['density']:.4f} comps={v['n_components']} "
            f"(t_low={v['t_low']:.2f}, t_high={v['t_high']:.2f})"
        )

    print("\n== Task 4 ==")
    for k, v in t4["ablation1"].items():
        print(
            f"  A1 {k}: std raw={v['grad_std_raw']:.2f} smoothed={v['grad_std_smoothed']:.2f} "
            f"suppression x{v['noise_suppression_ratio']:.2f}"
        )
    a2 = t4["ablation2"]
    print(
        f"  A2: density {a2['edge_density_before_nms']:.3f} -> {a2['edge_density_after_nms']:.3f} "
        f"(thinning x{a2['thinning_ratio']:.2f})"
    )
    for k in ("single_low", "single_high", "hysteresis"):
        v = t4["ablation3"][k]
        print(
            f"  A3 {k:11s}: F1={v['f1']:.3f} P={v['precision']:.3f} R={v['recall']:.3f} "
            f"comps={v['n_components']}"
        )

    print("\n== Task 5 ==")
    ss = t5["scale_space"]
    counts = [ss[k]["edge_pixels"] for k in ("sigma=1.0", "sigma=2.5", "sigma=5.0")]
    print(
        f"  scale-space edge counts sigma=1.0/2.5/5.0: {counts}  monotone={ss['monotone_non_increasing']}"
    )
    for ds, grid in t5["sensitivity"].items():
        best = max(grid.items(), key=lambda kv: kv[1]["f1"] or 0.0)
        print(f"  sensitivity {ds}: best {best[0]} -> F1={best[1]['f1']:.3f}")
    for k, v in t5["failure_cases"].items():
        print(f"  failure '{k}': F1={v['f1']:.3f} P={v['precision']:.3f} R={v['recall']:.3f}")


if __name__ == "__main__":
    main()
