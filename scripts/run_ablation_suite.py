#!/usr/bin/env python3
"""Run ablation experiments for Conway-Izhikevich thesis setup."""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
_project_root = Path(__file__).parent.parent
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from conway_izh.config import CouplingParams, MemoryParams, SimulationConfig, StrategyParams  # noqa: E402
from conway_izh.grid import NeuralGrid  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation suite across seeds.")
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="outputs/ablation")
    parser.add_argument("--calibration-json", type=str, default=None,
                        help="Optional calibration json from calibrate_baselines.py")
    return parser.parse_args()


def build_config(
    condition: str,
    height: int,
    width: int,
    steps: int,
    seed: int,
    calibrated_baseline: Dict[str, float] | None = None
) -> SimulationConfig:
    # Shared defaults for fair comparison.
    coupling = CouplingParams(
        k_neighbors=0.5,
        k_alive=2.0,
        bias=0.0,
        feedback_enabled=False,
        use_game_theory=False,
        propagation_strength=0.5,
        cooperation_factor=0.3,
        cooperation_strength=0.7,
    )
    memory = MemoryParams(enabled=False)
    strategy = StrategyParams(cellwise_enabled=False)

    if condition == "izh_only":
        coupling.k_neighbors = 0.0
        coupling.k_alive = 0.0
        coupling.bias = 9.0
    elif condition == "conway_izh":
        if calibrated_baseline is not None:
            coupling.k_neighbors = float(calibrated_baseline["k_neighbors"])
            coupling.k_alive = float(calibrated_baseline["k_alive"])
            coupling.bias = float(calibrated_baseline["bias"])
        else:
            coupling.k_neighbors = 0.8
            coupling.k_alive = 3.0
            coupling.bias = 2.0
    elif condition == "game_theory_no_memory":
        coupling.use_game_theory = True
        coupling.bias = 1.5
        strategy.cellwise_enabled = True
    elif condition == "full_model":
        coupling.use_game_theory = True
        coupling.bias = 1.5
        strategy.cellwise_enabled = True
        memory.enabled = True
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return SimulationConfig(
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        coupling=coupling,
        memory=memory,
        strategy=strategy,
        output_dir="outputs",
        save_gif=False,
    )


def summarize_run(metrics_history: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_history:
        return {
            "avg_efficiency": 0.0,
            "avg_firing_rate": 0.0,
            "avg_stability": 0.0,
            "avg_information": 0.0,
            "avg_memory": 0.0,
            "avg_cost": 0.0,
            "final_alive_ratio": 0.0,
            "avg_cooperative_ratio": 0.0,
        }
    return {
        "avg_efficiency": statistics.fmean(m["efficiency_score"] for m in metrics_history),
        "avg_firing_rate": statistics.fmean(m["firing_rate"] for m in metrics_history),
        "avg_stability": statistics.fmean(m["stability_score"] for m in metrics_history),
        "avg_information": statistics.fmean(m["information_score"] for m in metrics_history),
        "avg_memory": statistics.fmean(m["memory_score"] for m in metrics_history),
        "avg_cost": statistics.fmean(m["cost_score"] for m in metrics_history),
        "final_alive_ratio": float(metrics_history[-1]["alive_ratio"]),
        "avg_cooperative_ratio": statistics.fmean(m.get("cooperative_ratio", 0.0) for m in metrics_history),
    }


def aggregate_condition(rows: List[Dict[str, float]], condition: str) -> Dict[str, float]:
    cond_rows = [r for r in rows if r["condition"] == condition]
    metric_keys = [
        "avg_efficiency",
        "avg_firing_rate",
        "avg_stability",
        "avg_information",
        "avg_memory",
        "avg_cost",
        "final_alive_ratio",
        "avg_cooperative_ratio",
    ]
    out = {"condition": condition, "runs": len(cond_rows)}
    for key in metric_keys:
        vals = [float(r[key]) for r in cond_rows]
        out[f"{key}_mean"] = statistics.fmean(vals) if vals else 0.0
        out[f"{key}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    calibrated_baseline = None
    if args.calibration_json:
        payload = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
        calibrated_baseline = payload.get("selected")

    conditions = [
        "izh_only",
        "conway_izh",
        "game_theory_no_memory",
        "full_model",
    ]

    run_rows: List[Dict[str, float]] = []
    for condition in conditions:
        for i in range(args.seeds):
            seed = args.base_seed + 31 * i
            config = build_config(
                condition,
                args.height,
                args.width,
                args.steps,
                seed,
                calibrated_baseline=calibrated_baseline,
            )
            grid = NeuralGrid(config)
            for _ in range(args.steps):
                metrics, spikes = grid.step()
                grid.metrics_history.append(metrics)
                grid.spike_history.append(spikes.copy())
            summary = summarize_run(grid.metrics_history)
            row = {"condition": condition, "seed": seed, **summary}
            run_rows.append(row)
            print(
                f"{condition:>20} | seed={seed} | "
                f"E={summary['avg_efficiency']:.4f} | firing={summary['avg_firing_rate']:.4f}"
            )

    run_csv = out_dir / "ablation_runs.csv"
    with open(run_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(run_rows)

    summary_rows = [aggregate_condition(run_rows, c) for c in conditions]
    summary_csv = out_dir / "ablation_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(out_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print("-" * 72)
    print(f"Saved run-level results: {run_csv}")
    print(f"Saved summary results:   {summary_csv}")


if __name__ == "__main__":
    main()
