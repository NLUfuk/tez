#!/usr/bin/env python3
"""Calibrate baseline coupling parameters to target firing-rate band."""

import argparse
import json
import statistics
import sys
from pathlib import Path

# Add src to path for imports
_project_root = Path(__file__).parent.parent
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from conway_izh.config import CouplingParams, MemoryParams, SimulationConfig, StrategyParams  # noqa: E402
from conway_izh.grid import NeuralGrid  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate conway_izh baseline parameters.")
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--target-min", type=float, default=0.08)
    parser.add_argument("--target-max", type=float, default=0.20)
    parser.add_argument("--out", type=str, default="outputs/ablation/calibration.json")
    return parser.parse_args()


def run_condition(height, width, steps, seed, k_neighbors, k_alive, bias):
    config = SimulationConfig(
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        coupling=CouplingParams(
            k_neighbors=k_neighbors,
            k_alive=k_alive,
            bias=bias,
            feedback_enabled=False,
            use_game_theory=False,
        ),
        memory=MemoryParams(enabled=False),
        strategy=StrategyParams(cellwise_enabled=False),
        output_dir="outputs",
        save_gif=False,
    )
    grid = NeuralGrid(config)
    firing = []
    efficiency = []
    for _ in range(steps):
        metrics, spikes = grid.step()
        firing.append(float(metrics["firing_rate"]))
        efficiency.append(float(metrics["efficiency_score"]))
    return {
        "avg_firing_rate": statistics.fmean(firing) if firing else 0.0,
        "avg_efficiency": statistics.fmean(efficiency) if efficiency else 0.0,
    }


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = []
    for k_neighbors in [0.4, 0.8, 1.2, 1.6]:
        for k_alive in [2.0, 3.0, 4.0, 5.0]:
            for bias in [2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0]:
                runs = []
                for i in range(args.seeds):
                    seed = args.base_seed + i * 31
                    run = run_condition(
                        args.height, args.width, args.steps, seed,
                        k_neighbors, k_alive, bias
                    )
                    runs.append(run)
                avg_firing = statistics.fmean(r["avg_firing_rate"] for r in runs)
                avg_eff = statistics.fmean(r["avg_efficiency"] for r in runs)
                in_band = args.target_min <= avg_firing <= args.target_max
                candidates.append(
                    {
                        "k_neighbors": k_neighbors,
                        "k_alive": k_alive,
                        "bias": bias,
                        "avg_firing_rate": avg_firing,
                        "avg_efficiency": avg_eff,
                        "in_target_band": in_band,
                    }
                )
                print(
                    f"kN={k_neighbors:.1f}, kA={k_alive:.1f}, b={bias:.1f} "
                    f"=> firing={avg_firing:.4f}, E={avg_eff:.4f}, in_band={in_band}"
                )

    in_band = [c for c in candidates if c["in_target_band"]]
    if in_band:
        best = max(in_band, key=lambda c: c["avg_efficiency"])
    else:
        target_center = 0.5 * (args.target_min + args.target_max)
        best = min(candidates, key=lambda c: abs(c["avg_firing_rate"] - target_center))

    payload = {
        "target_band": [args.target_min, args.target_max],
        "selected": best,
        "candidates": candidates,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("-" * 72)
    print(f"Selected baseline: {best}")
    print(f"Saved calibration: {out_path}")


if __name__ == "__main__":
    main()
