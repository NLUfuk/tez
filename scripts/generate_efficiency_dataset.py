#!/usr/bin/env python3
"""Generate leakage-safe dataset for efficiency score prediction."""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add src to path for imports
_project_root = Path(__file__).parent.parent
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from conway_izh.config import (  # noqa: E402
    CouplingParams,
    MemoryParams,
    SimulationConfig,
    StrategyParams,
)
from conway_izh.conway import count_neighbors  # noqa: E402
from conway_izh.dataset_schema import FEATURE_CHANNELS, validate_arrays  # noqa: E402
from conway_izh.grid import NeuralGrid  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dataset for CNN efficiency score prediction."
    )
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--window", type=int, default=10,
                        help="Future window for E_window target")
    parser.add_argument("--rollouts", type=int, default=18,
                        help="Number of rollout simulations")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="outputs/datasets/efficiency")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--feedback", action="store_true")
    return parser.parse_args()


def _build_randomized_config(
    height: int,
    width: int,
    steps: int,
    seed: int,
    feedback: bool
) -> Tuple[SimulationConfig, Dict[str, float]]:
    # Controlled randomness for scenario diversity.
    propagation_strength = random.choice([0.3, 0.5, 0.7, 0.9])
    cooperation_factor = random.choice([0.2, 0.3, 0.4, 0.5])
    cooperation_strength = random.choice([0.5, 0.7, 0.9])
    k_alive = random.choice([1.5, 2.0, 2.5, 3.0])
    memory_decay = random.choice([0.88, 0.92, 0.95])
    strategy_temp = random.choice([4.0, 6.0, 8.0])

    config = SimulationConfig(
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        coupling=CouplingParams(
            feedback_enabled=feedback,
            use_game_theory=True,
            propagation_strength=propagation_strength,
            cooperation_factor=cooperation_factor,
            cooperation_strength=cooperation_strength,
            k_alive=k_alive,
            k_neighbors=0.5,
            bias=0.0,
        ),
        memory=MemoryParams(
            enabled=True,
            decay=memory_decay,
            spike_gain=0.35,
            neighbor_gain=0.08,
        ),
        strategy=StrategyParams(
            cellwise_enabled=True,
            temperature=strategy_temp,
            switch_cost=0.03,
        ),
        output_dir="outputs",
        save_gif=False,
    )
    scenario = {
        "propagation_strength": propagation_strength,
        "cooperation_factor": cooperation_factor,
        "cooperation_strength": cooperation_strength,
        "k_alive": k_alive,
        "memory_decay": memory_decay,
        "strategy_temp": strategy_temp,
        "feedback_enabled": feedback,
    }
    return config, scenario


def _collect_rollout(grid: NeuralGrid, steps: int) -> Dict[str, np.ndarray]:
    features = []
    efficiencies = []

    for _ in range(steps):
        metrics, spikes = grid.step()
        neighbors = count_neighbors(grid.gol_state, grid.config.wrap_around)
        x_t = np.stack(
            [
                grid.gol_state.astype(np.float32),
                grid.v.astype(np.float32),
                spikes.astype(np.float32),
                neighbors.astype(np.float32),
                grid.strategy_map.astype(np.float32),
                grid.memory_state.astype(np.float32),
            ],
            axis=0,
        )
        features.append(x_t)
        efficiencies.append(float(metrics["efficiency_score"]))

    return {
        "X": np.asarray(features, dtype=np.float32),  # [T, C, H, W]
        "E": np.asarray(efficiencies, dtype=np.float32),  # [T]
    }


def _to_supervised(rollout: Dict[str, np.ndarray], window: int) -> Tuple[np.ndarray, np.ndarray]:
    X_seq = rollout["X"]
    E_seq = rollout["E"]
    if len(E_seq) <= window:
        return np.empty((0, *X_seq.shape[1:]), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X_list = []
    y_list = []
    max_t = len(E_seq) - window
    for t in range(max_t):
        X_list.append(X_seq[t])
        y_list.append(float(np.mean(E_seq[t + 1:t + 1 + window])))
    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def _split_rollouts(rollout_ids: List[int], train_ratio: float, val_ratio: float):
    ids = rollout_ids[:]
    random.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    if not test_ids:
        test_ids = [val_ids.pop()]
    return train_ids, val_ids, test_ids


def _concat_split(rollout_store: Dict[int, Dict[str, np.ndarray]], ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    X_parts = [rollout_store[i]["X"] for i in ids]
    y_parts = [rollout_store[i]["y"] for i in ids]
    X = np.concatenate(X_parts, axis=0) if X_parts else np.empty((0, len(FEATURE_CHANNELS), 1, 1), dtype=np.float32)
    y = np.concatenate(y_parts, axis=0) if y_parts else np.empty((0,), dtype=np.float32)
    validate_arrays(X, y)
    return X, y


def main():
    args = parse_args()
    random.seed(args.base_seed)
    np.random.seed(args.base_seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_store = {}
    metadata = {
        "feature_channels": FEATURE_CHANNELS,
        "window": args.window,
        "rollouts": [],
    }

    for rollout_id in range(args.rollouts):
        seed = args.base_seed + rollout_id * 17
        config, scenario = _build_randomized_config(
            args.height, args.width, args.steps, seed, args.feedback
        )
        grid = NeuralGrid(config)
        raw = _collect_rollout(grid, args.steps)
        X, y = _to_supervised(raw, args.window)
        rollout_store[rollout_id] = {"X": X, "y": y}
        metadata["rollouts"].append(
            {
                "rollout_id": rollout_id,
                "seed": seed,
                "samples": int(len(y)),
                "scenario": scenario,
            }
        )
        print(
            f"[{rollout_id + 1}/{args.rollouts}] seed={seed}, "
            f"samples={len(y)}, mean_target={float(np.mean(y)) if len(y) > 0 else 0.0:.4f}"
        )

    rollout_ids = list(rollout_store.keys())
    train_ids, val_ids, test_ids = _split_rollouts(
        rollout_ids, args.train_ratio, args.val_ratio
    )

    X_train, y_train = _concat_split(rollout_store, train_ids)
    X_val, y_val = _concat_split(rollout_store, val_ids)
    X_test, y_test = _concat_split(rollout_store, test_ids)

    np.savez_compressed(out_dir / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(out_dir / "val.npz", X=X_val, y=y_val)
    np.savez_compressed(out_dir / "test.npz", X=X_test, y=y_test)

    metadata["split"] = {
        "train_rollouts": train_ids,
        "val_rollouts": val_ids,
        "test_rollouts": test_ids,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
    }
    metadata["shape"] = {
        "channels": int(X_train.shape[1]) if len(X_train) else len(FEATURE_CHANNELS),
        "height": args.height,
        "width": args.width,
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("-" * 70)
    print(f"Dataset saved to: {out_dir}")
    print(f"Train: {X_train.shape}, targets={y_train.shape}")
    print(f"Val:   {X_val.shape}, targets={y_val.shape}")
    print(f"Test:  {X_test.shape}, targets={y_test.shape}")


if __name__ == "__main__":
    main()
