#!/usr/bin/env python3
"""
k_syn × spike_trace_decay (gamma) parametre taraması: steady-state firing_rate ortalaması.

Tez çıktıları: uzun CSV (seed başına), toplulaştırılmış CSV (ısı haritası için), opsiyonel PNG.

metrics.firing_rate: adım başına spike oranı (= spike_count / (H*W)), literatürde sık kullanılan
normalize edilmiş anlık oran (bu scriptte transient sonrası adımlarda ortalanır).

Çalıştırma (proj kökünden veya PYTHONPATH=src ile):
  python scripts/sweep_k_syn_gamma.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# Tek yerden düzenlenebilir tarama parametreleri
# -----------------------------------------------------------------------------
NUM_SEEDS = 3  # Hızlı test: 1; tez overnight: 5 veya 10

BASE_SEED = 42
"""İlk fizik/grid tohumu; sonraki seed'ler deterministic aralıklıdır."""
SEED_STRIDE = 7919

K_SYN_VALUES = (0.0, 1.0, 2.0, 3.5, 5.0, 6.5, 8.0)
SPIKE_TRACE_DECAY_VALUES = (0.72, 0.80, 0.85, 0.88, 0.92, 0.96)

WARMUP_STEPS = 120
TOTAL_STEPS = 620

GRID_HEIGHT = 60
GRID_WIDTH = 60

# Stream / tez için A+C ile hizalı varsayı coupling (run_live ile tutarlı)
K_ALIVE = 4.0
K_NEIGHBORS = 0.5
BIAS = 0.5
FEEDBACK_ENABLED = True
FEEDBACK_GRAPH_NEIGHBORS = True

OUTPUT_SUBDIR = "sweep_k_syn_gamma"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from conway_izh.config import CouplingParams, SimulationConfig  # noqa: E402
from conway_izh.grid import NeuralGrid  # noqa: E402


def _steady_mean_firing_rate(
    *,
    k_syn: float,
    spike_trace_decay: float,
    seed: int,
    scratch_out: Path,
    warmup_steps: int,
    total_steps: int,
    grid_h: int,
    grid_w: int,
) -> tuple[float, int]:
    """
    Returns ``(mean firing_rate over [warmup, total_steps), len)``.
    """
    coupling = CouplingParams(
        k_neighbors=K_NEIGHBORS,
        k_alive=K_ALIVE,
        bias=BIAS,
        feedback_enabled=FEEDBACK_ENABLED,
        feedback_graph_neighbors=FEEDBACK_GRAPH_NEIGHBORS,
        k_syn=float(k_syn),
        spike_trace_decay=float(np.clip(spike_trace_decay, 1e-6, 0.9999)),
    )

    cfg = SimulationConfig(
        height=grid_h,
        width=grid_w,
        steps=total_steps,
        seed=int(seed),
        coupling=coupling,
        output_dir=str(scratch_out.parent),
        run_id=scratch_out.name,
        save_gif=False,
    )

    grid = NeuralGrid(cfg)
    rates: list[float] = []

    for _ in range(warmup_steps):
        _m, _s = grid.step()

    for _ in range(warmup_steps, total_steps):
        m, _s = grid.step()
        rates.append(float(m["firing_rate"]))

    n_avg = len(rates)
    if n_avg <= 0:
        return float("nan"), 0
    return float(np.mean(rates)), n_avg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="k_syn × gamma sweep; aggregates firing_rate over seeds.",
    )
    parser.add_argument(
        "--scratch-name",
        type=str,
        default="_sweep_neural_grid_scratch",
        help="Tek NeuralGrid çıktı alt dizini adı (üzerine yazılır).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Hızlı duman testi (küçük ızgara, kısa adım; CSV/PNG yine yazılır).",
    )
    args = parser.parse_args()

    num_seeds = NUM_SEEDS
    k_syn_values = tuple(K_SYN_VALUES)
    decay_values = tuple(SPIKE_TRACE_DECAY_VALUES)
    warmup_steps = WARMUP_STEPS
    total_steps = TOTAL_STEPS
    grid_h = GRID_HEIGHT
    grid_w = GRID_WIDTH

    if args.quick:
        num_seeds = min(NUM_SEEDS, 2)
        k_syn_values = (0.0, 4.0)
        decay_values = (0.85, 0.92)
        warmup_steps = 8
        total_steps = 32
        grid_h = 24
        grid_w = 24

    out_root = (
        (_PROJECT_ROOT / "outputs" / f"{OUTPUT_SUBDIR}_quick")
        if args.quick
        else (_PROJECT_ROOT / "outputs" / OUTPUT_SUBDIR)
    )
    out_root.mkdir(parents=True, exist_ok=True)
    scratch_root = _PROJECT_ROOT / "outputs" / args.scratch_name
    scratch_root.mkdir(parents=True, exist_ok=True)

    seeds = [BASE_SEED + j * SEED_STRIDE for j in range(num_seeds)]

    detailed_csv = out_root / "sweep_detail.csv"
    agg_csv = out_root / "sweep_aggregate.csv"

    detail_rows: list[dict[str, object]] = []
    agg_buckets: dict[tuple[float, float], list[float]] = {}

    for k_syn in k_syn_values:
        for gamma in decay_values:
            key = (float(k_syn), float(gamma))
            agg_buckets[key] = []

    print(
        f"Sweep | seeds={num_seeds} base={BASE_SEED} stride={SEED_STRIDE} | "
        f"warmup={warmup_steps} total={total_steps} | grid={grid_h}x{grid_w}",
        flush=True,
    )

    total_jobs = len(k_syn_values) * len(decay_values) * num_seeds
    done = 0

    for k_syn in k_syn_values:
        for gamma in decay_values:
            for seed in seeds:
                mu_fr, n_steps = _steady_mean_firing_rate(
                    k_syn=k_syn,
                    spike_trace_decay=gamma,
                    seed=seed,
                    scratch_out=scratch_root,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                    grid_h=grid_h,
                    grid_w=grid_w,
                )
                row = {
                    "k_syn": float(k_syn),
                    "spike_trace_decay_gamma": float(gamma),
                    "seed": int(seed),
                    "steady_mean_firing_rate": mu_fr,
                    "n_steps_averaged_after_warmup": int(n_steps),
                    "warmup_steps": int(warmup_steps),
                }
                detail_rows.append(row)
                agg_buckets[(float(k_syn), float(gamma))].append(mu_fr)

                done += 1
                if done % max(1, total_jobs // 10) == 0 or done == total_jobs:
                    print(f"  progress {done}/{total_jobs}", flush=True)

    # detailed CSV
    if not detail_rows:
        raise RuntimeError("No sweep rows generated (check parameter tuples).")
    with detailed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    # aggregate CSV (+ stderr across seeds / sqrt(n))
    agg_rows: list[dict[str, object]] = []
    for k_syn in k_syn_values:
        for gamma in decay_values:
            vals = np.array(agg_buckets[(float(k_syn), float(gamma))], dtype=np.float64)
            mean_v = float(np.mean(vals)) if vals.size else float("nan")
            if vals.size > 1:
                std_seeds = float(np.std(vals, ddof=1))
                stderr = std_seeds / np.sqrt(vals.size)
            else:
                std_seeds = 0.0
                stderr = 0.0
            agg_rows.append(
                {
                    "k_syn": float(k_syn),
                    "spike_trace_decay_gamma": float(gamma),
                    "firing_rate_mean_across_seeds": mean_v,
                    "firing_rate_std_across_seeds": std_seeds,
                    "firing_rate_stderr": float(stderr),
                    "num_seeds": int(vals.size),
                }
            )

    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)

    # Heatmap PNG (matplotlib)
    try:
        import matplotlib.pyplot as plt

        k_list = sorted({float(k) for k in k_syn_values})
        g_list = sorted({float(g) for g in decay_values})
        mat = np.full((len(k_list), len(g_list)), np.nan, dtype=np.float64)
        for i, kk in enumerate(k_list):
            for j, gg in enumerate(g_list):
                v = agg_buckets[(kk, gg)]
                mat[i, j] = float(np.mean(v)) if len(v) else np.nan

        # mat[i,j] ↔ k_syn = k_list[i], γ = g_list[j]; imshow: x=j (γ), y=i (k_syn)
        dg = (g_list[-1] - g_list[0]) / max(2 * (len(g_list) - 1), 2) if len(g_list) > 1 else 0.02
        dk = (k_list[-1] - k_list[0]) / max(2 * (len(k_list) - 1), 2) if len(k_list) > 1 else 0.25
        fig, ax = plt.subplots(figsize=(8.5, 6.0))
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            extent=[
                min(g_list) - dg,
                max(g_list) + dg,
                min(k_list) - dk,
                max(k_list) + dk,
            ],
            interpolation="nearest",
        )
        ax.set_xlabel(r"$\gamma$ (spike_trace_decay)")
        ax.set_ylabel(r"$k_{syn}$ graph coupling")
        ax.set_title(
            rf"Mean steady firing_rate (metrics) | NUM_SEEDS={num_seeds} | "
            f"{grid_h}×{grid_w} | warmup {warmup_steps}/{total_steps}"
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"Mean firing_rate ($\mathrm{spikes / cell / step}$)")

        heatmap_path = out_root / "sweep_firing_rate_heatmap.png"
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=160)
        plt.close(fig)
        print(f"Wrote heatmap: {heatmap_path}", flush=True)
    except ImportError:
        heatmap_path = None
        print("matplotlib yok — PNG üretimi atlandı.", flush=True)

    print(f"Detailed: {detailed_csv}", flush=True)
    print(f"Aggregate (heatmap input): {agg_csv}", flush=True)


if __name__ == "__main__":
    main()
