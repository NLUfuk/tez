#!/usr/bin/env python3
"""Create Turkish thesis figures from ablation and training outputs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate thesis-ready figures.")
    parser.add_argument("--ablation-dir", type=str, default="outputs/ablation")
    parser.add_argument("--model-dir", type=str, default="outputs/models/efficiency_cnn")
    parser.add_argument("--out", type=str, default="outputs/thesis_figures")
    return parser.parse_args()


def _load_csv(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_ablation_efficiency(summary_rows, out_path: Path):
    labels = [r["condition"] for r in summary_rows]
    means = np.array([float(r["avg_efficiency_mean"]) for r in summary_rows], dtype=float)
    stds = np.array([float(r["avg_efficiency_std"]) for r in summary_rows], dtype=float)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=6, alpha=0.85, color="#2E86AB")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Ortalama Verim Skoru (E)")
    plt.title("Ablation Sonuclari: Verim Skoru Karsilastirmasi")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_component_breakdown(summary_rows, out_path: Path):
    labels = [r["condition"] for r in summary_rows]
    comps = [
        ("avg_stability_mean", "Stabilite"),
        ("avg_information_mean", "Bilgi Tasima"),
        ("avg_memory_mean", "Hafiza"),
        ("avg_cost_mean", "Maliyet"),
    ]
    x = np.arange(len(labels))
    width = 0.18

    plt.figure(figsize=(12, 6))
    for i, (key, name) in enumerate(comps):
        vals = [float(r[key]) for r in summary_rows]
        plt.bar(x + (i - 1.5) * width, vals, width=width, label=name, alpha=0.9)

    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Skor")
    plt.title("Ablation Bilesen Skorlari")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_training_curves(history, out_path: Path):
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_mae = [h["val_mae"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", linewidth=2)
    axes[0].set_title("CNN Egitim Kaybi")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_mae, color="#D1495B", linewidth=2)
    axes[1].set_title("CNN Dogrulama MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_model_vs_naive(metrics, out_path: Path):
    labels = ["Model MAE", "Naive MAE", "Model RMSE", "Naive RMSE"]
    vals = [
        float(metrics.get("test_mae", 0.0)),
        float(metrics.get("naive_mae", 0.0)),
        float(metrics.get("test_rmse", 0.0)),
        float(metrics.get("naive_rmse", 0.0)),
    ]
    colors = ["#00798C", "#EDA247", "#00798C", "#EDA247"]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals, color=colors, alpha=0.9)
    plt.title("CNN ve Naive Baseline Karsilastirmasi")
    plt.ylabel("Hata")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    ablation_dir = Path(args.ablation_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = ablation_dir / "ablation_summary.csv"
    if summary_csv.exists():
        summary_rows = _load_csv(summary_csv)
        plot_ablation_efficiency(summary_rows, out_dir / "ablation_efficiency.png")
        plot_component_breakdown(summary_rows, out_dir / "ablation_components.png")
        print(f"Ablation figures created in: {out_dir}")
    else:
        print(f"Skipping ablation plots (missing {summary_csv})")

    history_path = model_dir / "train_history.json"
    metrics_path = model_dir / "metrics.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        plot_training_curves(history, out_dir / "cnn_training_curves.png")
    else:
        print(f"Skipping training curves (missing {history_path})")

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        plot_model_vs_naive(metrics, out_dir / "cnn_vs_naive.png")
    else:
        print(f"Skipping model-vs-naive plot (missing {metrics_path})")

    print(f"Done. Thesis figures folder: {out_dir}")


if __name__ == "__main__":
    main()
