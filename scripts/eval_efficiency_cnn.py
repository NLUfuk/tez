#!/usr/bin/env python3
"""Evaluate trained CNN on efficiency dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_efficiency_cnn import EfficiencyCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate efficiency CNN model")
    parser.add_argument("--data-dir", type=str, default="outputs/datasets/efficiency")
    parser.add_argument("--model-dir", type=str, default="outputs/models/efficiency_cnn")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_dir) / f"{args.split}.npz"
    model_dir = Path(args.model_dir)
    model_path = model_dir / "best_model.pt"

    data = np.load(data_path)
    X = torch.from_numpy(data["X"]).float()
    y = torch.from_numpy(data["y"]).float()
    norm = torch.load(model_dir / "normalization.pt", map_location="cpu")
    X = (X - norm["x_mean"]) / norm["x_std"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficiencyCNN(in_channels=X.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred_n = model(X.to(device)).cpu()
    pred = pred_n * norm["y_std"] + norm["y_mean"]

    mae = float(torch.mean(torch.abs(pred - y)).item()) if len(y) else 0.0
    rmse = float(torch.sqrt(torch.mean((pred - y) ** 2)).item()) if len(y) else 0.0
    corr = float(np.corrcoef(pred.numpy(), y.numpy())[0, 1]) if len(y) > 1 else 0.0

    train_data = np.load(Path(args.data_dir) / "train.npz")
    train_y = torch.from_numpy(train_data["y"]).float()
    naive = float(train_y.mean().item()) if len(train_y) else 0.0
    naive_mae = float(torch.mean(torch.abs(y - naive)).item()) if len(y) else 0.0

    out = {
        "split": args.split,
        "samples": int(len(y)),
        "mae": mae,
        "rmse": rmse,
        "pearson_corr": corr,
        "naive_mae": naive_mae,
    }
    with open(model_dir / f"eval_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
