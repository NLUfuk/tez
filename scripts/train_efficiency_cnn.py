#!/usr/bin/env python3
"""Train a compact CNN to predict efficiency score."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class EfficiencyCNN(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN for efficiency score prediction")
    parser.add_argument("--data-dir", type=str, default="outputs/datasets/efficiency")
    parser.add_argument("--out", type=str, default="outputs/models/efficiency_cnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(path: Path):
    data = np.load(path)
    X = torch.from_numpy(data["X"]).float()
    y = torch.from_numpy(data["y"]).float()
    return X, y


def normalize_inputs(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (X - mean) / std


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            losses.append(loss.item())
            preds.append(out.detach().cpu().numpy())
            targets.append(yb.detach().cpu().numpy())
    if not losses:
        return {"loss": math.inf, "mae": math.inf, "rmse": math.inf}
    preds_arr = np.concatenate(preds)
    targets_arr = np.concatenate(targets)
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    rmse = float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2)))
    return {"loss": float(np.mean(losses)), "mae": mae, "rmse": rmse}


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split(data_dir / "train.npz")
    X_val, y_val = load_split(data_dir / "val.npz")
    X_test, y_test = load_split(data_dir / "test.npz")

    # Feature and target normalization from train split only.
    x_mean = X_train.mean(dim=(0, 2, 3), keepdim=True)
    x_std = X_train.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    y_mean = y_train.mean()
    y_std = y_train.std().clamp_min(1e-6)

    X_train = normalize_inputs(X_train, x_mean, x_std)
    X_val = normalize_inputs(X_val, x_mean, x_std)
    X_test = normalize_inputs(X_test, x_mean, x_std)
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    train_loader = DataLoader(
        TensorDataset(X_train, y_train_n),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val_n),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test_n),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficiencyCNN(in_channels=X_train.shape[1]).to(device)
    criterion = nn.HuberLoss(delta=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = math.inf
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else math.inf
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
            }
        )

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_mae={val_metrics['mae']:.5f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    test_norm_metrics = evaluate(model, test_loader, criterion, device)

    # Compute denormalized metrics for thesis reporting.
    model.eval()
    preds_n = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds_n.append(model(xb.to(device)).cpu())
    pred_n = torch.cat(preds_n) if preds_n else torch.empty(0)
    pred = pred_n * y_std + y_mean
    test_mae = float(torch.mean(torch.abs(pred - y_test)).item()) if len(y_test) else math.inf
    test_rmse = float(torch.sqrt(torch.mean((pred - y_test) ** 2)).item()) if len(y_test) else math.inf

    naive_pred = float(y_train.mean().item()) if len(y_train) else 0.0
    naive_mae = float(torch.mean(torch.abs(y_test - naive_pred)).item()) if len(y_test) else math.inf
    naive_rmse = float(torch.sqrt(torch.mean((y_test - naive_pred) ** 2)).item()) if len(y_test) else math.inf

    result = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_loss_normalized": test_norm_metrics["loss"],
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "naive_mae": naive_mae,
        "naive_rmse": naive_rmse,
        "device": str(device),
        "epochs_trained": len(history),
    }

    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    torch.save(
        {
            "x_mean": x_mean.cpu(),
            "x_std": x_std.cpu(),
            "y_mean": y_mean.cpu(),
            "y_std": y_std.cpu(),
        },
        out_dir / "normalization.pt",
    )

    print("-" * 70)
    print(f"Model saved: {out_dir / 'best_model.pt'}")
    print(f"Test MAE={result['test_mae']:.5f} (naive={result['naive_mae']:.5f})")
    print(f"Test RMSE={result['test_rmse']:.5f} (naive={result['naive_rmse']:.5f})")


if __name__ == "__main__":
    main()
