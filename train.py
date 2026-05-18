"""
Main training script for GlucoSense AI — glucose prediction models.

Usage:
    python train.py --config configs/training_config.yaml
    python train.py --config configs/training_config.yaml --model temporal_cnn
    python train.py --config configs/training_config.yaml --model all   # ablation
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.models import GlucoseLSTM, GlucoseTemporalCNN, LinearBaseline, NaiveLastValueBaseline
from src.preprocessing import CGMProcessor
from src.evaluation import compute_all_metrics, clarke_error_grid, plot_clarke_error_grid
from src.visualization import plot_learning_curves, plot_ablation_table


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: str) -> torch.device:
    if cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg)


def build_model(model_type: str, cfg: dict, num_horizons: int) -> nn.Module:
    if model_type == "lstm":
        c = cfg["model"]["lstm"]
        return GlucoseLSTM(
            hidden_size=c["hidden_size"],
            num_layers=c["num_layers"],
            dropout=c["dropout"],
            bidirectional=c["bidirectional"],
            num_horizons=num_horizons,
        )
    if model_type == "temporal_cnn":
        c = cfg["model"]["temporal_cnn"]
        return GlucoseTemporalCNN(
            num_channels=c["num_channels"],
            kernel_size=c["kernel_size"],
            dropout=c["dropout"],
            num_horizons=num_horizons,
        )
    if model_type == "baseline":
        return LinearBaseline(seq_len=cfg["data"]["seq_len"], num_horizons=num_horizons)
    raise ValueError(f"Unknown model type: {model_type}")


def load_data(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load OhioT1DM data or fall back to synthetic sample data.

    OhioT1DM has one XML file per patient — timestamps overlap across patients,
    so each file is cleaned independently before windows are concatenated.
    """
    import pandas as pd

    data_dir = Path(cfg["data"]["data_dir"])
    sample_dir = Path(cfg["data"]["sample_dir"])

    processor = CGMProcessor(
        seq_len=cfg["data"]["seq_len"],
        pred_horizons=cfg["data"]["pred_horizons"],
    )

    xml_files = sorted(data_dir.glob("*.xml")) if data_dir.exists() else []
    if xml_files:
        print(f"Loading OhioT1DM data from {data_dir} ({len(xml_files)} files)...")

        # Collect all training data to fit a single shared scaler
        all_train_dfs = []
        per_file_splits = []
        for f in xml_files:
            df = processor.load_ohio_xml(str(f))
            df = processor.clean(df)
            n = len(df)
            train_end = int(n * cfg["data"]["train_split"])
            val_end = train_end + int(n * cfg["data"]["val_split"])
            all_train_dfs.append(df.iloc[:train_end])
            per_file_splits.append((df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]))

        processor.fit_scaler(pd.concat(all_train_dfs, ignore_index=True))
        normalize = cfg["data"]["normalize"]

        Xs_train, ys_train = [], []
        Xs_val, ys_val = [], []
        Xs_test, ys_test = [], []
        for train_df, val_df, test_df in per_file_splits:
            if len(train_df) > cfg["data"]["seq_len"] + max(cfg["data"]["pred_horizons"]):
                X, y = processor.make_windows(train_df, normalize=normalize)
                Xs_train.append(X); ys_train.append(y)
            if len(val_df) > cfg["data"]["seq_len"] + max(cfg["data"]["pred_horizons"]):
                X, y = processor.make_windows(val_df, normalize=normalize)
                Xs_val.append(X); ys_val.append(y)
            if len(test_df) > cfg["data"]["seq_len"] + max(cfg["data"]["pred_horizons"]):
                X, y = processor.make_windows(test_df, normalize=normalize)
                Xs_test.append(X); ys_test.append(y)

        X_train = np.concatenate(Xs_train); y_train = np.concatenate(ys_train)
        X_val   = np.concatenate(Xs_val);   y_val   = np.concatenate(ys_val)
        X_test  = np.concatenate(Xs_test);  y_test  = np.concatenate(ys_test)

    else:
        print(f"OhioT1DM data not found — loading synthetic sample data from {sample_dir}.")
        csv_files = list(sample_dir.glob("*.csv"))
        if not csv_files:
            print("No data found. Run: python data/sample/generate_synthetic.py")
            raise FileNotFoundError("No data available. Generate synthetic data first.")
        df = pd.concat([pd.read_csv(f, parse_dates=["datetime"]) for f in csv_files],
                       ignore_index=True)
        df = processor.clean(df)
        n = len(df)
        train_end = int(n * cfg["data"]["train_split"])
        val_end = train_end + int(n * cfg["data"]["val_split"])
        train_df, val_df, test_df = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
        processor.fit_scaler(train_df)
        normalize = cfg["data"]["normalize"]
        X_train, y_train = processor.make_windows(train_df, normalize=normalize)
        X_val,   y_val   = processor.make_windows(val_df,   normalize=normalize)
        X_test,  y_test  = processor.make_windows(test_df,  normalize=normalize)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), processor


def make_loaders(
    X_train, y_train, X_val, y_val, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    def to_tensor(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    train_loader = DataLoader(to_tensor(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(to_tensor(X_val, y_val), batch_size=batch_size)
    return train_loader, val_loader


def train_one_model(
    model_type: str, cfg: dict, train_loader, val_loader, device: torch.device
) -> tuple[nn.Module, list, list]:
    num_horizons = len(cfg["data"]["pred_horizons"])
    model = build_model(model_type, cfg, num_horizons).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []
    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if epoch % cfg["logging"]["log_every"] == 0:
            print(f"  Epoch {epoch:3d}/{cfg['training']['epochs']} | "
                  f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / f"{model_type}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["patience"]:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(checkpoint_dir / f"{model_type}_best.pth", map_location=device))
    return model, train_losses, val_losses


def evaluate(model, X_test, y_test, processor, cfg, device) -> dict:
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    horizons = cfg["data"]["pred_horizons"]
    horizon_minutes = [h * 5 for h in horizons]

    metrics = {}
    for i, h_min in enumerate(horizon_minutes):
        true_denorm = processor.inverse_transform(y_test[:, i])
        pred_denorm = processor.inverse_transform(preds[:, i])
        metrics.update(compute_all_metrics(true_denorm, pred_denorm, h_min))

    # Clarke EGA on the primary horizon (first)
    true_primary = processor.inverse_transform(y_test[:, 0])
    pred_primary = processor.inverse_transform(preds[:, 0])
    zones, _ = clarke_error_grid(true_primary, pred_primary)
    metrics["zone_A_pct"] = zones["A"]
    metrics["zone_B_pct"] = zones["B"]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--model", default=None,
                        help="lstm | temporal_cnn | baseline | all")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"]["type"] = args.model

    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    print(f"Device: {device}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test), processor = load_data(cfg)
    print(f"Data splits — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val,
                                             cfg["training"]["batch_size"])

    model_types = ["lstm", "temporal_cnn", "baseline"] if cfg["model"]["type"] == "all" \
        else [cfg["model"]["type"]]

    all_results = {}
    save_dir = Path(cfg["logging"]["save_dir"])
    (save_dir / "plots").mkdir(parents=True, exist_ok=True)

    for mt in model_types:
        print(f"\nTraining: {mt}")
        if mt in ("lstm", "temporal_cnn", "baseline"):
            model, train_losses, val_losses = train_one_model(
                mt, cfg, train_loader, val_loader, device
            )
            plot_learning_curves(
                train_losses, val_losses,
                title=f"Learning Curves — {mt}",
                save_path=str(save_dir / "plots" / f"{mt}_learning_curves.png"),
            )
        else:
            model = NaiveLastValueBaseline()

        metrics = evaluate(model, X_test, y_test, processor, cfg, device)
        all_results[mt] = metrics
        print(f"  Results: {metrics}")

        # Clarke EGA plot
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        true_mg = processor.inverse_transform(y_test[:, 0])
        pred_mg = processor.inverse_transform(preds[:, 0])
        plot_clarke_error_grid(
            true_mg, pred_mg,
            title=f"Clarke Error Grid — {mt} (30-min horizon)",
            save_path=str(save_dir / "plots" / f"{mt}_clarke_error_grid.png"),
        )

    if len(all_results) > 1:
        plot_ablation_table(
            all_results,
            save_path=str(save_dir / "plots" / "ablation_results.png"),
        )
        print("\nAblation table saved to results/plots/ablation_results.png")

    print("\nTraining complete. Results saved to", save_dir)


if __name__ == "__main__":
    main()
