#!/usr/bin/env python3
"""
Stage 2: Supervised Center Loss Training
Training script for GDN with sensor-level center loss.

Objective: Learn separate centers for normal vs. anomalous per-sensor embeddings
to enable better sensor attribution in knowledge graphs.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.gdn_model import GDN
from models.center_loss import SensorOnlyCenterLoss
from data.fault_injection import inject_faults_with_sensor_labels

# Reuse data preprocessing from Stage 1
from training.train_stage1 import (
    remove_zero_variance_columns,
    mean_fill_missing_timestamps_and_remove_duplicates,
    downsample,
    filter_long_drives,
    add_cross_channel_features,
    build_forecast_windows,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)

torch.set_default_dtype(torch.float32)

# ============================================================================
# Constants
# ============================================================================

# Data path - relative to GARAGE-Final directory
DATA_PATH = str(Path(__file__).parent.parent / "data" / "carOBD" / "obdiidata")
SENSOR_COLS = [
    "ENGINE_RPM ()",
    "VEHICLE_SPEED ()",
    "THROTTLE ()",
    "ENGINE_LOAD ()",
    "COOLANT_TEMPERATURE ()",
    "INTAKE_MANIFOLD_PRESSURE ()",
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
    "LONG_TERM_FUEL_TRIM_BANK_1 ()",
]
ID_COL = "drive_id"
TIME_COL = "ENGINE_RUN_TINE ()"
WINDOW_SIZE = 300

# Training hyperparameters
NUM_EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 5e-4  # Reduced from 1e-3 (fine-tuning, not training from scratch)
WEIGHT_DECAY = 1e-4
LAMBDA_CENTER = 0.5
LAMBDA_GLOBAL = 0.3

# Model architecture (must match Stage 1)
EMBED_DIM = 16
TOP_K = 5
HIDDEN_DIM = 32

# Center Loss parameters
MLC_MARGIN = 2.0
MLC_LAMBDA_INTRA = 1.5


def build_clean_windows(
    df, sensor_cols, id_col, time_col, window_size, horizon=1, scaler=None
):
    """Build windows from CLEAN data only. Returns normalized windows."""
    df = df.copy().sort_values([id_col, time_col])
    df_sensors = df[[id_col, time_col] + sensor_cols].copy()

    if scaler is None:
        scaler = MinMaxScaler()
        df_sensors[sensor_cols] = scaler.fit_transform(df_sensors[sensor_cols])
    else:
        df_sensors[sensor_cols] = scaler.transform(df_sensors[sensor_cols])

    X_list, y_list = [], []

    for drive_id, group in df_sensors.groupby(id_col):
        values = group[sensor_cols].values
        T_, num_sensors = values.shape
        if T_ <= window_size + horizon:
            continue

        for t in range(T_ - window_size - horizon + 1):
            X_window = values[t : t + window_size]
            y_target = values[t + window_size + horizon - 1]
            X_list.append(X_window)
            y_list.append(y_target)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(np.stack(y_list), dtype=torch.float32)
    return X, y, scaler


# ============================================================================
# Training Function
# ============================================================================


def train_stage2(
    train_loader,
    val_loader,
    num_sensors,
    window_size,
    stage1_checkpoint_path,
    num_epochs=NUM_EPOCHS,
    device="cpu",
    lambda_center=LAMBDA_CENTER,
    lambda_global=LAMBDA_GLOBAL,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    checkpoint_dir="checkpoints",
    use_compile=False,
    compile_mode="reduce-overhead",
    gradient_accumulation_steps=1,
    use_amp=False,
    max_batches_per_epoch=None,
):
    """
    Train GDN with sensor-level center loss.

    Args:
        stage1_checkpoint_path: Path to Stage 1 checkpoint to load

    Returns:
        model: Trained model
        center_loss: Trained SensorOnlyCenterLoss module
    """
    # Load Stage 1 checkpoint
    print(f"\nLoading Stage 1 checkpoint from {stage1_checkpoint_path}...")
    stage1_checkpoint = torch.load(stage1_checkpoint_path, map_location=device)

    # Initialize model
    model = GDN(
        num_nodes=num_sensors,
        window_size=window_size,
        embed_dim=EMBED_DIM,
        top_k=TOP_K,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    # Apply torch.compile() if requested
    if use_compile and hasattr(torch, "compile"):
        print(f"\nCompiling model with mode='{compile_mode}'...")
        print("  Note: First epoch will be slower due to compilation")
        model = torch.compile(model, mode=compile_mode)
    elif use_compile:
        print("\nWarning: torch.compile() not available (requires PyTorch 2.0+)")
        print("  Continuing without compilation")

    # Load base model state (from Stage 1)
    if "base_model_state_dict" in stage1_checkpoint:
        # Stage 1 used GDNWithForecasting wrapper
        base_state = stage1_checkpoint["base_model_state_dict"]
        # Remove 'base_model.' prefix if present
        new_state_dict = {}
        for k, v in base_state.items():
            if k.startswith("base_model."):
                new_state_dict[k[len("base_model.") :]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("  ✓ Loaded model state from Stage 1")
    elif "model_state_dict" in stage1_checkpoint:
        model.load_state_dict(stage1_checkpoint["model_state_dict"], strict=False)
        print("  ✓ Loaded model state from Stage 1")
    else:
        print("  ⚠ No model state found in checkpoint, starting from scratch")

    # Unfreeze sensor embeddings with reduced LR for fine-tuning
    print("\nUnfreezing sensor embeddings with reduced learning rate...")
    model.sensor_embeddings.requires_grad = True

    # Create separate parameter groups
    embedding_params = [model.sensor_embeddings]
    other_params = [
        p
        for p in model.parameters()
        if p is not model.sensor_embeddings and p.requires_grad
    ]

    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "lr": learning_rate, "weight_decay": weight_decay},
            {
                "params": embedding_params,
                "lr": learning_rate * 0.1,
                "weight_decay": weight_decay,
            },
        ]
    )
    print(f"  Sensor embeddings unfrozen with reduced LR: {learning_rate * 0.1:.6f}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Initialize sensor-only center loss
    center_loss = SensorOnlyCenterLoss(
        embed_dim=HIDDEN_DIM,
        num_sensors=num_sensors,
        num_classes=2,
        margin=MLC_MARGIN,
        lambda_intra=MLC_LAMBDA_INTRA,
    ).to(device)

    # Initialize sensor centers at 120° angle
    print("\nInitializing sensor centers at 120° angle...")
    print(f"  Sensor centers: {num_sensors} sensors × 2 classes = {num_sensors * 2} centers")

    with torch.no_grad():
        target_dot = -0.5  # 120° angle

        for sensor_idx in range(num_sensors):
            s_c0 = torch.randn(HIDDEN_DIM, device=device)
            s_c1 = torch.randn(HIDDEN_DIM, device=device)

            s_c0 = F.normalize(s_c0, p=2, dim=0)
            s_c1 = F.normalize(s_c1, p=2, dim=0)

            # Perpendicular component
            s_c1_perp = s_c1 - (s_c1 @ s_c0) * s_c0
            s_c1_perp = F.normalize(s_c1_perp, p=2, dim=0)

            # 120° angle
            s_c1_new = (
                target_dot * s_c0
                + torch.sqrt(torch.tensor(1 - target_dot**2, device=device)) * s_c1_perp
            )
            s_c1_new = F.normalize(s_c1_new, p=2, dim=0)

            center_loss.sensor_centers[sensor_idx, 0].copy_(s_c0)
            center_loss.sensor_centers[sensor_idx, 1].copy_(s_c1_new)

    print(f"  ✓ All sensor centers initialized at 120° (separation ~1.73)")

    # Use Adam for center updates
    optimizer_center = torch.optim.Adam(center_loss.parameters(), lr=0.01)

    # Mixed precision training (AMP) - only for CUDA
    scaler = None
    if use_amp and device.startswith("cuda"):
        scaler = GradScaler()
        print("  ✓ Mixed precision training (AMP) enabled for CUDA")
    elif use_amp:
        print("  ⚠ AMP requested but not on CUDA device, disabling AMP")
        use_amp = False

    # Loss functions
    sensor_criterion = nn.BCEWithLogitsLoss(reduction="none")
    global_criterion = nn.BCEWithLogitsLoss()

    # Learning rate schedulers
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    scheduler_center = torch.optim.lr_scheduler.StepLR(
        optimizer_center, step_size=10, gamma=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 15

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "stage2_best.pt")

    print(f"\n{'=' * 80}")
    print("Stage 2: Sensor-Only Center Loss Training")
    print(f"{'=' * 80}")
    print(f"Embedding dim: {EMBED_DIM}, Hidden dim: {HIDDEN_DIM}")
    print(f"Lambda_center: {lambda_center}, Lambda_global: {lambda_global}")
    print(f"MLC margin: {MLC_MARGIN}, MLC lambda_intra: {MLC_LAMBDA_INTRA}")
    print(f"Trainable parameters: all (sensor_embeddings with reduced LR: {learning_rate * 0.1:.6f})")
    print(f"Device: {device}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Mixed precision (AMP): {use_amp}")
    print(f"Model compilation: {use_compile}\n")

    for epoch in range(num_epochs):
        model.train()
        center_loss.train()

        train_loss_sensor = 0.0
        train_loss_global = 0.0
        train_loss_center = 0.0

        train_iter = train_loader
        if max_batches_per_epoch:
            train_iter = list(train_loader)[:max_batches_per_epoch]

        with tqdm(
            train_iter, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        ) as pbar:
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            for batch_idx, (X_batch, _, sensor_labels_batch, window_labels_batch) in enumerate(pbar):
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)

                # Forward pass with optional AMP
                if use_amp and scaler is not None:
                    with autocast():
                        sensor_logits, global_logits, sensor_embeddings = model(
                            X_batch,
                            return_global=True,
                            return_sensor_embeddings=True,
                        )

                        # Classification losses
                        loss_sensor_clf = sensor_criterion(
                            sensor_logits, sensor_labels_batch
                        ).mean()
                        loss_global_clf = global_criterion(
                            global_logits, window_labels_batch.float()
                        )

                        # Normalize sensor embeddings
                        sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=2)

                        # Sensor-only center loss
                        loss_center = center_loss(
                            sensor_embeddings=sensor_embeddings,
                            sensor_labels=sensor_labels_batch.long(),
                        )

                        # Combined loss
                        loss = (
                            loss_sensor_clf
                            + lambda_global * loss_global_clf
                            + lambda_center * loss_center
                        ) / gradient_accumulation_steps
                else:
                    sensor_logits, global_logits, sensor_embeddings = model(
                        X_batch,
                        return_global=True,
                        return_sensor_embeddings=True,
                    )

                    loss_sensor_clf = sensor_criterion(
                        sensor_logits, sensor_labels_batch
                    ).mean()
                    loss_global_clf = global_criterion(
                        global_logits, window_labels_batch.float()
                    )

                    sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=2)

                    loss_center = center_loss(
                        sensor_embeddings=sensor_embeddings,
                        sensor_labels=sensor_labels_batch.long(),
                    )

                    loss = (
                        loss_sensor_clf
                        + lambda_global * loss_global_clf
                        + lambda_center * loss_center
                    ) / gradient_accumulation_steps

                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf in loss at epoch {epoch + 1}, skipping batch")
                    continue

                # Backward pass
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update every gradient_accumulation_steps batches
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if use_amp and scaler is not None:
                        scaler.unscale_(optimizer)
                        scaler.unscale_(optimizer_center)

                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(
                        center_loss.parameters(), max_norm=0.5
                    )

                    # Update optimizers
                    if use_amp and scaler is not None:
                        scaler.step(optimizer)
                        scaler.step(optimizer_center)
                        scaler.update()
                    else:
                        optimizer.step()
                        optimizer_center.step()

                    optimizer.zero_grad()
                    optimizer_center.zero_grad()

                train_loss_sensor += (
                    loss_sensor_clf.item() * X_batch.size(0) * gradient_accumulation_steps
                )
                train_loss_global += (
                    loss_global_clf.item() * X_batch.size(0) * gradient_accumulation_steps
                )
                train_loss_center += (
                    loss_center.item() * X_batch.size(0) * gradient_accumulation_steps
                )

        # Update for any remaining accumulated gradients
        if (batch_idx + 1) % gradient_accumulation_steps != 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                scaler.unscale_(optimizer_center)

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(center_loss.parameters(), max_norm=0.5)

            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.step(optimizer_center)
                scaler.update()
            else:
                optimizer.step()
                optimizer_center.step()

            optimizer.zero_grad()
            optimizer_center.zero_grad()

        train_loss_sensor /= len(train_loader.dataset)
        train_loss_global /= len(train_loader.dataset)
        train_loss_center /= len(train_loader.dataset)

        # Validation
        model.eval()
        center_loss.eval()
        val_loss_sensor = 0.0
        val_loss_global = 0.0
        val_loss_center = 0.0

        with torch.no_grad():
            for X_batch, _, sensor_labels_batch, window_labels_batch in val_loader:
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)

                sensor_logits, global_logits, sensor_embeddings = model(
                    X_batch,
                    return_global=True,
                    return_sensor_embeddings=True,
                )

                loss_sensor = sensor_criterion(sensor_logits, sensor_labels_batch).mean()
                loss_global = global_criterion(global_logits, window_labels_batch.float())

                sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=2)

                loss_center_val = center_loss(
                    sensor_embeddings=sensor_embeddings,
                    sensor_labels=sensor_labels_batch.long(),
                )

                val_loss_sensor += loss_sensor.item() * X_batch.size(0)
                val_loss_global += loss_global.item() * X_batch.size(0)
                val_loss_center += loss_center_val.item() * X_batch.size(0)

        val_loss_sensor /= len(val_loader.dataset)
        val_loss_global /= len(val_loader.dataset)
        val_loss_center /= len(val_loader.dataset)

        val_total_loss = (
            val_loss_sensor + lambda_global * val_loss_global + lambda_center * val_loss_center
        )

        # Update scheduler
        scheduler.step(val_total_loss)
        scheduler_center.step()

        # Get separation metrics
        separations = center_loss.get_separations()

        # Logging
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Sensor: {train_loss_sensor:.4f} | "
                f"Global: {train_loss_global:.4f} | "
                f"Center: {train_loss_center:.4f} | "
                f"Val Total: {val_total_loss:.4f} | "
                f"Sensor mean sep: {separations['sensor_mean_separation']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Sensor: {train_loss_sensor:.4f} | "
                f"Global: {train_loss_global:.4f} | "
                f"Center: {train_loss_center:.4f} | "
                f"Val Total: {val_total_loss:.4f}"
            )

        # Save best model
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "center_loss_state_dict": center_loss.state_dict(),
                    "sensor_centers": center_loss.get_sensor_centers().cpu(),
                    "separations": separations,
                    "sensor_names": SENSOR_COLS,
                    "window_size": window_size,
                    "embed_dim": EMBED_DIM,
                    "top_k": TOP_K,
                    "hidden_dim": HIDDEN_DIM,
                    "sensor_embeddings": model.sensor_embeddings.data.cpu(),
                    "lambda_center": lambda_center,
                    "lambda_global": lambda_global,
                    "epoch": epoch + 1,
                    "best_val_loss": val_total_loss,
                    "stage": 2,
                },
                best_checkpoint_path,
            )
            print(f"  ✓ Best model saved (Val Loss: {val_total_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(
                f"\nEarly stopping at epoch {epoch + 1} "
                f"(no improvement for {max_patience} epochs)"
            )
            break

    # Load best checkpoint
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    center_loss.load_state_dict(checkpoint["center_loss_state_dict"])

    print(f"\n{'=' * 80}")
    print(f"Stage 2 training complete!")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print(
        f"Sensor mean separation: {checkpoint['separations']['sensor_mean_separation']:.4f}"
    )
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"{'=' * 80}\n")

    return model, center_loss


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Supervised Center Loss Training")
    parser.add_argument(
        "--data_path", type=str, default=DATA_PATH, help="Path to data directory"
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        required=True,
        help="Path to Stage 1 checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--lambda_center", type=float, default=LAMBDA_CENTER, help="Center loss weight"
    )
    parser.add_argument(
        "--lambda_global", type=float, default=LAMBDA_GLOBAL, help="Global loss weight"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Force CPU usage (disable CUDA auto-detection)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients before updating",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="Use torch.compile() to optimize model (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile() mode",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision (AMP) for CUDA devices",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--max_batches_per_epoch",
        type=int,
        default=None,
        help="Limit number of batches per epoch (for testing)",
    )
    args = parser.parse_args()

    # Device detection
    if args.cpu_only:
        device = "cpu"
        print("Using device: cpu (forced via --cpu_only flag)")
    elif args.device is not None:
        device = args.device
        if device == "mps":
            print("Warning: MPS not supported. Falling back to CPU.")
            device = "cpu"
        print(f"Using device: {device} (specified via --device)")
    else:
        if torch.cuda.is_available():
            device = "cuda"
            print("Using device: cuda (auto-detected)")
        else:
            device = "cpu"
            print("Using device: cpu (auto-detected)")

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    df_list = []
    for file in os.listdir(args.data_path):
        if file.endswith(".csv"):
            df = pd.read_csv(f"{args.data_path}/{file}", index_col=False)
            df["drive_id"] = file
            df_list.append(df)

    print(f"Loaded {len(df_list)} files")

    data = pd.concat(df_list, ignore_index=True)
    print(f"Total samples: {len(data):,}")
    print(f"Unique drives: {data[ID_COL].nunique()}")

    # Preprocessing
    print("\nPreprocessing data...")
    data = data.drop(
        columns=[
            "WARM_UPS_SINCE_CODES_CLEARED ()",
            "TIME_SINCE_TROUBLE_CODES_CLEARED ()",
        ],
        errors="ignore",
    )
    data = mean_fill_missing_timestamps_and_remove_duplicates(
        data, time_col=TIME_COL, id_cols=[ID_COL]
    )
    data = remove_zero_variance_columns(data, exclude_cols=[ID_COL])
    data = downsample(
        data, time_col=TIME_COL, source_file_col=ID_COL, downsample_factor=2
    )
    data = filter_long_drives(data, id_col=ID_COL, min_length=WINDOW_SIZE + 1)
    data = add_cross_channel_features(data)
    print("Added cross-channel features")

    data = data.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    # Split by drive (TRAIN_RATIO / VAL_RATIO / TEST_RATIO)
    print("\nSplitting data by drive...")
    unique_drives = np.sort(data[ID_COL].unique())
    n_drives = len(unique_drives)
    t_end = int(TRAIN_RATIO * n_drives)
    v_end = int((TRAIN_RATIO + VAL_RATIO) * n_drives)
    train_drives = unique_drives[:t_end]
    val_drives = unique_drives[t_end:v_end]
    test_drives = unique_drives[v_end:]

    print(
        f"Train drives: {len(train_drives)}, Val drives: {len(val_drives)}, Test drives: {len(test_drives)}"
    )

    train_data = data[data[ID_COL].isin(train_drives)].copy()
    val_data = data[data[ID_COL].isin(val_drives)].copy()

    print(f"Train shape: {train_data.shape}, Val shape: {val_data.shape}")

    # Build clean windows
    print("\nBuilding clean windows...")
    X_train, y_train, scaler_train = build_clean_windows(
        train_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=None
    )
    X_val, y_val, _ = build_clean_windows(
        val_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=scaler_train
    )

    print(f"Clean train windows: {len(X_train)}")
    print(f"Clean val windows: {len(X_val)}")

    # Inject faults with sensor-level labels
    print("\nInjecting faults with sensor-level labels (stratified distribution)...")
    X_train_sensor, _, train_sensor_labels, train_window_labels, _ = (
        inject_faults_with_sensor_labels(
            X_train,
            y_train,
            SENSOR_COLS,
            fault_percentage=0.15,
            random_state=42,
            use_stratified=True,
        )
    )
    X_val_sensor, _, val_sensor_labels, val_window_labels, _ = (
        inject_faults_with_sensor_labels(
            X_val,
            y_val,
            SENSOR_COLS,
            fault_percentage=0.15,
            random_state=43,
            use_stratified=True,
        )
    )

    # Statistics
    train_faulty = (train_sensor_labels.sum(dim=1) > 0).sum().item()
    val_faulty = (val_sensor_labels.sum(dim=1) > 0).sum().item()

    print(f"\nTrain: {train_faulty}/{len(X_train_sensor)} faulty windows")
    print(f"Val:   {val_faulty}/{len(X_val_sensor)} faulty windows")

    # Create dataloaders
    train_ds = TensorDataset(
        X_train_sensor, y_train, train_sensor_labels, train_window_labels
    )
    val_ds = TensorDataset(X_val_sensor, y_val, val_sensor_labels, val_window_labels)

    pin_memory = device.startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    num_sensors = len(SENSOR_COLS)
    print(f"\nTrain windows: {len(train_ds)}, Sensors: {num_sensors}")

    # Train model
    model, center_loss = train_stage2(
        train_loader,
        val_loader,
        num_sensors=num_sensors,
        window_size=WINDOW_SIZE,
        stage1_checkpoint_path=args.stage1_checkpoint,
        num_epochs=args.epochs,
        device=device,
        lambda_center=args.lambda_center,
        lambda_global=args.lambda_global,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        checkpoint_dir=args.checkpoint_dir,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        max_batches_per_epoch=args.max_batches_per_epoch,
    )

    print("✓ Stage 2 training complete!")


if __name__ == "__main__":
    main()
