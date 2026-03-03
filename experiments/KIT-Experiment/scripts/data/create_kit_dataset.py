#!/usr/bin/env python3
"""
Build train/val/test .npz dataset files from formatted KIT data.

Self-contained adaptation of create_shared_dataset for the KIT experiment.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent.parent
GARAGE_ROOT = EXPERIMENT_ROOT.parent.parent
sys.path.insert(0, str(GARAGE_ROOT.parent))
sys.path.insert(0, str(EXPERIMENT_ROOT))
from fault_injection import inject_faults_with_sensor_labels

FORMATTED_DIR = EXPERIMENT_ROOT / "data" / "formatted"
OUTPUT_DIR = EXPERIMENT_ROOT / "data" / "dataset"

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
STRIDE = 30
FORECAST_HORIZONS = [1, 5, 10]  # For Stage 1 compatibility
MIN_WINDOWS_PER_DRIVE = 30
FAULT_RATE = 0.25
RANDOM_SEED = 42
MIN_TIMESTEPS = WINDOW_SIZE + 10 + 1  # 311


def remove_zero_variance_columns(df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_check = [c for c in numeric_cols if c not in exclude_cols]
    if not cols_to_check:
        return df
    std_df = df[cols_to_check].std()
    zero_var = std_df[std_df == 0].index.tolist()
    if zero_var:
        df = df.drop(columns=[c for c in zero_var if c in df.columns])
    return df


def add_cross_channel_features(data: pd.DataFrame) -> pd.DataFrame:
    if "ENGINE_RPM ()" in data.columns and "VEHICLE_SPEED ()" in data.columns:
        data["RPM_SPEED_RATIO"] = data["ENGINE_RPM ()"] / (data["VEHICLE_SPEED ()"] + 1)
    if "THROTTLE ()" in data.columns and "ENGINE_LOAD ()" in data.columns:
        data["THROTTLE_LOAD_RATIO"] = data["THROTTLE ()"] / (data["ENGINE_LOAD ()"] + 1)
    if "VEHICLE_SPEED ()" in data.columns:
        data["IS_IDLE"] = (data["VEHICLE_SPEED ()"] < 5).astype(float)
        data["IS_HIGHWAY"] = (data["VEHICLE_SPEED ()"] > 60).astype(float)
    if "ENGINE_RPM ()" in data.columns:
        data["RPM_ACCEL"] = data.groupby(ID_COL)["ENGINE_RPM ()"].diff().fillna(0)
    return data


def filter_long_drives(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    lengths = df.groupby(ID_COL).size()
    valid = lengths[lengths >= min_length].index
    return df[df[ID_COL].isin(valid)].reset_index(drop=True)


def build_windows(
    df: pd.DataFrame,
    sensor_cols: list,
    scaler: Optional[MinMaxScaler],
) -> tuple:
    """Build windows with forecast targets for Stage 1. Returns (X, drive_ids, y_forecast, scaler)."""
    df = df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    max_h = max(FORECAST_HORIZONS)
    min_T = WINDOW_SIZE + max_h  # need t+300+10-1 = t+309 < T

    X_list, drive_ids_list, y_list = [], [], []

    for drive_id, group in df.groupby(ID_COL):
        vals = group[sensor_cols].values.astype(np.float32)
        T, _ = vals.shape
        if T < min_T:
            continue
        for t in range(0, T - min_T + 1, STRIDE):
            X_list.append(vals[t : t + WINDOW_SIZE])
            y_targets = np.stack([vals[t + WINDOW_SIZE + h - 1] for h in FORECAST_HORIZONS], axis=0)
            y_list.append(y_targets)
            drive_ids_list.append(str(drive_id))

    X = np.stack(X_list, dtype=np.float32)
    y_forecast = np.stack(y_list, dtype=np.float32)  # (N, 3, 8)
    drive_ids = np.array(drive_ids_list)

    if scaler is None:
        scaler = MinMaxScaler()
        X_flat = X.reshape(-1, len(sensor_cols))
        scaler.fit(X_flat)
    X_flat = X.reshape(-1, len(sensor_cols))
    X_norm = scaler.transform(X_flat).astype(np.float32)
    X = X_norm.reshape(-1, WINDOW_SIZE, len(sensor_cols))
    y_flat = y_forecast.reshape(-1, len(sensor_cols))
    y_norm = scaler.transform(y_flat).astype(np.float32)
    y_forecast = y_norm.reshape(-1, len(FORECAST_HORIZONS), len(sensor_cols))
    return X, drive_ids, y_forecast, scaler


def process_split(
    data: pd.DataFrame,
    sensor_cols: list,
    scaler: Optional[MinMaxScaler],
    fault_rate: float,
    fault_seed: int,
) -> tuple:
    X_clean, drive_ids, y_forecast, scaler = build_windows(data, sensor_cols, scaler)
    X_t = torch.tensor(X_clean.copy(), dtype=torch.float32)
    y_dummy = X_t[:, -1, :]
    X_t, _, sensor_labels, window_labels, _ = inject_faults_with_sensor_labels(
        X_t, y_dummy, sensor_cols,
        fault_percentage=fault_rate,
        random_state=fault_seed,
    )
    X_faulty = X_t.numpy().astype(np.float32)
    y_window = window_labels.numpy().astype(np.int64)
    y_sensor = sensor_labels.numpy().astype(np.int64)

    counts = pd.Series(drive_ids).value_counts()
    valid_drives = counts[counts >= MIN_WINDOWS_PER_DRIVE].index
    mask = np.isin(drive_ids, valid_drives)
    dropped = (~mask).sum()
    if dropped > 0:
        n_dropped_drives = (counts < MIN_WINDOWS_PER_DRIVE).sum()
        print(f"  Warning: Dropped {dropped} windows from {n_dropped_drives} drives (< {MIN_WINDOWS_PER_DRIVE} windows)")
        X_clean = X_clean[mask]
        X_faulty = X_faulty[mask]
        y_window = y_window[mask]
        y_sensor = y_sensor[mask]
        y_forecast = y_forecast[mask]
        drive_ids = drive_ids[mask]

    return X_clean, X_faulty, y_window, y_sensor, y_forecast, drive_ids, scaler


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading formatted KIT CSVs...")
    csv_files = sorted(FORMATTED_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs in {FORMATTED_DIR}")

    df_list = [pd.read_csv(f) for f in csv_files]
    data = pd.concat(df_list, ignore_index=True)
    print(f"   Loaded {len(csv_files)} files, {len(data):,} rows")

    print("2. Removing zero-variance columns...")
    data = remove_zero_variance_columns(data, exclude_cols=[ID_COL])
    sensor_cols = [c for c in SENSOR_COLS if c in data.columns]
    if len(sensor_cols) != 8:
        raise ValueError(f"Expected 8 sensors, got {len(sensor_cols)}: {sensor_cols}")

    print("3. Filtering short drives (< 311 timesteps)...")
    data = filter_long_drives(data, MIN_TIMESTEPS)
    print(f"   Kept {data[ID_COL].nunique()} drives")

    print("4. Adding cross-channel features...")
    data = add_cross_channel_features(data)
    data = data.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    print("5. Splitting drives (70% train, 15% val, 15% test)...")
    unique_drives = np.sort(data[ID_COL].unique())
    n = len(unique_drives)
    train_drives = unique_drives[: int(0.70 * n)]
    val_drives = unique_drives[int(0.70 * n) : int(0.85 * n)]
    test_drives = unique_drives[int(0.85 * n) :]
    print(f"   Train: {len(train_drives)}, Val: {len(val_drives)}, Test: {len(test_drives)}")

    train_data = data[data[ID_COL].isin(train_drives)].copy()
    val_data = data[data[ID_COL].isin(val_drives)].copy()
    test_data = data[data[ID_COL].isin(test_drives)].copy()

    print("6. Building windows and injecting faults...")
    scaler = None
    X_train_clean, X_train, y_train_w, y_train_s, y_train_f, did_train, scaler = process_split(
        train_data, sensor_cols, scaler, FAULT_RATE, fault_seed=RANDOM_SEED
    )
    X_val_clean, X_val, y_val_w, y_val_s, y_val_f, did_val, _ = process_split(
        val_data, sensor_cols, scaler, FAULT_RATE, fault_seed=RANDOM_SEED + 1
    )
    X_test_clean, X_test, y_test_w, y_test_s, y_test_f, did_test, _ = process_split(
        test_data, sensor_cols, scaler, FAULT_RATE, fault_seed=RANDOM_SEED + 2
    )

    print("7. Saving .npz files (X_clean for Stage 1, X for Stage 2)...")
    np.savez_compressed(
        OUTPUT_DIR / "train.npz",
        X_clean=X_train_clean, X=X_train, y_window=y_train_w, y_sensor=y_train_s,
        y_forecast=y_train_f, drive_ids=did_train,
    )
    np.savez_compressed(
        OUTPUT_DIR / "val.npz",
        X_clean=X_val_clean, X=X_val, y_window=y_val_w, y_sensor=y_val_s,
        y_forecast=y_val_f, drive_ids=did_val,
    )
    np.savez_compressed(
        OUTPUT_DIR / "test.npz",
        X_clean=X_test_clean, X=X_test, y_window=y_test_w, y_sensor=y_test_s,
        y_forecast=y_test_f, drive_ids=did_test,
    )

    def pct(x):
        return 100 * (x > 0).sum() / len(x) if len(x) > 0 else 0

    print("\n--- Summary ---")
    print(f"Train: {len(X_train)} windows, {len(np.unique(did_train))} drives, {pct(y_train_w):.1f}% fault rate")
    print(f"Val:   {len(X_val)} windows, {len(np.unique(did_val))} drives, {pct(y_val_w):.1f}% fault rate")
    print(f"Test:  {len(X_test)} windows, {len(np.unique(did_test))} drives, {pct(y_test_w):.1f}% fault rate")


if __name__ == "__main__":
    main()
