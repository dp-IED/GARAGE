#!/usr/bin/env python3
"""
KIT-Experiment fault injection with spike, drift, flatline, cluster, and inconsistency patterns.

Fault type distribution: spike 30%, drift 30%, flatline 20%, cluster 10%, inconsistency 10%.

Fault type interpretations (real-world automotive):
- spike:      Sudden transient increase (1.3x) in sensor reading (electrical noise, connector glitch).
              Note: subtle for low-valued sensors (e.g. 0.1 -> 0.13).
- drift:      Gradual offset over time (sensor aging, calibration drift).
- flatline:   Sensor stuck at constant value (sensor failure, wiring short).
- cluster:    Correlated multi-sensor fault in a physical cluster (e.g. MAP sensor failure
              affecting fuel trim and load readings together).
- inconsistency: Opposing patterns in normally correlated sensors (wheel slip, torque
                 converter fault, intake restriction).
"""

import numpy as np
import torch

FAULT_TYPE_DISTRIBUTION = {
    "spike": 0.30,
    "drift": 0.30,
    "flatline": 0.20,
    "cluster": 0.10,
    "inconsistency": 0.10,
}

COMBUSTION_CLUSTER = [
    "ENGINE_LOAD ()",
    "INTAKE_MANIFOLD_PRESSURE ()",
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
]
DRIVETRAIN_CLUSTER = [
    "ENGINE_RPM ()",
    "VEHICLE_SPEED ()",
    "THROTTLE ()",
]

INCONSISTENCY_PAIRS = [
    (("ENGINE_RPM ()", 0.2), ("VEHICLE_SPEED ()", -0.2)),
    (("ENGINE_LOAD ()", 0.2), ("THROTTLE ()", 0.0)),
]


def _apply_spike(win, sensor_idx, window_size):
    start = int(window_size * 0.25)
    end = int(window_size * 0.75)
    win[start:end, sensor_idx] = np.clip(
        win[start:end, sensor_idx] * 1.8, 0.0, 1.0
    )


def _apply_drift(win, sensor_idx, window_size):
    drift = 0.40
    t = np.linspace(0, 1, window_size)
    offset = drift * t
    win[:, sensor_idx] = np.clip(win[:, sensor_idx] + offset, 0.0, 1.0)


def _apply_flatline(win, sensor_idx, window_size):
    mid = window_size // 2
    stuck_value = win[mid, sensor_idx]
    win[:, sensor_idx] = stuck_value


def _apply_cluster(win, pid_idx, sensor_cols, cluster_sensors):
    drift_mag = 0.40
    t = np.linspace(0, 1, win.shape[0])
    offset = drift_mag * t
    for name in cluster_sensors:
        if name in pid_idx:
            idx = pid_idx[name]
            win[:, idx] = np.clip(win[:, idx] + offset, 0.0, 1.0)


def _apply_inconsistency(win, pid_idx, pair):
    (name_a, delta_a), (name_b, delta_b) = pair
    if name_a not in pid_idx or name_b not in pid_idx:
        return
    idx_a, idx_b = pid_idx[name_a], pid_idx[name_b]
    win[:, idx_a] = np.clip(win[:, idx_a] + delta_a, 0.0, 1.0)
    if abs(delta_b) > 1e-6:
        win[:, idx_b] = np.clip(win[:, idx_b] + delta_b, 0.0, 1.0)
    else:
        mid = win.shape[0] // 2
        stuck = win[mid, idx_b]
        win[:, idx_b] = stuck


def inject_faults_with_sensor_labels(
    X_windows,
    y_windows,
    sensor_cols,
    fault_percentage=0.25,
    random_state=42,
):
    """
    Inject faults with spike/drift/flatline/cluster/inconsistency distribution.

    Returns:
        X_faulty, y_windows, sensor_labels, window_labels
    """
    np.random.seed(random_state)

    N, W, D = X_windows.shape
    n_fault = max(1, int(N * fault_percentage))

    if isinstance(X_windows, torch.Tensor):
        X_faulty = X_windows.clone()
    else:
        X_faulty = torch.tensor(X_windows.copy(), dtype=torch.float32)
    sensor_labels = torch.zeros(N, D, dtype=torch.float32)
    window_labels = torch.zeros(N, dtype=torch.float32)

    pid_idx = {name: i for i, name in enumerate(sensor_cols)}
    fault_types = ["spike", "drift", "flatline", "cluster", "inconsistency"]
    fault_probs = [
        FAULT_TYPE_DISTRIBUTION[t] for t in fault_types
    ]
    type_counts = {t: 0 for t in fault_types}

    fault_indices = np.random.choice(N, n_fault, replace=False)

    for idx in fault_indices:
        win = X_faulty[idx].numpy()
        fault_type = np.random.choice(fault_types, p=fault_probs)
        type_counts[fault_type] += 1
        affected_sensors = []

        if fault_type in ("spike", "drift", "flatline"):
            n_faulty_sensors = np.random.randint(1, 4)
            sensor_indices = np.random.choice(D, min(n_faulty_sensors, D), replace=False)
            for si in sensor_indices:
                if fault_type == "spike":
                    _apply_spike(win, si, W)
                elif fault_type == "drift":
                    _apply_drift(win, si, W)
                else:
                    _apply_flatline(win, si, W)
                affected_sensors.append(si)

        elif fault_type == "cluster":
            cluster = np.random.choice(["combustion", "drivetrain"])
            cluster_sensors = (
                COMBUSTION_CLUSTER if cluster == "combustion" else DRIVETRAIN_CLUSTER
            )
            _apply_cluster(win, pid_idx, sensor_cols, cluster_sensors)
            for name in cluster_sensors:
                if name in pid_idx:
                    affected_sensors.append(pid_idx[name])

        else:
            pair = INCONSISTENCY_PAIRS[np.random.randint(len(INCONSISTENCY_PAIRS))]
            _apply_inconsistency(win, pid_idx, pair)
            for (name, _) in pair:
                if name in pid_idx:
                    affected_sensors.append(pid_idx[name])

        if affected_sensors:
            X_faulty[idx] = torch.tensor(win, dtype=torch.float32)
            window_labels[idx] = 1
            for si in set(affected_sensors):
                sensor_labels[idx, si] = 1.0

    total = sum(type_counts.values())
    if total > 0:
        print("Fault type distribution:")
        for t in fault_types:
            pct = 100 * type_counts[t] / total
            print(f"  {t}: {type_counts[t]} ({pct:.1f}%)")

    return X_faulty, y_windows, sensor_labels, window_labels
