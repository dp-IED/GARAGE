#!/usr/bin/env python3
"""
Shared fault injection utilities with stratified sensor distribution.

Ensures even fault distribution across all sensors for balanced evaluation.
Returns per-window fault type strings alongside binary labels.
"""

import numpy as np
import torch


FAULT_TYPE_MAP = {
    "VEHICLE_SPEED ()": "VSS_DROPOUT",
    "INTAKE_MANIFOLD_PRESSURE ()": "MAF_SCALE_LOW",
    "COOLANT_TEMPERATURE ()": "COOLANT_DROPOUT",
    "THROTTLE ()": "TPS_STUCK",
    "ENGINE_RPM ()": "RPM_ANOMALY",
    "ENGINE_LOAD ()": "ENGINE_LOAD_DRIFT",
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()": "STFT_STUCK_HIGH",
    "LONG_TERM_FUEL_TRIM_BANK_1 ()": "LTFT_DRIFT",
}

LEGACY_FAULT_TYPES = [
    "vss_dropout",
    "maf_scale_low",
    "coolant_dropout",
    "tps_stuck",
    "rpm_speed_decouple",
]


def inject_sensor_specific_fault(win, sensor_idx, sensor_name, pid_idx, window_size):
    """
    Inject a fault specific to a given sensor.

    Args:
        win: (window_size, num_sensors) numpy array - window data
        sensor_idx: Index of the sensor to fault
        sensor_name: Name of the sensor
        pid_idx: Dictionary mapping sensor names to indices
        window_size: Size of the time window

    Returns:
        List of affected sensor indices (may include correlated sensors)
    """
    affected_sensors = [sensor_idx]

    if "VEHICLE_SPEED ()" in sensor_name:
        if win[:, sensor_idx].mean() > 0.15:
            start = int(window_size * 0.30)
            end = int(window_size * 0.70)
            win[start:end, sensor_idx] = 0.0
            win[start:end, sensor_idx] += np.random.uniform(0, 0.02, end - start)

    elif "INTAKE_MANIFOLD_PRESSURE ()" in sensor_name:
        scale_factor = np.random.uniform(0.75, 0.80)
        win[:, sensor_idx] = win[:, sensor_idx] * scale_factor
        if "SHORT_TERM_FUEL_TRIM_BANK_1 ()" in pid_idx:
            stft_i = pid_idx["SHORT_TERM_FUEL_TRIM_BANK_1 ()"]
            win[:, stft_i] = np.clip(win[:, stft_i] + 0.15, 0.0, 1.0)
            affected_sensors.append(stft_i)

    elif "COOLANT_TEMPERATURE ()" in sensor_name:
        if win[:, sensor_idx].mean() > 0.5:
            n_dropouts = np.random.randint(2, 5)
            for _ in range(n_dropouts):
                drop_start = np.random.randint(0, window_size - 60)
                drop_len = np.random.randint(30, 60)
                win[drop_start : drop_start + drop_len, sensor_idx] = np.random.uniform(
                    0.05, 0.15
                )

    elif "THROTTLE ()" in sensor_name:
        freeze_point = window_size // 2
        stuck_value = win[freeze_point, sensor_idx]
        if stuck_value > 0.15 and win[:freeze_point, sensor_idx].std() > 0.05:
            win[freeze_point:, sensor_idx] = stuck_value

    elif "ENGINE_RPM ()" in sensor_name:
        if win[:, sensor_idx].mean() > 0.30:
            start = int(window_size * 0.25)
            end = int(window_size * 0.75)
            if np.random.random() > 0.5:
                win[start:end, sensor_idx] = np.clip(
                    win[start:end, sensor_idx] * 1.8, 0.0, 1.0
                )
            else:
                win[start:end, sensor_idx] = win[start:end, sensor_idx] * 0.4

    elif "ENGINE_LOAD ()" in sensor_name:
        drift_factor = np.random.uniform(0.25, 0.60)
        win[:, sensor_idx] = win[:, sensor_idx] * drift_factor

    elif "SHORT_TERM_FUEL_TRIM_BANK_1 ()" in sensor_name:
        if win[:, sensor_idx].mean() > 0.3:
            start = int(window_size * 0.30)
            end = int(window_size * 0.70)
            stuck_value = np.random.uniform(0.7, 0.9)
            win[start:end, sensor_idx] = stuck_value

    elif "LONG_TERM_FUEL_TRIM_BANK_1 ()" in sensor_name:
        drift = np.random.uniform(0.25, 0.60)
        win[:, sensor_idx] = np.clip(win[:, sensor_idx] + drift, 0.0, 1.0)

    return affected_sensors


def inject_faults_with_sensor_labels(
    X_windows,
    y_windows,
    sensor_cols,
    fault_percentage=0.30,
    random_state=42,
    use_stratified=True,
):
    """
    Inject faults with STRATIFIED sensor distribution to ensure even coverage.

    Args:
        X_windows: (N, W, D) tensor of window data
        y_windows: (N, D) tensor of target values
        sensor_cols: List of sensor column names
        fault_percentage: Percentage of windows to inject faults into
        random_state: Random seed for reproducibility
        use_stratified: If True, ensures each sensor gets roughly equal faults

    Returns:
        X_faulty: (N, W, D) windows with injected faults
        y_windows: (N, D) unchanged target values
        sensor_labels: (N, D) binary matrix - 1 if sensor i is faulty in window j
        window_labels: (N,) binary - 1 if any fault exists in window
        fault_types: list of N strings - fault type name per window ("normal" if clean)
    """
    np.random.seed(random_state)

    N, W, D = X_windows.shape
    n_fault = max(1, int(N * fault_percentage))

    X_faulty = X_windows.clone()
    sensor_labels = torch.zeros(N, D, dtype=torch.float32)
    window_labels = torch.zeros(N, dtype=torch.long)
    fault_types = ["normal"] * N

    pid_idx = {name: i for i, name in enumerate(sensor_cols)}

    if use_stratified:
        min_faults_per_sensor = n_fault // D
        extra_faults = n_fault % D

        sensor_fault_list = []
        for sensor_idx in range(D):
            count = min_faults_per_sensor
            if sensor_idx < extra_faults:
                count += 1
            sensor_fault_list.extend([sensor_idx] * count)

        np.random.shuffle(sensor_fault_list)

        fault_indices = np.random.choice(N, n_fault, replace=False)

        for fault_idx, target_sensor_idx in zip(fault_indices, sensor_fault_list):
            win = X_faulty[fault_idx].numpy()
            sensor_name = sensor_cols[target_sensor_idx]

            affected_sensors = inject_sensor_specific_fault(
                win, target_sensor_idx, sensor_name, pid_idx, W
            )

            if len(affected_sensors) > 0:
                X_faulty[fault_idx] = torch.tensor(win, dtype=torch.float32)
                window_labels[fault_idx] = 1
                for sensor_i in affected_sensors:
                    sensor_labels[fault_idx, sensor_i] = 1.0
                fault_types[fault_idx] = FAULT_TYPE_MAP.get(sensor_name, "UNKNOWN_FAULT")

    else:
        fault_indices = np.random.choice(N, n_fault, replace=False)

        for idx in fault_indices:
            win = X_faulty[idx].numpy()

            fault_type = np.random.choice(
                LEGACY_FAULT_TYPES,
                p=[0.20, 0.20, 0.20, 0.20, 0.20],
            )

            affected_sensors = []

            if fault_type == "vss_dropout" and "VEHICLE_SPEED ()" in pid_idx:
                speed_i = pid_idx["VEHICLE_SPEED ()"]
                if win[:, speed_i].mean() > 0.15:
                    start = int(W * 0.30)
                    end = int(W * 0.70)
                    win[start:end, speed_i] = 0.0
                    win[start:end, speed_i] += np.random.uniform(0, 0.02, end - start)
                    affected_sensors.append(speed_i)

            elif (
                fault_type == "maf_scale_low"
                and "INTAKE_MANIFOLD_PRESSURE ()" in pid_idx
            ):
                map_i = pid_idx["INTAKE_MANIFOLD_PRESSURE ()"]
                scale_factor = np.random.uniform(0.75, 0.80)
                win[:, map_i] = win[:, map_i] * scale_factor
                affected_sensors.append(map_i)

                if "SHORT_TERM_FUEL_TRIM_BANK_1 ()" in pid_idx:
                    stft_i = pid_idx["SHORT_TERM_FUEL_TRIM_BANK_1 ()"]
                    win[:, stft_i] = np.clip(win[:, stft_i] + 0.15, 0.0, 1.0)
                    affected_sensors.append(stft_i)

            elif (
                fault_type == "coolant_dropout" and "COOLANT_TEMPERATURE ()" in pid_idx
            ):
                cool_i = pid_idx["COOLANT_TEMPERATURE ()"]
                if win[:, cool_i].mean() > 0.5:
                    n_dropouts = np.random.randint(2, 5)
                    for _ in range(n_dropouts):
                        drop_start = np.random.randint(0, W - 60)
                        drop_len = np.random.randint(30, 60)
                        win[drop_start : drop_start + drop_len, cool_i] = (
                            np.random.uniform(0.05, 0.15)
                        )
                    affected_sensors.append(cool_i)

            elif fault_type == "tps_stuck" and "THROTTLE ()" in pid_idx:
                thr_i = pid_idx["THROTTLE ()"]
                freeze_point = W // 2
                stuck_value = win[freeze_point, thr_i]
                if stuck_value > 0.15 and win[:freeze_point, thr_i].std() > 0.05:
                    win[freeze_point:, thr_i] = stuck_value
                    affected_sensors.append(thr_i)

            elif fault_type == "rpm_speed_decouple":
                if "ENGINE_RPM ()" in pid_idx and "VEHICLE_SPEED ()" in pid_idx:
                    speed_i = pid_idx["VEHICLE_SPEED ()"]
                    rpm_i = pid_idx["ENGINE_RPM ()"]
                    if win[:, speed_i].mean() > 0.20 and win[:, rpm_i].mean() > 0.30:
                        start = int(W * 0.25)
                        end = int(W * 0.75)
                        win[start:end, speed_i] = win[
                            start:end, speed_i
                        ] * np.random.uniform(0.3, 0.5)
                        affected_sensors.append(speed_i)

            if len(affected_sensors) > 0:
                X_faulty[idx] = torch.tensor(win, dtype=torch.float32)
                window_labels[idx] = 1
                for sensor_i in affected_sensors:
                    sensor_labels[idx, sensor_i] = 1.0
                fault_types[idx] = fault_type.upper()

    return X_faulty, y_windows, sensor_labels, window_labels, fault_types
