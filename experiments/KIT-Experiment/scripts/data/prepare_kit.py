#!/usr/bin/env python3
"""
Preprocess KIT OBD-II dataset for the GARAGE pipeline.

Reads CSVs from data/KIT/dataset/trimmed/, maps columns to GARAGE sensor names,
downsamples to 1Hz, and saves to data/formatted/.
"""

import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent.parent
GARAGE_ROOT = EXPERIMENT_ROOT.parent.parent
INPUT_DIR = GARAGE_ROOT / "data" / "KIT" / "dataset" / "trimmed"
OUTPUT_DIR = EXPERIMENT_ROOT / "data" / "formatted"

KIT_TO_GARAGE = {
    "Engine RPM [RPM]": "ENGINE_RPM ()",
    "Vehicle Speed Sensor [km/h]": "VEHICLE_SPEED ()",
    "Absolute Throttle Position [%]": "THROTTLE ()",
    "Air Flow Rate from Mass Flow Sensor [g/s]": "ENGINE_LOAD ()",
    "Intake Manifold Absolute Pressure [kPa]": "INTAKE_MANIFOLD_PRESSURE ()",
    "Ambient Air Temperature [°C]": "COOLANT_TEMPERATURE ()",
    "Accelerator Pedal Position D [%]": "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
    "Accelerator Pedal Position E [%]": "LONG_TERM_FUEL_TRIM_BANK_1 ()",
}

OUTPUT_SENSOR_ORDER = [
    "ENGINE_RPM ()",
    "VEHICLE_SPEED ()",
    "THROTTLE ()",
    "ENGINE_LOAD ()",
    "COOLANT_TEMPERATURE ()",
    "INTAKE_MANIFOLD_PRESSURE ()",
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
    "LONG_TERM_FUEL_TRIM_BANK_1 ()",
]
TIME_COL = "ENGINE_RUN_TINE ()"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace("Â°", "°") for c in df.columns]
    return df


def parse_time_to_elapsed_seconds(series: pd.Series) -> pd.Series:
    times = pd.to_datetime(series, format="%H:%M:%S.%f", errors="coerce")
    if times.isna().all():
        times = pd.to_datetime(series, format="%H:%M:%S", errors="coerce")
    start = times.iloc[0]
    elapsed = (times - start).dt.total_seconds()
    return elapsed


def process_file(csv_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = _normalize_columns(df)
    if df.empty or "Time" not in df.columns:
        return None

    time_elapsed = parse_time_to_elapsed_seconds(df["Time"])
    out_cols = {}
    for kit_name, garage_name in KIT_TO_GARAGE.items():
        if kit_name not in df.columns:
            return None
        out_cols[garage_name] = pd.to_numeric(df[kit_name], errors="coerce")

    result = pd.DataFrame({TIME_COL: time_elapsed, **out_cols})
    result["_sec"] = result[TIME_COL].round(0).astype(int)
    grouped = result.groupby("_sec", as_index=False)[OUTPUT_SENSOR_ORDER].mean()
    grouped = grouped.rename(columns={"_sec": TIME_COL})
    grouped[TIME_COL] = grouped[TIME_COL].astype(float)
    return grouped


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}")
        return

    summary_rows = []
    for csv_path in csv_files:
        df_raw = pd.read_csv(csv_path, encoding="utf-8")
        processed = process_file(csv_path)
        if processed is None:
            continue

        n_orig, n_down = len(df_raw), len(processed)
        duration_s = int(processed[TIME_COL].iloc[-1] - processed[TIME_COL].iloc[0]) if n_down > 1 else 0
        processed["drive_id"] = csv_path.name
        out_cols = [TIME_COL] + OUTPUT_SENSOR_ORDER + ["drive_id"]
        processed[out_cols].to_csv(OUTPUT_DIR / csv_path.name, index=False)
        summary_rows.append((csv_path.name, n_orig, n_down, duration_s))

    print(f"\n{'File':<50} {'Original rows':>14} {'After downsample':>16} {'Duration (s)':>12}")
    print("-" * 96)
    for fname, n_orig, n_down, dur in summary_rows:
        print(f"{fname:<50} {n_orig:>14} {n_down:>16} {dur:>12}")
    print("-" * 96)
    print(f"Total files processed: {len(summary_rows)}")


if __name__ == "__main__":
    main()
