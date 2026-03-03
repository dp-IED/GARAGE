# KIT OBD-II Experiment

Self-contained experiment running the GARAGE pipeline on the KIT OBD-II dataset (Seat Leon drives).

## Column Mappings

KIT columns are mapped to GARAGE's expected 8-sensor schema. Some mappings use proxy sensors where KIT lacks the exact equivalent:

| KIT Column | GARAGE Column | Notes |
|------------|---------------|-------|
| Engine RPM [RPM] | ENGINE_RPM () | Direct |
| Vehicle Speed Sensor [km/h] | VEHICLE_SPEED () | Direct |
| Absolute Throttle Position [%] | THROTTLE () | Direct |
| Air Flow Rate from Mass Flow Sensor [g/s] | ENGINE_LOAD () | Proxy (mass flow → load) |
| Intake Manifold Absolute Pressure [kPa] | INTAKE_MANIFOLD_PRESSURE () | Direct |
| Ambient Air Temperature [°C] | COOLANT_TEMPERATURE () | Proxy (ambient → coolant) |
| Accelerator Pedal Position D [%] | SHORT_TERM_FUEL_TRIM_BANK_1 () | Proxy (pedal D) |
| Accelerator Pedal Position E [%] | LONG_TERM_FUEL_TRIM_BANK_1 () | Proxy (pedal E) |

## Pipeline Steps

Run in order (from GARAGE-Final root):

1. **Prepare** – Preprocess KIT CSVs (column mapping, 1Hz downsampling)
   ```bash
   python experiments/KIT-Experiment/scripts/data/prepare_kit.py
   ```

2. **Dataset** – Build train/val/test .npz from formatted data
   ```bash
   python experiments/KIT-Experiment/scripts/data/create_kit_dataset.py
   ```

3. **Stage 1** – Self-supervised graph structure learning
   ```bash
   python experiments/KIT-Experiment/scripts/training/train_stage1.py \
     --data_path experiments/KIT-Experiment/data/formatted \
     --checkpoint_dir experiments/KIT-Experiment/checkpoints ...
   ```

4. **Stage 2** – Supervised center loss training
   ```bash
   python experiments/KIT-Experiment/scripts/training/train_stage2_clean.py \
     --stage1_checkpoint experiments/KIT-Experiment/checkpoints/stage1_best_forecast.pt ...
   ```

Or run the full pipeline:
```bash
cd GARAGE-Final
bash experiments/KIT-Experiment/run_kit_experiment.sh
```

## Outputs

- `data/formatted/` – Preprocessed CSVs (1Hz, GARAGE column names)
- `data/dataset/` – Shared train/val/test splits (when created)
- `checkpoints/` – Stage 1 and Stage 2 model checkpoints
- `logs/` – Training and evaluation logs
