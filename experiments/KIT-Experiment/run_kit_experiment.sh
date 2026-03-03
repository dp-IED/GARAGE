#!/bin/bash
set -e
cd "$(dirname "$0")/../.."  # cd to GARAGE-Final root

EXPERIMENT_DIR="experiments/KIT-Experiment"
DATA_DIR="$EXPERIMENT_DIR/data/formatted"
DATASET_DIR="$EXPERIMENT_DIR/data/dataset"
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"
LOG_DIR="$EXPERIMENT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "[1/4] Preparing KIT data..."
python -u "$EXPERIMENT_DIR/scripts/data/prepare_kit.py"

echo "[2/4] Building shared dataset..."
python -u "$EXPERIMENT_DIR/scripts/data/create_kit_dataset.py"

echo "[3/4] Training Stage 1 (Graph Structure Learning)..."
python -u "$EXPERIMENT_DIR/scripts/training/train_stage1.py" \
  --data_path         "$DATASET_DIR" \
  --checkpoint_dir    "$CHECKPOINT_DIR" \
  --epochs            75 \
  --scheduler_patience  15 \
  --scheduler_factor    0.5 \
  --contrastive_temperature 0.1 \
  --lambda_contrast   0.7 \
  --lambda_forecast   1.0 \
  --early_stopping_patience 25 \
  --seed              42

echo "✓ Stage 1 complete. Checkpoint: $CHECKPOINT_DIR/stage1_best_forecast.pt"

echo "[4/4] Training Stage 2..."
python -u "$EXPERIMENT_DIR/scripts/training/train_stage2_clean.py" \
  --stage1_checkpoint "$CHECKPOINT_DIR/stage1_best_forecast.pt" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --data_path "$DATASET_DIR" \
  --epochs 50 \
  --seed 42

echo "✓ Stage 2 complete. Checkpoint: $CHECKPOINT_DIR/stage2_clean_best.pt"
