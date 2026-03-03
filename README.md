# GARAGE: Graph Attention-based Anomaly Detection for Automotive Sensors

A Graph Neural Network (GNN) based anomaly detection system for automotive OBD-II sensor data using Graph Deviation Networks (GDN).

## Project Structure

```
GARAGE-Final/
├── models/
│   ├── __init__.py
│   └── gdn_model.py          # Core GDN model architecture & loss functions
├── training/
│   └── train_stage1.py       # Stage 1: Self-supervised graph structure learning
├── data/
│   └── carOBD/
│       └── obdiidata/        # OBD-II sensor data (CSV files)
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure data files are in `data/carOBD/obdiidata/` directory

## Usage

### Stage 1: Self-Supervised Graph Structure Learning

Train the model with forecasting loss (no labels needed):

```bash
python training/train_stage1.py \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-3 \
    --device cpu
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 75)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--device`: Device to use: `cpu`, `cuda`, or `mps` (default: auto-detect)
- `--cpu_only`: Force CPU usage
- `--checkpoint_dir`: Directory to save checkpoints (default: `checkpoints`)
- `--max_batches_per_epoch`: Limit number of batches per epoch for quick testing (default: None)

**Example with a few epochs for testing:**
```bash
python training/train_stage1.py --epochs 3 --batch_size 16
```

## Model Architecture

- **GDN**: Graph Deviation Network for anomaly detection
  - Single-layer unidirectional GRU for temporal encoding
  - Graph Attention Network (GAT) for spatial relationships
  - LayerNorm and residual connections for training stability
  - Graph caching for efficiency

## Data Format

The training script expects CSV files in `data/carOBD/obdiidata/` with the following sensor columns:
- ENGINE_RPM
- VEHICLE_SPEED
- THROTTLE
- ENGINE_LOAD
- COOLANT_TEMPERATURE
- INTAKE_MANIFOLD_PRESSURE
- SHORT_TERM_FUEL_TRIM_BANK_1
- LONG_TERM_FUEL_TRIM_BANK_1

Each CSV file should have a `drive_id` column (automatically added from filename).

## Training Process

Stage 1 training uses:
- **Forecasting Loss**: Predicts future sensor values (multi-horizon: t+1, t+5, t+10)
- **Reconstruction Loss**: Reconstructs full window from embeddings
- **Contrastive Loss**: Embeddings from same drive should be similar

This self-supervised approach learns meaningful sensor embeddings and graph structure without requiring labeled anomaly data.
