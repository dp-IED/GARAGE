## Cursor Cloud specific instructions

### Project overview

GARAGE is a Graph Neural Network anomaly detection system for automotive OBD-II sensor data. It has two main components:

- **Python ML pipeline** (root): data preparation, Stage 1/2 training, knowledge graph creation, LLM evaluation
- **Eval Dashboard** (`web/`): React/Vite/TypeScript app for visualizing evaluation results

### Running the ML pipeline

Before training, generate the shared dataset from raw CSVs (only needed once unless data changes):

```
python data/create_shared_dataset.py --raw-data-path data/carOBD/obdiidata --output-dir data/shared_dataset
```

Stage 1 quick test: `python training/train_stage1.py --epochs 2 --batch_size 16 --cpu_only --max_batches_per_epoch 3`

See `README.md` for full training arguments.

### Running the eval dashboard

```
cd web && npm run dev
```

The dashboard runs on port 5173 by default.

### Key gotchas

- `data/fault_injection.py` is a required module referenced by `training/train_stage2.py` and `data/create_shared_dataset.py`. If missing, the dataset creation and Stage 2 training will fail with `ModuleNotFoundError`.
- `web/index.html` is the Vite entry point. If missing, `npm run build` and `npm run dev` will fail.
- No dedicated ESLint or Python linter configs exist. Use `npx tsc --noEmit` in `web/` for TypeScript checks and `python -m py_compile <file>` for Python syntax verification.
- The project uses CPU fallback automatically when CUDA is unavailable. MPS (Apple Silicon) is explicitly unsupported for PyTorch Geometric.
- LLM evaluation scripts (`llm/`) require an external LM Studio server at `localhost:1234` — these are optional for core pipeline testing.
