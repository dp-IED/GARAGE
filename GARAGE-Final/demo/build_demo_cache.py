#!/usr/bin/env python3
"""
Build the Streamlit demo cache (GDN + KG only; LLM diagnostics are always live).

Writes ``demo_cache.pkl`` with GDN outputs, per-window graphs/stats, ``kg_contexts``,
labels, and ``drive_ids``. Does **not** store any LLM output.

Run once before ``streamlit run demo/demo_app.py`` (or when data/checkpoint changes).

Usage:
    python demo/build_demo_cache.py --checkpoint checkpoints/.../stage2_clean_best.pt
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gdn_model import GDN
from kg.create_kg import KnowledgeGraph, compute_adjacency_matrix


def _best_f1_threshold_np(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold on score in [0,1] that maximizes F1 vs binary labels (same idea as train_stage2_clean)."""
    scores_np = np.asarray(scores, dtype=np.float64).ravel()
    labels_np = np.asarray(labels, dtype=np.int64).ravel()
    if scores_np.size == 0 or np.unique(labels_np).size <= 1:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(labels_np, scores_np)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    return float(np.clip(thresholds[min(best_idx, len(thresholds) - 1)], 0.0, 1.0))


def load_data(data_path: str) -> Dict[str, Any]:
    """
    Load test data from shared dataset.

    Args:
        data_path: Path to test.npz file

    Returns:
        Dictionary containing:
        - X_windows: (num_windows, window_size, num_sensors) unnormalized windows (plots / KG stats)
        - X_windows_model: (num_windows, window_size, num_sensors) inputs for GDN (must match training)
        - window_labels_true: (num_windows,) window labels
        - sensor_labels_true: (num_windows, num_sensors) sensor labels
        - fault_types: (num_windows,) fault type strings
        - sensor_names: list of sensor names
    """
    data = np.load(data_path, allow_pickle=True)

    X_windows = data['unnormalized_windows']
    sensor_labels_true = data['sensor_labels']
    fault_types = data['fault_types']
    drive_ids = None
    if 'drive_ids' in data.files:
        drive_ids = np.asarray(data['drive_ids'])
    else:
        drive_ids = np.array(['all_windows'] * len(X_windows), dtype=object)

    # Derive window_labels from fault_types: None or "normal" string = normal (0), other strings = faulty (1)
    window_labels_true = np.array([
        0 if ft is None or (isinstance(ft, str) and ft.lower() == "normal") else 1
        for ft in fault_types
    ], dtype=np.int64)

    # Load sensor names from metadata
    sensor_names = [
        'ENGINE_RPM ()',
        'VEHICLE_SPEED ()',
        'THROTTLE ()',
        'ENGINE_LOAD ()',
        'COOLANT_TEMPERATURE ()',
        'INTAKE_MANIFOLD_PRESSURE ()',
        'SHORT_TERM_FUEL_TRIM_BANK_1 ()',
        'LONG_TERM_FUEL_TRIM_BANK_1 ()'
    ]

    if X_windows.shape[2] != len(sensor_names):
        raise ValueError(
            f"Sensor axis mismatch: unnormalized_windows has {X_windows.shape[2]} channels "
            f"but sensor_names has {len(sensor_names)} entries."
        )

    if "normalized_windows" in data.files:
        X_model = np.asarray(data["normalized_windows"], dtype=np.float32)
        if X_model.shape != X_windows.shape:
            raise ValueError(
                f"normalized_windows shape {X_model.shape} != unnormalized_windows {X_windows.shape}"
            )
        print("  - GDN inference will use normalized_windows (same scale as training).")
    else:
        X_model = np.asarray(X_windows, dtype=np.float32)
        print(
            "  ⚠ No normalized_windows in .npz — running GDN on unnormalized inputs. "
            "Scores often saturate (all windows red). Regenerate the dataset with create_shared_dataset.py "
            "or add normalized_windows to the archive."
        )

    print(f"Loaded {len(X_windows)} windows from {data_path}")
    print(f"  - Window size: {X_windows.shape[1]}, Sensors: {X_windows.shape[2]}")
    print(f"  - Normal windows: {(window_labels_true == 0).sum()}, Faulty windows: {(window_labels_true == 1).sum()}")

    # Count fault types
    unique_faults, counts = np.unique(fault_types[window_labels_true == 1], return_counts=True)
    print("  - Fault types breakdown:")
    for fault, count in zip(unique_faults, counts):
        print(f"      {fault}: {count}")

    return {
        'X_windows': X_windows,
        'X_windows_model': X_model,
        'window_labels_true': window_labels_true,
        'sensor_labels_true': sensor_labels_true,
        'fault_types': fault_types,
        'sensor_names': sensor_names,
        'drive_ids': drive_ids,
    }


def run_gdn_inference(model: GDN, X_windows: np.ndarray, device: str = 'cpu') -> Dict[str, np.ndarray]:
    """
    Run GDN inference on all windows.

    Args:
        model: Trained GDN model
        X_windows: (num_windows, window_size, num_sensors) input windows
        device: Device to run on

    Returns:
        Dictionary containing:
        - sensor_logits: (num_windows, num_sensors) sensor anomaly logits
        - sensor_embeddings: (num_windows, num_sensors, hidden_dim) sensor embeddings
        - anomaly_scores: (num_windows, num_sensors) sigmoid anomaly scores
        - window_scores: (num_windows,) sigmoid(global window logits), same as training eval
    """
    model.eval()
    num_windows = X_windows.shape[0]
    batch_size = 32

    all_sensor_logits = []
    all_global_logits = []
    all_sensor_embeddings = []

    # Convert to tensor
    X_tensor = torch.from_numpy(X_windows).float().to(device)

    print(f"Running GDN inference on {num_windows} windows...")

    with torch.no_grad():
        for i in tqdm(range(0, num_windows, batch_size), desc="  GDN inference"):
            batch = X_tensor[i:i + batch_size]
            sensor_logits, global_logits, sensor_embeddings = model(
                batch, return_global=True, return_sensor_embeddings=True
            )
            all_sensor_logits.append(sensor_logits.cpu())
            all_global_logits.append(global_logits.cpu())
            all_sensor_embeddings.append(sensor_embeddings.cpu())

    sensor_logits = torch.cat(all_sensor_logits, dim=0).numpy()
    global_logits = torch.cat(all_global_logits, dim=0).numpy()
    sensor_embeddings = torch.cat(all_sensor_embeddings, dim=0).numpy()
    anomaly_scores = 1 / (1 + np.exp(-sensor_logits))  # sigmoid
    window_scores = 1 / (1 + np.exp(-global_logits))

    print(f"  ✓ Computed anomaly scores, range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")
    print(f"  ✓ Window scores (global head), range: [{window_scores.min():.3f}, {window_scores.max():.3f}]")

    return {
        'sensor_logits': sensor_logits,
        'sensor_embeddings': sensor_embeddings,
        'anomaly_scores': anomaly_scores,
        'window_scores': window_scores.astype(np.float32),
    }


def build_knowledge_graph(
    sensor_names: List[str],
    sensor_embeddings: np.ndarray,
    anomaly_scores: np.ndarray,
    X_windows: np.ndarray,
    sensor_labels_true: np.ndarray,
    window_labels_true: np.ndarray,
    calibrated_sensor_threshold: float = 0.5,
    calibrated_per_sensor_thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Build knowledge graph for all windows.

    Args:
        sensor_names: List of sensor names
        sensor_embeddings: (num_windows, num_sensors, hidden_dim) sensor embeddings
        anomaly_scores: (num_windows, num_sensors) anomaly scores
        X_windows: (num_windows, window_size, num_sensors) unnormalized windows
        sensor_labels_true: (num_windows, num_sensors) ground truth sensor labels
        window_labels_true: (num_windows,) ground truth window labels

    Returns:
        Dictionary containing:
        - kg: KnowledgeGraph instance
        - window_graphs: dict of window_idx -> NetworkX graph
        - window_stats: dict of window_idx -> sensor stats
        - kg_contexts: dict of window_idx -> KG context dict
    """
    num_windows = X_windows.shape[0]

    # Compute adjacency matrix from average sensor embeddings
    avg_sensor_embeddings = sensor_embeddings.mean(axis=0)  # (num_sensors, hidden_dim)
    adjacency_matrix = compute_adjacency_matrix(avg_sensor_embeddings)

    # Create KnowledgeGraph instance
    kg = KnowledgeGraph(
        sensor_names=sensor_names,
        sensor_embeddings=avg_sensor_embeddings,
        adjacency_matrix=adjacency_matrix,
    )

    # Build KG for all windows
    print("Building knowledge graph for all windows...")
    kg.construct(
        X_windows=X_windows,
        gdn_predictions=anomaly_scores,
        X_windows_unnormalized=X_windows,
        sensor_labels_true=sensor_labels_true,
        window_labels_true=window_labels_true,
        calibrated_sensor_threshold=calibrated_sensor_threshold,
        calibrated_per_sensor_thresholds=calibrated_per_sensor_thresholds,
    )

    print(f"  ✓ Built KG: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")

    # Extract window graphs and stats
    window_graphs = kg.window_graphs
    window_stats = kg.window_stats

    # Generate KG contexts for LLM (first 100 windows for demo speed)
    print("Generating KG contexts for LLM...")
    kg_contexts = {}
    for idx in tqdm(range(num_windows), desc="  KG contexts"):
        kg_context = kg.get_window_kg(window_idx=idx, temporal_context_windows=1)
        kg_contexts[idx] = kg_context

    print(f"  ✓ Generated {len(kg_contexts)} KG contexts")

    return {
        'kg': kg,
        'window_graphs': window_graphs,
        'window_stats': window_stats,
        'kg_contexts': kg_contexts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build demo_cache.pkl (GDN + KG; no LLM output)."
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to GDN checkpoint (stage2_clean_best.pt)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/shared_dataset/test.npz',
        help='Path to test data .npz file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='demo/demo_cache.pkl',
        help='Output path for demo cache pickle'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run on'
    )
    args = parser.parse_args()

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dict = load_data(args.data)
    X_windows = data_dict['X_windows']
    X_windows_model = data_dict['X_windows_model']
    window_labels_true = data_dict['window_labels_true']
    sensor_labels_true = data_dict['sensor_labels_true']
    fault_types = data_dict['fault_types']
    sensor_names = data_dict['sensor_names']
    drive_ids = data_dict['drive_ids']

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)

    # Extract model config
    if isinstance(checkpoint, dict):
        sensor_names_ckpt = checkpoint.get('sensor_names', sensor_names)
        window_size = checkpoint.get('window_size', 300)
        embed_dim = checkpoint.get('embed_dim', 32)
        top_k = checkpoint.get('top_k', 5)
        hidden_dim = checkpoint.get('hidden_dim', 64)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        state_dict = checkpoint
        sensor_names_ckpt = sensor_names
        window_size = 300
        embed_dim = 32
        top_k = 5
        hidden_dim = 64

    # Initialize model
    model = GDN(
        num_nodes=len(sensor_names_ckpt),
        window_size=window_size,
        embed_dim=embed_dim,
        top_k=top_k,
        hidden_dim=hidden_dim,
    ).to(args.device)

    # Handle PyG compatibility
    state_dict = dict(state_dict)
    model_expects_lin = "gat.lin.weight" in model.state_dict()
    model_expects_lin_src = "gat.lin_src.weight" in model.state_dict()
    ckpt_has_lin = "gat.lin.weight" in state_dict
    ckpt_has_lin_src = "gat.lin_src.weight" in state_dict

    if model_expects_lin_src and ckpt_has_lin:
        lin_weight = state_dict.pop("gat.lin.weight")
        state_dict["gat.lin_src.weight"] = lin_weight.clone()
        state_dict["gat.lin_dst.weight"] = lin_weight.clone()
    elif model_expects_lin and ckpt_has_lin_src:
        state_dict["gat.lin.weight"] = state_dict.pop("gat.lin_src.weight")
        state_dict.pop("gat.lin_dst.weight", None)

    model.load_state_dict(state_dict, strict=True)
    print("  ✓ Model loaded successfully")

    # Run GDN inference (training uses normalized [0,1] windows, not raw OBD units)
    gdn_results = run_gdn_inference(model, X_windows_model, args.device)
    anomaly_scores = gdn_results['anomaly_scores']
    window_scores = gdn_results['window_scores']
    sensor_embeddings = gdn_results['sensor_embeddings']
    if float(np.min(window_scores)) >= 0.999 and float(np.max(window_scores)) <= 1.0001:
        print(
            "  ⚠ Window scores are saturated at ~1.0 — check that X_windows_model matches training "
            "(use normalized_windows from the shared dataset .npz)."
        )

    # Thresholds: keep checkpoint sensor/per-sensor; refit window on *this* split so the demo matches
    # eval behaviour when global scores are shifted (logs: normals still ~1.0 vs ckpt window 0.52 → ~90% red).
    calibrated_src = (
        checkpoint.get("calibrated_thresholds", {}) if isinstance(checkpoint, dict) else {}
    )
    calibrated = dict(calibrated_src) if isinstance(calibrated_src, dict) else {}
    sensor_threshold = float(calibrated.get("sensor", 0.5))
    per_list = calibrated.get("per_sensor", [])
    per_arr = np.asarray(per_list, dtype=np.float32) if per_list else None
    if per_arr is not None and len(per_arr) != len(sensor_names):
        per_arr = None

    ckpt_window_thr = float(calibrated.get("window", 0.5))
    calibrated["window_checkpoint"] = ckpt_window_thr
    ws_flat = np.asarray(window_scores, dtype=np.float64).ravel()
    y_win = np.asarray(window_labels_true, dtype=np.int64)
    if ws_flat.std() > 1e-8 and np.unique(y_win).size > 1:
        calibrated["window"] = float(_best_f1_threshold_np(ws_flat, y_win))
        print(
            f"  ✓ Demo window threshold (PR–F1 on this split): {calibrated['window']:.6f} "
            f"(checkpoint: {ckpt_window_thr:.6f})"
        )
    else:
        calibrated["window"] = ckpt_window_thr
        print(
            "  ⚠ Using checkpoint window threshold only (degenerate window scores or single class)."
        )

    # Build knowledge graph
    kg_results = build_knowledge_graph(
        sensor_names=sensor_names,
        sensor_embeddings=sensor_embeddings,
        anomaly_scores=anomaly_scores,
        X_windows=X_windows,
        sensor_labels_true=sensor_labels_true,
        window_labels_true=window_labels_true,
        calibrated_sensor_threshold=sensor_threshold,
        calibrated_per_sensor_thresholds=per_arr,
    )

    # Build cache (omit full KG object — unused by Streamlit; reduces pickle size)
    cache = {
        'X_windows': X_windows,
        'window_labels_true': window_labels_true,
        'sensor_labels_true': sensor_labels_true,
        'fault_types': fault_types,
        'sensor_names': sensor_names,
        'window_graphs': kg_results['window_graphs'],
        'window_stats': kg_results['window_stats'],
        'anomaly_scores': anomaly_scores,
        'window_scores': window_scores,
        'kg_contexts': kg_results['kg_contexts'],
        'calibrated_thresholds': calibrated,
        'drive_ids': drive_ids,
    }

    # Save cache
    print(f"\nSaving cache to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(cache, f)

    print("  ✓ Cache saved successfully")
    print(f"\nDemo cache ready with {len(X_windows)} windows")
    print(f"Run: streamlit run demo/demo_app.py")


if __name__ == '__main__':
    main()
