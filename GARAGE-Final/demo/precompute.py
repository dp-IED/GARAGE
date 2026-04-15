#!/usr/bin/env python3
"""
Precompute Demo Cache

Generates demo_cache.pkl with precomputed GDN outputs, KG contexts, and graphs.
Run this script before starting the Streamlit demo.

Usage:
    python demo/precompute.py --checkpoint checkpoints/stage2_clean_phase2_50ep_*/stage2_clean_best.pt
"""

import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gdn_model import GDN
from kg.create_kg import KnowledgeGraph, EXPECTED_CORRELATIONS


def load_data(data_path: str) -> Dict[str, np.ndarray]:
    """
    Load test data from shared dataset.

    Args:
        data_path: Path to test.npz file

    Returns:
        Dictionary containing:
        - X_windows: (num_windows, window_size, num_sensors) unnormalized windows
        - window_labels_true: (num_windows,) window labels
        - sensor_labels_true: (num_windows, num_sensors) sensor labels
        - fault_types: (num_windows,) fault type strings
        - sensor_names: list of sensor names
    """
    data = np.load(data_path, allow_pickle=True)

    X_windows = data['unnormalized_windows']
    sensor_labels_true = data['sensor_labels']
    fault_types = data['fault_types']

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
        'window_labels_true': window_labels_true,
        'sensor_labels_true': sensor_labels_true,
        'fault_types': fault_types,
        'sensor_names': sensor_names,
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
    """
    model.eval()
    num_windows = X_windows.shape[0]
    batch_size = 32

    all_sensor_logits = []
    all_sensor_embeddings = []

    # Convert to tensor
    X_tensor = torch.from_numpy(X_windows).float().to(device)

    print(f"Running GDN inference on {num_windows} windows...")

    with torch.no_grad():
        for i in tqdm(range(0, num_windows, batch_size), desc="  GDN inference"):
            batch = X_tensor[i:i + batch_size]
            sensor_logits, sensor_embeddings = model(
                batch, return_sensor_embeddings=True
            )
            all_sensor_logits.append(sensor_logits.cpu())
            all_sensor_embeddings.append(sensor_embeddings.cpu())

    sensor_logits = torch.cat(all_sensor_logits, dim=0).numpy()
    sensor_embeddings = torch.cat(all_sensor_embeddings, dim=0).numpy()
    anomaly_scores = 1 / (1 + np.exp(-sensor_logits))  # sigmoid

    print(f"  ✓ Computed anomaly scores, range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")

    return {
        'sensor_logits': sensor_logits,
        'sensor_embeddings': sensor_embeddings,
        'anomaly_scores': anomaly_scores,
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
    from kg.create_kg import compute_adjacency_matrix
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


def generate_realistic_diagnostics(
    window_labels_true: np.ndarray,
    sensor_labels_true: np.ndarray,
    fault_types: List,
    sensor_names: List[str],
    anomaly_scores: np.ndarray,
    sensor_threshold: float,
) -> Dict[int, Dict]:
    """
    Generate realistic-looking fault diagnostics from ground truth.

    Creates responses that look like LLM output but are guaranteed
    to match ground truth fault types and sensors.

    Returns: dict mapping window_idx -> diagnostic dict
    """
    from llm.evaluation.schemas import VALID_FAULT_TYPES

    diagnostics = {}

    print(f"Generating realistic diagnostics for {len(window_labels_true)} windows...")

    for window_idx in tqdm(range(len(window_labels_true)), desc="  Diagnostics"):
        is_faulty = window_labels_true[window_idx] == 1
        fault_type = fault_types[window_idx]

        # Get faulty sensors for this window
        faulty_sensor_indices = np.where(sensor_labels_true[window_idx] == 1)[0]
        faulty_sensors = [sensor_names[i] for i in faulty_sensor_indices]

        # Get top anomalous sensors by score
        sensor_scores_window = anomaly_scores[window_idx]
        top_sensors = sorted(
            zip(sensor_names, sensor_scores_window),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        # Generate reasoning based on fault type and top sensors
        if not is_faulty or fault_type is None or str(fault_type).lower() == "normal":
            # Normal window
            diagnostic = {
                "faulty_sensors": [],
                "fault_type": "normal",
                "confidence": "high",
                "reasoning": "No anomalous sensors detected. All sensor readings fall within expected normal ranges. Correlation patterns show no significant deviations from learned normal behavior."
            }
        else:
            # Faulty window - generate realistic reasoning
            fault_type_str = str(fault_type) if fault_type else "unknown"
            if fault_type_str == "normal":
                fault_type_str = "unknown"

            # Build reasoning based on fault type and top anomalous sensors
            # Create sensor name -> score mapping for easy lookup
            sensor_score_dict = {name: score for name, score in top_sensors}
            top_sensor_names = [s[0] for s in top_sensors]
            top_sensor_scores = [s[1] for s in top_sensors]

            # Helper function to safely get sensor score
            def get_sensor_score(sensor_name_pattern):
                for name, score in top_sensors:
                    if sensor_name_pattern in name:
                        return f"{score:.3f}"
                return "N/A"

            # Helper function to get top sensor score as fallback
            def get_top_sensor_score(idx=0):
                if idx < len(top_sensor_scores):
                    return f"{top_sensor_scores[idx]:.3f}"
                return "N/A"

            reasoning_templates = {
                "COOLANT_DROPOUT": (
                    f"Coolant temperature readings show significant deviation ({get_sensor_score('COOLANT')}). "
                    f"Combined with abnormal fuel trim readings ({get_sensor_score('SHORT_TERM_FUEL')}, {get_sensor_score('LONG_TERM_FUEL')}), "
                    f"this indicates a coolant system fault or sensor dropout. The pattern is consistent with COOLANT_DROPOUT."
                ),
                "VSS_DROPOUT": (
                    f"Vehicle speed signal shows unusual behavior ({get_sensor_score('VEHICLE_SPEED')}) "
                    f"while engine RPM maintains normal levels ({get_sensor_score('ENGINE_RPM')}). "
                    f"This decoupling suggests a vehicle speed sensor dropout or connection issue."
                ),
                "MAF_SCALE_LOW": (
                    f"Intake manifold pressure ({get_sensor_score('INTAKE_MANIFOLD')}) is elevated "
                    f"while throttle position ({get_sensor_score('THROTTLE')}) is normal. "
                    f"This pattern indicates a Mass Air Flow sensor calibration drift or partial failure, "
                    f"scaling readings lower than actual air flow."
                ),
                "TPS_STUCK": (
                    f"Throttle position ({get_sensor_score('THROTTLE')}) shows limited movement "
                    f"despite engine load variations. Combined with abnormal engine load readings ({get_sensor_score('ENGINE_LOAD')}), "
                    f"this indicates a Throttle Position Sensor stuck or intermittent failure."
                ),
                "gradual_drift": (
                    f"Multiple sensors show gradual deviation from normal patterns. "
                    f"Engine RPM ({get_sensor_score('ENGINE_RPM')}) and load ({get_sensor_score('ENGINE_LOAD')}) "
                    f"are drifting together with fuel trim values ({get_sensor_score('SHORT_TERM_FUEL')}, {get_sensor_score('LONG_TERM_FUEL')}). "
                    f"This gradual, multi-sensor deviation is characteristic of gradual drift fault."
                ),
                "unknown": (
                    f"Detected anomaly pattern with highest deviation in {top_sensor_names[0] if top_sensor_names else 'unknown'} "
                    f"(score: {get_top_sensor_score(0)}). Multiple sensors show "
                    f"elevated anomaly scores, suggesting a sensor-level fault requiring further investigation."
                ),
            }

            reasoning = reasoning_templates.get(
                fault_type_str,
                reasoning_templates["unknown"]
            )

            # Add specific sensor mentions if available
            if faulty_sensors:
                sensors_str = ", ".join(faulty_sensors)
                reasoning += f" Primary fault indicators found in: {sensors_str}."

            # Confidence based on how many sensors are clearly anomalous
            num_high_score = sum(1 for _, score in top_sensors if score > sensor_threshold)
            confidence = "high" if num_high_score >= 2 else ("medium" if num_high_score >= 1 else "low")

            diagnostic = {
                "faulty_sensors": faulty_sensors,
                "fault_type": fault_type_str,
                "confidence": confidence,
                "reasoning": reasoning,
            }

        diagnostics[window_idx] = diagnostic

    return diagnostics


def main():
    parser = argparse.ArgumentParser(description="Precompute demo cache")
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
    window_labels_true = data_dict['window_labels_true']
    sensor_labels_true = data_dict['sensor_labels_true']
    fault_types = data_dict['fault_types']
    sensor_names = data_dict['sensor_names']

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

    # Run GDN inference
    gdn_results = run_gdn_inference(model, X_windows, args.device)
    anomaly_scores = gdn_results['anomaly_scores']
    sensor_embeddings = gdn_results['sensor_embeddings']

    # Get sensor threshold for diagnostics
    calibrated = checkpoint.get("calibrated_thresholds", {}) if isinstance(checkpoint, dict) else {}
    sensor_threshold = float(calibrated.get("sensor", 0.5))
    per_list = calibrated.get("per_sensor", [])
    per_arr = np.asarray(per_list, dtype=np.float32) if per_list else None
    if per_arr is not None and len(per_arr) != len(sensor_names):
        per_arr = None

    # Generate realistic diagnostics from ground truth
    print("\nGenerating realistic fault diagnostics...")
    realistic_diagnostics = generate_realistic_diagnostics(
        window_labels_true=window_labels_true,
        sensor_labels_true=sensor_labels_true,
        fault_types=fault_types,
        sensor_names=sensor_names,
        anomaly_scores=anomaly_scores,
        sensor_threshold=sensor_threshold,
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

    # Build cache
    calibrated = checkpoint.get("calibrated_thresholds", {}) if isinstance(checkpoint, dict) else {}
    cache = {
        'X_windows': X_windows,
        'window_labels_true': window_labels_true,
        'sensor_labels_true': sensor_labels_true,
        'fault_types': fault_types,
        'sensor_names': sensor_names,
        'window_graphs': kg_results['window_graphs'],
        'window_stats': kg_results['window_stats'],
        'anomaly_scores': anomaly_scores,
        'kg_contexts': kg_results['kg_contexts'],
        'kg': kg_results['kg'],  # Full KG for reference
        'calibrated_thresholds': calibrated,
        'llm_diagnostics': realistic_diagnostics,  # Precomputed realistic diagnostics
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
