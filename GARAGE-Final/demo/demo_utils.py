"""
Demo Utilities for Streamlit App

Shared styling and graph rendering functions for the GARAGE-Final demo.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network


def per_sensor_thr(
    name: str,
    sensor_threshold: float,
    per_sensor_thresholds: Optional[Dict[str, float]],
) -> float:
    if per_sensor_thresholds:
        return float(per_sensor_thresholds.get(name, sensor_threshold))
    return float(sensor_threshold)


def window_scores_top2_mean(anomaly_scores: np.ndarray) -> np.ndarray:
    """
    Fallback window-level score when the cache has no global head outputs.

    Matches training's fallback: mean of the top-2 per-sensor probabilities per window.
    """
    x = np.asarray(anomaly_scores, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("anomaly_scores must be (num_windows, num_sensors)")
    n_s = x.shape[1]
    if n_s < 2:
        return np.max(x, axis=1)
    part = np.partition(x, -2, axis=1)[:, -2:]
    return part.mean(axis=1)


def gdn_window_alert_mask(window_scores: np.ndarray, window_threshold: float) -> np.ndarray:
    """Per window: True iff GDN window score exceeds calibrated window threshold (training convention)."""
    return np.asarray(window_scores, dtype=np.float64) > float(window_threshold)


def gdn_flagged_sensor_names(
    scores_1d: np.ndarray,
    sensor_names: List[str],
    sensor_threshold: float,
    per_sensor_thresholds: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Sensors above threshold for one window, sorted by score descending."""
    flagged: List[Tuple[str, float]] = []
    for i, name in enumerate(sensor_names):
        s = float(scores_1d[i])
        if s > per_sensor_thr(name, sensor_threshold, per_sensor_thresholds):
            flagged.append((name, s))
    flagged.sort(key=lambda x: -x[1])
    return [n for n, _ in flagged]


def get_anomaly_color(score: float) -> str:
    """
    Return color string for anomaly score (0-1) using green->yellow->red gradient.

    Args:
        score: Anomaly score between 0.0 and 1.0

    Returns:
        RGB color string
    """
    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, score))

    if score < 0.5:
        # Green to yellow (0.0 -> 0.5)
        # Start: rgb(34, 197, 94) = #22c55e (green-500)
        # End: rgb(234, 179, 8) = #eab308 (yellow-500)
        t = score / 0.5
        r = int(34 + (234 - 34) * t)
        g = int(197 + (179 - 197) * t)
        b = int(94 + (8 - 94) * t)
    else:
        # Yellow to red (0.5 -> 1.0)
        # Start: rgb(234, 179, 8) = #eab308 (yellow-500)
        # End: rgb(239, 68, 68) = #ef4444 (red-500)
        t = (score - 0.5) / 0.5
        r = int(234 + (239 - 234) * t)
        g = int(179 + (68 - 179) * t)
        b = int(8 + (68 - 8) * t)

    return f"rgb({r},{g},{b})"


def get_anomaly_color_hex(score: float) -> str:
    """
    Return hex color string for anomaly score (0-1).

    Args:
        score: Anomaly score between 0.0 and 1.0

    Returns:
        Hex color string (e.g., "#22c55e")
    """
    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, score))

    if score < 0.5:
        # Green to yellow (0.0 -> 0.5)
        # Start: #22c55e (green-500)
        # End: #eab308 (yellow-500)
        t = score / 0.5
        r = int(34 + (234 - 34) * t)
        g = int(197 + (179 - 197) * t)
        b = int(94 + (8 - 94) * t)
    else:
        # Yellow to red (0.5 -> 1.0)
        # Start: #eab308 (yellow-500)
        # End: #ef4444 (red-500)
        t = (score - 0.5) / 0.5
        r = int(234 + (239 - 234) * t)
        g = int(179 + (68 - 179) * t)
        b = int(8 + (68 - 8) * t)

    return f"#{r:02x}{g:02x}{b:02x}"


def render_pyvis_graph(
    graph: nx.Graph,
    sensor_names: list,
    anomaly_scores: np.ndarray,
    window_status: int = 0,  # 0 = normal, 1 = faulty
    height: str = "400px",
    width: str = "100%",
    bgcolor: str = "#0e1117",
    font_color: str = "#e2e8f0"
) -> str:
    """
    Render NetworkX graph as HTML using PyVis with dark theme.

    Args:
        graph: NetworkX graph object
        sensor_names: List of sensor names (for node ordering)
        anomaly_scores: (num_sensors,) array of anomaly scores (0 = normal, 1 = faulty)
        window_status: Window status (0 = normal, 1 = faulty) - affects edge coloring
        height: Graph height
        width: Graph width
        bgcolor: Background color
        font_color: Font color

    Returns:
        HTML string for embedding in Streamlit
    """
    net = Network(height=height, width=width, bgcolor=bgcolor, font_color=font_color)

    # Add nodes explicitly to control attributes
    for i, sensor_name in enumerate(sensor_names):
        if i < len(anomaly_scores):
            score = float(anomaly_scores[i])
            color = get_anomaly_color(score)
            size = 15 + score * 30  # min 15, max 45
            border_width = 2 if score > 0.5 else 1

            # Check if node exists in graph and get its attributes
            if sensor_name in graph.nodes:
                node_attrs = dict(graph.nodes[sensor_name])
                # Convert numpy types to native Python types
                node_attrs_clean = {}
                for k, v in node_attrs.items():
                    if isinstance(v, (np.bool_, np.integer, np.floating)):
                        node_attrs_clean[k] = v.item() if hasattr(v, 'item') else bool(v) if isinstance(v, np.bool_) else float(v)
                    elif isinstance(v, np.ndarray):
                        node_attrs_clean[k] = v.tolist()
                    else:
                        node_attrs_clean[k] = v

                net.add_node(
                    sensor_name,
                    label=sensor_name,
                    color=color,
                    size=size,
                    borderWidth=border_width,
                    **node_attrs_clean
                )
            else:
                net.add_node(
                    sensor_name,
                    label=sensor_name,
                    color=color,
                    size=size,
                    borderWidth=border_width
                )

    # Add edges explicitly with controlled attributes
    for edge in graph.edges(data=True):
        src, dst, attrs = edge

        # Determine edge color based on window status
        if window_status == 1:  # Faulty window
            # Show edges between faulty sensors in red
            # Get node anomaly scores to determine if sensors are faulty
            src_idx = sensor_names.index(src) if src in sensor_names else -1
            dst_idx = sensor_names.index(dst) if dst in sensor_names else -1

            src_faulty = (src_idx >= 0 and src_idx < len(anomaly_scores) and anomaly_scores[src_idx] > 0.5)
            dst_faulty = (dst_idx >= 0 and dst_idx < len(anomaly_scores) and anomaly_scores[dst_idx] > 0.5)

            if src_faulty and dst_faulty:
                edge_color = "#ef4444"  # Red - both endpoints faulty
            elif src_faulty or dst_faulty:
                edge_color = "#f59e0b"  # Orange - one endpoint faulty
            else:
                edge_color = "#22c55e"  # Green - both normal
        else:  # Normal window
            # All edges green for normal windows
            edge_color = "#22c55e"

        edge_attrs = {
            'color': edge_color,
            'width': 2,
        }

        # Add other edge attributes if they exist (convert numpy types)
        for k, v in attrs.items():
            if k not in ['color', 'width']:
                if isinstance(v, (np.bool_, np.integer, np.floating)):
                    edge_attrs[k] = v.item() if hasattr(v, 'item') else bool(v) if isinstance(v, np.bool_) else float(v)
                elif isinstance(v, np.ndarray):
                    edge_attrs[k] = v.tolist()
                else:
                    edge_attrs[k] = v

        net.add_edge(src, dst, **edge_attrs)

    # Configure physics for better layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.09
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200
      }
    }
    """)

    return net.generate_html()


def apply_dark_theme():
    """
    Configure Plotly charts for dark theme.

    Returns:
        Dict with layout configuration for dark theme
    """
    return {
        "layout": {
            "paper_bgcolor": "#0e1117",
            "plot_bgcolor": "#0e1117",
            "font": {"color": "#e2e8f0"},
            "xaxis": {
                "gridcolor": "#1e293b",
                "linecolor": "#334155",
                "tickcolor": "#334155",
            },
            "yaxis": {
                "gridcolor": "#1e293b",
                "linecolor": "#334155",
                "tickcolor": "#334155",
            },
            "legend": {
                "bgcolor": "#1e293b",
                "bordercolor": "#334155",
                "font": {"color": "#e2e8f0"},
            },
            "margin": {"l": 50, "r": 20, "t": 30, "b": 40},
        }
    }


def build_single_window_sensor_figure(
    widx: int,
    X_windows: np.ndarray,
    sensor_names: List[str],
    anomaly_scores: np.ndarray,
    sensor_threshold: float,
    per_sensor_thresholds: Dict[str, float],
    use_per_sensor_thresholds: bool,
) -> go.Figure:
    """Single-window multi-sensor traces (actual units) for detail / diagnostics."""
    fig = go.Figure()
    for i, sname in enumerate(sensor_names):
        score = anomaly_scores[widx, i]
        color = get_anomaly_color(score)
        thr = (
            per_sensor_thresholds.get(sname, sensor_threshold)
            if use_per_sensor_thresholds
            else sensor_threshold
        )
        lw = 2.5 if score > thr else 1.0
        fig.add_trace(
            go.Scatter(
                y=X_windows[widx, :, i],
                mode="lines",
                line=dict(color=color, width=lw),
                name=sname,
            )
        )
    layout = apply_dark_theme()
    fig.update_layout(
        **layout["layout"],
        showlegend=True,
        height=420,
        hovermode="x unified",
        xaxis_title="Timestep",
        yaxis_title="Sensor value",
        title=f"Sensor traces — window {widx}",
    )
    return fig


def _normalize_fault_label(value: Any) -> str:
    """Map dataset / model fault labels to a single lowercase form for comparison."""
    if value is None:
        return "normal"
    s = str(value).strip()
    if not s or s.lower() in ("none", "normal", "nan"):
        return "normal"
    return s.lower().replace(" ", "_")


def check_fault_match(gt_fault: Any, pred_fault: Any) -> str:
    """
    Compare ground-truth fault label to the model's structured fault_type.

    Returns:
        "correct", "partial", or "none"
    """
    gt = _normalize_fault_label(gt_fault)
    pred = _normalize_fault_label(pred_fault)

    if gt == pred:
        return "correct"
    if gt == "normal" or pred == "normal":
        return "none"

    gt_parts = [p for p in gt.split("_") if len(p) > 2]
    pred_parts = [p for p in pred.split("_") if len(p) > 2]
    for a in gt_parts:
        if a in pred or a in pred_parts:
            return "partial"
    for b in pred_parts:
        if b in gt or b in gt_parts:
            return "partial"

    return "none"


def create_timeline_colormap(window_labels: np.ndarray) -> list:
    """
    Create color list for timeline visualization.

    Args:
        window_labels: (num_windows,) array of binary labels

    Returns:
        List of hex color strings
    """
    return [
        get_anomaly_color_hex(0.0) if label == 0 else get_anomaly_color_hex(1.0)
        for label in window_labels
    ]


def normalize_lm_base_url(url: str) -> str:
    """Ensure OpenAI-compatible root ends with /v1."""
    u = str(url).strip().rstrip("/")
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u


def get_lm_settings() -> Dict[str, str]:
    """
    Resolve LM Studio base URL and model name: env vars, then optional demo/lm_config.json.
    """
    base_url = normalize_lm_base_url(os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"))
    model = os.environ.get("LM_STUDIO_MODEL", "granite-4.0-h-micro-GGUF")
    cfg = Path(__file__).resolve().parent / "lm_config.json"
    if cfg.is_file():
        with open(cfg, encoding="utf-8") as f:
            data = json.load(f)
        if "base_url" in data:
            base_url = normalize_lm_base_url(str(data["base_url"]))
        model = str(data.get("model", model))
    return {"base_url": base_url, "model": model}


def build_drive_window_map(drive_ids: np.ndarray) -> Tuple[list, Dict[Any, list]]:
    """Return ordered unique drive ids and mapping drive_id -> list of global window indices."""
    seen = set()
    order: list = []
    for d in drive_ids:
        if d not in seen:
            seen.add(d)
            order.append(d)
    drive_to_idx: Dict[Any, list] = {d: [] for d in order}
    for i, d in enumerate(drive_ids):
        drive_to_idx[d].append(i)
    return order, drive_to_idx


def resolve_demo_cache_path(demo_dir: Path, cache_path: str = "demo/demo_cache.pkl") -> Path:
    """Prefer `cache_path` if it exists (e.g. cwd-relative); otherwise `demo_dir / demo_cache.pkl`."""
    p = Path(cache_path)
    if not p.is_file():
        p = demo_dir / "demo_cache.pkl"
    return p


DEMO_DRIVE_DROPDOWN_ORDER: Tuple[str, ...] = (
    "drive11.csv",
    "idle29.csv",
    "idle44.csv",
    "idle46.csv",
    "live31.csv",
    "long10.csv",
    "drive6.csv",
    "drive10.csv",
)


def expand_demo_cache(cache: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unpack a demo_cache.pkl dict into arrays, thresholds, drive maps, and GDN window flags.

    Used by the Streamlit home page and the Diagnostics multipage so both stay consistent.
    """
    X_windows = cache["X_windows"]
    window_labels_true = cache["window_labels_true"]
    sensor_labels_true = cache["sensor_labels_true"]
    fault_types = cache["fault_types"]
    sensor_names = cache["sensor_names"]
    window_graphs = cache["window_graphs"]
    window_stats = cache["window_stats"]
    anomaly_scores = cache["anomaly_scores"]
    kg_contexts = cache["kg_contexts"]
    calibrated_thresholds = cache.get("calibrated_thresholds", {})

    N = len(X_windows)
    num_sensors = len(sensor_names)
    drive_ids = cache.get("drive_ids")
    if drive_ids is None or len(drive_ids) != N:
        drive_ids = np.array(["all_windows"] * N, dtype=object)

    _ord, _map = build_drive_window_map(drive_ids)
    unique_drives = [str(d) for d in _ord]
    drive_to_indices = {str(k): list(v) for k, v in _map.items()}
    _ud_set = set(unique_drives)
    _first = [d for d in DEMO_DRIVE_DROPDOWN_ORDER if d in _ud_set]
    _seen = set(_first)
    _tail = [d for d in unique_drives if d not in _seen]
    unique_drives = _first + _tail
    synthetic_drives = len(unique_drives) == 1 and unique_drives[0] == "all_windows"

    sensor_threshold = float(calibrated_thresholds.get("sensor", 0.5))
    window_threshold = float(calibrated_thresholds.get("window", 0.5))
    per_sensor_array = calibrated_thresholds.get("per_sensor", [])
    per_sensor_thresholds: Dict[str, float] = {}
    if len(per_sensor_array) == len(sensor_names):
        per_sensor_thresholds = {
            sensor_names[i]: float(per_sensor_array[i]) for i in range(len(sensor_names))
        }
        use_per_sensor_thresholds = True
    else:
        use_per_sensor_thresholds = False

    gt_window_fault = window_labels_true == 1

    _window_scores = cache.get("window_scores")
    if _window_scores is not None:
        _window_scores = np.asarray(_window_scores, dtype=np.float64).reshape(-1)
    if _window_scores is None or _window_scores.shape[0] != N:
        window_scores = window_scores_top2_mean(anomaly_scores).astype(np.float64)
        window_score_fallback = True
    else:
        window_scores = _window_scores
        window_score_fallback = False

    gdn_window_flag = gdn_window_alert_mask(window_scores, window_threshold)

    return {
        "X_windows": X_windows,
        "N": N,
        "window_labels_true": window_labels_true,
        "sensor_labels_true": sensor_labels_true,
        "fault_types": fault_types,
        "sensor_names": sensor_names,
        "window_graphs": window_graphs,
        "window_stats": window_stats,
        "anomaly_scores": anomaly_scores,
        "kg_contexts": kg_contexts,
        "calibrated_thresholds": calibrated_thresholds,
        "drive_ids": drive_ids,
        "unique_drives": unique_drives,
        "drive_to_indices": drive_to_indices,
        "synthetic_drives": synthetic_drives,
        "sensor_threshold": sensor_threshold,
        "window_threshold": window_threshold,
        "per_sensor_thresholds": per_sensor_thresholds,
        "use_per_sensor_thresholds": use_per_sensor_thresholds,
        "gt_window_fault": gt_window_fault,
        "window_scores": window_scores,
        "window_score_fallback": window_score_fallback,
        "num_sensors": num_sensors,
        "gdn_window_flag": gdn_window_flag,
    }


def format_fault_type(fault_type: Optional[str]) -> str:
    """
    Format fault type for display.

    Args:
        fault_type: Raw fault type string

    Returns:
        Formatted fault type string
    """
    if not fault_type or fault_type == "None":
        return "Normal"

    # Convert underscores to spaces and capitalize
    return fault_type.replace("_", " ").title()
