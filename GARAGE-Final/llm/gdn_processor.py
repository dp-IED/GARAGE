"""
GDN Processor Adapter

Lightweight adapter that wraps existing GARAGE-Final helper functions from kg/create_kg.py
to provide a compatible interface for LLM/RAG code.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Import existing helper functions from kg.create_kg
from kg.create_kg import (
    load_gdn_model,
    extract_sensor_embeddings,
    compute_adjacency_matrix,
    predict_anomalies,
    extract_window_embeddings,
)
from models.gdn_model import GDN


class GDNPredictor:
    """
    GDN Predictor adapter that wraps existing GARAGE-Final helper functions.

    This class provides a compatible interface for LLM/RAG code while reusing
    the existing helper functions from kg/create_kg.py.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        sensor_names: List[str],
        window_size: int = 300,
        embed_dim: int = 64,
        top_k: int = 3,
        hidden_dim: int = 32,
        device: str = "cpu",
    ):
        """
        Initialize the GDN Predictor.

        Args:
            model_path: Path to trained model checkpoint (.pt file)
            sensor_names: List of sensor names (must match model's num_nodes)
            window_size: Size of input windows (overridden by checkpoint if present)
            embed_dim: Embedding dimension (overridden by checkpoint if present)
            top_k: Top-K neighbors (overridden by checkpoint if present)
            hidden_dim: Hidden dimension (overridden by checkpoint if present)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_path = Path(model_path)
        self.sensor_names = sensor_names
        self.device = device

        self.model, self.metadata = load_gdn_model(str(self.model_path), device=device)

        self._normal_center = None
        self._anomalous_center = None
        self._global_mask_threshold = 0.5
        self._per_sensor_thresholds = np.array([], dtype=np.float32)
        self._sensor_score_threshold = 0.3
        try:
            ckpt = torch.load(
                str(self.model_path), map_location=device, weights_only=False
            )
            if isinstance(ckpt, dict) and "sensor_centers" in ckpt:
                sc = ckpt["sensor_centers"]
                sc = sc.detach().cpu().numpy()
                if sc.ndim == 3:
                    self._normal_center = sc[:, 0, :].mean(axis=0).astype(np.float32)
                    self._anomalous_center = sc[:, 1, :].mean(axis=0).astype(np.float32)
            if isinstance(ckpt, dict):
                calibrated = ckpt.get("calibrated_thresholds", {})
                self._global_mask_threshold = float(calibrated.get("window", 0.5))
                self._sensor_score_threshold = float(
                    calibrated.get("sensor", ckpt.get("sensor_threshold", 0.3))
                )
                per_sensor = calibrated.get("per_sensor", [])
                if per_sensor:
                    self._per_sensor_thresholds = np.array(per_sensor, dtype=np.float32)
        except Exception:
            pass

        self.window_size = self.metadata.get("window_size", window_size)
        self.embed_dim = self.metadata.get("embed_dim", embed_dim)
        self.top_k = self.metadata.get("top_k", top_k)
        self.hidden_dim = self.metadata.get("hidden_dim", hidden_dim)
        self.num_sensors = len(sensor_names)

    @property
    def per_sensor_thresholds(self) -> np.ndarray:
        """Per-sensor thresholds from checkpoint (empty array if not available)."""
        return self._per_sensor_thresholds

    @property
    def sensor_score_threshold(self) -> float:
        """Global calibrated anomaly score threshold (sensor-level) from checkpoint."""
        return float(self._sensor_score_threshold)

    def get_sensor_embeddings(self) -> np.ndarray:
        """
        Extract learned sensor embeddings from the model.

        Returns:
            sensor_embeddings: (num_sensors, embed_dim) numpy array
        """
        return extract_sensor_embeddings(self.model)

    def compute_adjacency_matrix(self) -> np.ndarray:
        """
        Compute adjacency matrix from sensor embeddings using cosine similarity.

        Returns:
            adjacency_matrix: (num_sensors, num_sensors) numpy array
        """
        sensor_embeddings = self.get_sensor_embeddings()
        return compute_adjacency_matrix(sensor_embeddings)

    def get_corr_embedding(
        self,
        X_windows: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract window embeddings for distance-based scoring.

        Args:
            X_windows: (num_windows, window_size, num_sensors) input windows
            batch_size: Batch size for inference

        Returns:
            embeddings: (num_windows, hidden_dim) numpy array of embeddings
        """
        return extract_window_embeddings(
            self.model, X_windows, batch_size=batch_size, device=self.device
        )

    def predict(
        self,
        X_windows: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32,
        apply_global_mask: bool = True,
        global_mask_threshold: Optional[float] = None,
        return_global: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on data windows to get sensor anomaly probabilities.

        Args:
            X_windows: (num_windows, window_size, num_sensors) input windows
            batch_size: Batch size for inference
            global_mask_threshold: If None, use checkpoint-calibrated value.

        Returns:
            sensor_probs: (num_windows, num_sensors) numpy array of anomaly probabilities.
            If `return_global=True`, returns tuple (sensor_probs, global_probs).
        """
        thr = (
            global_mask_threshold
            if global_mask_threshold is not None
            else self._global_mask_threshold
        )
        return predict_anomalies(
            self.model,
            X_windows,
            batch_size=batch_size,
            device=self.device,
            return_global=return_global,
            apply_global_mask=apply_global_mask,
            global_mask_threshold=thr,
        )

    def process_for_kg(
        self,
        X_windows: Union[np.ndarray, torch.Tensor],
        sensor_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        window_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        batch_size: int = 32,
        apply_global_mask: bool = True,
        global_mask_threshold: Optional[float] = None,
        return_window_labels_from_mask: bool = True,
    ) -> Dict[str, Union[List[str], np.ndarray]]:
        """
        Process data and return dict needed for KG construction.
        """
        if isinstance(X_windows, torch.Tensor):
            X_windows = X_windows.cpu().numpy()
        X_windows = np.asarray(X_windows)

        thr = (
            global_mask_threshold
            if global_mask_threshold is not None
            else self._global_mask_threshold
        )
        sensor_embeddings = self.get_sensor_embeddings()
        adjacency_matrix = self.compute_adjacency_matrix()
        gdn_predictions = self.predict(
            X_windows,
            batch_size=batch_size,
            apply_global_mask=apply_global_mask,
            global_mask_threshold=thr,
        )

        if sensor_labels is not None:
            if isinstance(sensor_labels, torch.Tensor):
                sensor_labels = sensor_labels.cpu().numpy()
            sensor_labels = np.asarray(sensor_labels)
        else:
            sensor_labels = np.zeros(
                (len(X_windows), self.num_sensors), dtype=np.float32
            )

        if window_labels is not None:
            if isinstance(window_labels, torch.Tensor):
                window_labels = window_labels.cpu().numpy()
            window_labels = np.asarray(window_labels).astype(np.int64)
        else:
            window_labels = (
                (gdn_predictions.max(axis=1) > thr).astype(np.int64)
                if return_window_labels_from_mask
                else (gdn_predictions.max(axis=1) > 0.5).astype(np.int64)
            )

        return {
            "sensor_names": self.sensor_names,
            "sensor_embeddings": sensor_embeddings,
            "adjacency_matrix": adjacency_matrix,
            "X_windows": X_windows,
            "gdn_predictions": gdn_predictions,
            "sensor_labels": sensor_labels,
            "window_labels": window_labels,
        }
