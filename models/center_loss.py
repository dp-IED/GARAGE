"""
Sensor-Only Center Loss for Supervised Anomaly Detection

Learns separate normal/anomaly centers for each sensor type.
This enables better sensor attribution in knowledge graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorOnlyCenterLoss(nn.Module):
    """
    Sensor-level center loss ONLY (no window-level centers).

    This is simpler, faster, and better for sensor attribution in KAG.
    Learns separate normal/anomaly centers for each sensor type.

    CRITICAL: All embeddings and centers are L2-normalized before computing
    distances to avoid degenerate solutions (centers at opposite poles).
    """

    def __init__(
        self,
        embed_dim=64,
        num_sensors=8,
        num_classes=2,
        margin=2.0,
        lambda_intra=1.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_sensors = num_sensors
        self.num_classes = num_classes
        self.margin = margin
        self.lambda_intra = lambda_intra

        # Only sensor-level centers (num_sensors × num_classes)
        self.sensor_centers = nn.Parameter(
            torch.randn(num_sensors, num_classes, embed_dim)
        )
        nn.init.xavier_uniform_(self.sensor_centers)

    def forward(self, sensor_embeddings, sensor_labels):
        """
        Compute sensor-only center loss.

        Args:
            sensor_embeddings: (B, N, embed_dim) - sensor-level embeddings
            sensor_labels: (B, N) - sensor labels (0=normal, 1=anomaly)

        Returns:
            total_loss: scalar
        """
        B, N, D = sensor_embeddings.shape

        # Normalize embeddings and centers
        sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=2)
        sensor_centers_norm = F.normalize(self.sensor_centers, p=2, dim=2)

        # Intra-class loss (pull embeddings to centers)
        intra_loss = 0.0
        count = 0

        for sensor_idx in range(N):
            sensor_embs = sensor_embeddings[:, sensor_idx, :]  # (B, D)
            sensor_labs = sensor_labels[:, sensor_idx]  # (B,)

            for class_id in range(self.num_classes):
                mask = sensor_labs == class_id
                if mask.sum() == 0:
                    continue

                class_embeddings = sensor_embs[mask]
                class_center = sensor_centers_norm[sensor_idx, class_id]

                # Euclidean distance to center
                distances = torch.norm(class_embeddings - class_center, p=2, dim=1)
                intra_loss += distances.mean()
                count += 1

        if count > 0:
            intra_loss /= count

        # Repulsion loss (push normal/anomaly centers apart per sensor)
        repulsion_loss = 0.0
        if self.num_classes == 2:
            for sensor_idx in range(N):
                center_distance = torch.norm(
                    sensor_centers_norm[sensor_idx, 0]
                    - sensor_centers_norm[sensor_idx, 1],
                    p=2,
                )
                # Exponential repulsion (stronger as centers get closer)
                repulsion_loss += torch.exp(-center_distance + 1.0)
            repulsion_loss /= N

        # Combined loss
        total_loss = self.lambda_intra * intra_loss + repulsion_loss

        return total_loss

    def get_sensor_centers(self):
        """Return normalized sensor centers."""
        return F.normalize(self.sensor_centers, p=2, dim=2)

    def get_separations(self):
        """Compute per-sensor separation metrics."""
        sensor_centers_norm = F.normalize(self.sensor_centers, p=2, dim=2)

        sensor_seps = []
        for sensor_idx in range(self.num_sensors):
            sep = torch.norm(
                sensor_centers_norm[sensor_idx, 0] - sensor_centers_norm[sensor_idx, 1],
                p=2,
            ).item()
            sensor_seps.append(sep)

        sensor_seps = torch.tensor(sensor_seps)

        return {
            "sensor_mean_separation": sensor_seps.mean().item(),
            "sensor_min_separation": sensor_seps.min().item(),
            "sensor_max_separation": sensor_seps.max().item(),
        }
