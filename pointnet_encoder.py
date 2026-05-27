"""
PointNet-Style Feature Encoder for Damage Severity Routing
===========================================================
Replaces the dummy `extract_global_features()` with a lightweight but
meaningful encoder that produces a 384-dim feature vector from raw 3D
point clouds. The output dimension matches GeoFlow's d_model for
architectural consistency.

Architecture:
  - Shared MLP (3 -> 64 -> 128 -> 256) with BatchNorm + ReLU
  - Symmetric max-pooling -> 256-dim global shape signature
  - Concatenation with hand-crafted statistical descriptors (128-dim)
  - Projection MLP (256 + 128 -> 384) -> final feature vector

The statistical features are domain-informed for structural assessment:
  - Per-axis mean, std (6 dims)
  - Height percentiles: 10th, 25th, 50th, 75th, 90th (5 dims)
  - Bounding box dimensions (3 dims)
  - Bounding box aspect ratios (2 dims)
  - Point density proxy (1 dim)
  - Planarity eigenvalue ratios (3 dims)
  - Total: 20 dims (padded to 128 for MLP compatibility)

Reference:
  Qi et al., "PointNet: Deep Learning on Point Sets" (CVPR 2017)
"""

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Statistical Feature Extraction (NumPy, runs on CPU)
# ─────────────────────────────────────────────────────────────

def compute_statistical_features(pc: np.ndarray) -> np.ndarray:
    """Compute domain-informed statistical descriptors from a point cloud.
    
    Args:
        pc: (N, 3) numpy array of XYZ coordinates.
    
    Returns:
        (128,) numpy array of statistical features (zero-padded).
    """
    features = np.zeros(128, dtype=np.float32)
    if pc.shape[0] < 3:
        return features

    xs, ys, zs = pc[:, 0], pc[:, 1], pc[:, 2]

    # Per-axis mean and std (6 dims)
    features[0] = xs.mean()
    features[1] = ys.mean()
    features[2] = zs.mean()
    features[3] = xs.std()
    features[4] = ys.std()
    features[5] = zs.std()

    # Height percentiles (5 dims) — critical for damage assessment
    for i, p in enumerate([10, 25, 50, 75, 90]):
        features[6 + i] = np.percentile(zs, p)

    # Bounding box dimensions (3 dims)
    bbox = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()])
    features[11:14] = bbox

    # Bounding box aspect ratios (2 dims)
    # height/footprint_x, height/footprint_y
    features[14] = bbox[2] / max(bbox[0], 1e-6)
    features[15] = bbox[2] / max(bbox[1], 1e-6)

    # Point density proxy: N / bbox_volume (1 dim)
    volume = max(bbox[0] * bbox[1] * bbox[2], 1e-8)
    features[16] = pc.shape[0] / volume

    # Planarity via PCA eigenvalue ratios (3 dims)
    try:
        centered = pc - pc.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))
        eig_sum = eigenvalues.sum()
        if eig_sum > 1e-8:
            # Linearity, planarity, sphericity (Weinmann et al., 2015)
            features[17] = (eigenvalues[2] - eigenvalues[1]) / eig_sum  # linearity
            features[18] = (eigenvalues[1] - eigenvalues[0]) / eig_sum  # planarity
            features[19] = eigenvalues[0] / eig_sum                      # sphericity
    except np.linalg.LinAlgError:
        pass

    return features


# ─────────────────────────────────────────────────────────────
# PointNet Encoder (PyTorch)
# ─────────────────────────────────────────────────────────────

class PointNetEncoder(nn.Module):
    """Lightweight PointNet-style encoder for global feature extraction.
    
    Produces a d_out-dimensional feature vector capturing:
    - Local geometric patterns (via shared MLPs on per-point features)
    - Global shape signature (via symmetric max-pooling)
    - Statistical distribution features (domain-informed descriptors)
    
    Args:
        d_out: Output feature dimension (default: 384, matching GeoFlow d_model).
        stat_dim: Dimension of statistical feature vector (default: 128).
    """

    def __init__(self, d_out: int = 384, stat_dim: int = 128):
        super().__init__()
        self.d_out = d_out
        self.stat_dim = stat_dim

        # Shared MLP: per-point feature learning
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Projection: learned_features (256) + statistical_features (stat_dim) -> d_out
        self.projection = nn.Sequential(
            nn.Linear(256 + stat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, xyz: torch.Tensor, stat_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            xyz: (B, N, 3) point cloud coordinates.
            stat_features: (B, stat_dim) pre-computed statistical features.
        
        Returns:
            (B, d_out) global feature vector.
        """
        # Per-point features via shared MLP
        x = xyz.permute(0, 2, 1)       # (B, 3, N)
        x = self.shared_mlp(x)          # (B, 256, N)

        # Symmetric max-pooling -> global shape signature
        x_global = x.max(dim=2)[0]      # (B, 256)

        # Concatenate with statistical features
        combined = torch.cat([x_global, stat_features], dim=1)  # (B, 256 + stat_dim)

        # Project to output dimension
        return self.projection(combined)  # (B, d_out)

    def extract_features(self, point_cloud: np.ndarray) -> torch.Tensor:
        """Convenience method: numpy point cloud -> feature tensor.
        
        Args:
            point_cloud: (N, 3) numpy array.
        
        Returns:
            (1, d_out) feature tensor (on CPU, detached).
        """
        self.eval()
        with torch.no_grad():
            # Compute statistical features
            stat = compute_statistical_features(point_cloud)
            stat_tensor = torch.tensor(stat, dtype=torch.float32).unsqueeze(0)

            # Prepare point cloud tensor
            pc_tensor = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0)

            # Forward
            features = self.forward(pc_tensor, stat_tensor)
        return features


def extract_global_features(partial_pc: np.ndarray, encoder: PointNetEncoder = None) -> torch.Tensor:
    """Drop-in replacement for the old dummy extract_global_features().
    
    When no encoder is provided, falls back to statistical-only features
    projected to the expected dimension (384) — still far better than
    the old zero-padded bounding box approach.
    
    Args:
        partial_pc: (N, 3) numpy array.
        encoder: Optional trained PointNetEncoder instance.
    
    Returns:
        (1, 384) feature tensor.
    """
    if encoder is not None:
        return encoder.extract_features(partial_pc)

    # Fallback: statistical features only (no learned component)
    # Pad to 384 dims to match router input expectation
    stat = compute_statistical_features(partial_pc)
    padded = np.zeros(384, dtype=np.float32)
    padded[:len(stat)] = stat
    return torch.tensor(padded, dtype=torch.float32).unsqueeze(0)


# ─────────────────────────────────────────────────────────────
# Self-Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing PointNetEncoder...")

    # Test statistical features
    mock_pc = np.random.rand(512, 3).astype(np.float32) * 10
    stat = compute_statistical_features(mock_pc)
    print(f"  Statistical features shape: {stat.shape}")
    print(f"  Non-zero entries: {np.count_nonzero(stat)}")

    # Test encoder
    encoder = PointNetEncoder(d_out=384)
    features = encoder.extract_features(mock_pc)
    print(f"  Encoder output shape: {features.shape}")
    assert features.shape == (1, 384), f"Expected (1, 384), got {features.shape}"

    # Test drop-in function
    feat_fallback = extract_global_features(mock_pc)
    print(f"  Fallback output shape: {feat_fallback.shape}")

    feat_encoder = extract_global_features(mock_pc, encoder=encoder)
    print(f"  Encoder-based output shape: {feat_encoder.shape}")

    print("  [OK] All tests passed!")
