"""
GeoFlow-PC Wrapper — Self-Contained Integration Module
========================================================
Provides the GeoFlowPCWrapper class for loading the trained GeoFlow-PC v8
model and running inference (completion + descriptor extraction).

This is a self-contained version that imports from the local geoflow_model.py
instead of requiring the parent directory on sys.path.
"""

import numpy as np
import torch
from pathlib import Path

from geoflow_model import GeoFlowV8, CFG


class GeoFlowPCWrapper:
    """Wrapper for GeoFlow-PC v8 inference.
    
    Handles:
      - Model loading with automatic device selection
      - Point cloud normalization (unit sphere) before completion
      - Denormalization (restore world coordinates) after completion
      - Global descriptor extraction via the DGCNN encoder
      - Graceful fallback on completion failure
    
    Args:
        checkpoint_path: Path to the trained .pth checkpoint.
        device: Target device ('cuda' or 'cpu'). Auto-detected if None.
    """

    def __init__(self, checkpoint_path="best_v8.pth", device=None):
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading GeoFlow-PC from {checkpoint_path} onto {resolved_device}...", flush=True)
        self.device = resolved_device

        # Initialize the architecture using the local CFG
        self.model = GeoFlowV8(CFG)

        # Load weights
        try:
            state_dict = torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
            if "model_state_dict" in state_dict:
                self.model.load_state_dict(state_dict["model_state_dict"])
            elif "model" in state_dict:
                self.model.load_state_dict(state_dict["model"])
            else:
                self.model.load_state_dict(state_dict)

            # Apply EMA weights if available (critical for v8 stability)
            if "ema" in state_dict:
                print("Applying EMA weights...", flush=True)
                for k, v in state_dict["ema"].items():
                    if k in dict(self.model.named_parameters()):
                        dict(self.model.named_parameters())[k].data.copy_(v)

            self.model.to(self.device)
            self.model.eval()
            print("[OK] GeoFlow-PC weights loaded successfully.", flush=True)
        except Exception as e:
            print(f"[ERR] Failed to load weights: {e}", flush=True)

    def extract_descriptor(self, point_cloud: np.ndarray) -> np.ndarray:
        """Extract 384-d global descriptor from a point cloud.
        
        Args:
            point_cloud: (N, 3) numpy array.
        
        Returns:
            (384,) numpy array — the global geometry descriptor.
        """
        with torch.no_grad():
            n_partial = CFG["n_partial"]
            if point_cloud.shape[0] > n_partial:
                idx = np.random.choice(point_cloud.shape[0], n_partial, replace=False)
                point_cloud = point_cloud[idx, :]
            elif point_cloud.shape[0] < n_partial:
                idx = np.random.choice(point_cloud.shape[0], n_partial, replace=True)
                point_cloud = point_cloud[idx, :]

            # Normalize to unit sphere robustly
            centroid = point_cloud.mean(axis=0)
            centered = point_cloud - centroid
            distances = np.linalg.norm(centered, axis=1)
            scale = np.percentile(distances, 95)  # Tightened from 99 to 95 for robustness against extreme debris
            if scale < 1e-8:
                scale = 1.0
            normalized = centered / scale
            normalized = np.clip(normalized, -1.0, 1.0)

            pc_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
            per_point, global_descriptor = self.model.encoder(pc_tensor)
            return global_descriptor.cpu().numpy().flatten()

    def complete_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Complete a partial point cloud to n_complete points.
        
        Normalizes to unit sphere before completion, then denormalizes
        to restore absolute world coordinates.
        
        Args:
            point_cloud: (N, 3) numpy array — partial scan.
        
        Returns:
            (n_complete, 3) numpy array — completed point cloud.
        """
        with torch.no_grad():
            n_partial = CFG["n_partial"]
            if point_cloud.shape[0] > n_partial:
                idx = np.random.choice(point_cloud.shape[0], n_partial, replace=False)
                point_cloud = point_cloud[idx, :]
            elif point_cloud.shape[0] < n_partial:
                idx = np.random.choice(point_cloud.shape[0], n_partial, replace=True)
                point_cloud = point_cloud[idx, :]

            # Normalize to unit sphere robustly
            centroid = point_cloud.mean(axis=0)
            centered = point_cloud - centroid
            distances = np.linalg.norm(centered, axis=1)
            scale = np.percentile(distances, 95)  # Tightened from 99 to 95 for robustness against extreme debris
            if scale < 1e-8:
                scale = 1.0
            normalized = centered / scale
            # Clip to prevent massive outliers from destroying transformer attention
            normalized = np.clip(normalized, -1.0, 1.0)

            pc_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
            try:
                coarse, final = self.model(pc_tensor)
                out_normalized = final[0].cpu().numpy()
                # Denormalize to world coordinates
                out_world = out_normalized * scale + centroid
                return out_world
            except Exception as e:
                print(f"GeoFlow completion failed: {e}", flush=True)
                # Fallback: repeat/pad input
                n_complete = CFG.get("n_complete", 2048)
                if point_cloud.shape[0] >= n_complete:
                    return point_cloud[:n_complete]
                else:
                    reps = int(np.ceil(n_complete / point_cloud.shape[0]))
                    arr = np.tile(point_cloud, (reps, 1))[:n_complete]
                    return arr


if __name__ == "__main__":
    print("Testing GeoFlowPCWrapper (standalone)...")
    # Will fail gracefully if no checkpoint is present
    try:
        wrapper = GeoFlowPCWrapper("best_v8.pth")
        mock_pc = np.random.rand(512, 3).astype(np.float32)
        g = wrapper.extract_descriptor(mock_pc)
        print(f"  Descriptor shape: {g.shape}")
        completed = wrapper.complete_point_cloud(mock_pc)
        print(f"  Completed shape: {completed.shape}")
    except Exception as e:
        print(f"  Test skipped (no checkpoint): {e}")
