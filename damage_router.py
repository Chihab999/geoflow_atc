"""
Damage Severity Router — Lightweight MLP Classifier
=====================================================
Routes partial point clouds to the appropriate GeoFlow completion
branch based on inferred damage severity.

Classes:
  0: Intact  -> Branch A (full completion)
  1: Moderate -> Branch B (partial completion with LoRA)
  2: Severe  -> Branch C (bypass completion, use partial directly)

Architecture:
  4-layer MLP with BatchNorm, ReLU, and Dropout regularization.
  Input dimension matches PointNet encoder output (384).
"""

import torch
import torch.nn as nn

from config import PipelineConfig

cfg = PipelineConfig()


class DamageSeverityRouter(nn.Module):
    """MLP router mapping global point cloud features to severity classes.
    
    Args:
        input_dim: Feature vector dimension (default: 384 from PointNet encoder).
        hidden_dim: Hidden layer dimension.
        num_classes: Number of output classes (default: 3).
        dropout: Dropout probability for regularization.
    """

    def __init__(self, input_dim: int = None, hidden_dim: int = None,
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        input_dim = input_dim or cfg.router_input_dim
        hidden_dim = hidden_dim or cfg.router_hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout // 2),  # Lower dropout in final layers

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (batch_size, input_dim) feature tensor.
        
        Returns:
            (batch_size, num_classes) logits.
        """
        return self.mlp(x)

    def predict(self, x: torch.Tensor) -> tuple:
        """Predict severity class with probabilities.
        
        Args:
            x: (batch_size, input_dim) feature tensor.
        
        Returns:
            (class_indices, probabilities) tuple.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            classes = torch.argmax(probs, dim=-1)
        return classes, probs


def test_router():
    """Verify router forward pass with the new input dimension."""
    model = DamageSeverityRouter(input_dim=cfg.router_input_dim)
    x = torch.randn(4, cfg.router_input_dim)
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)
    assert probs.shape == (4, 3), f"Expected (4, 3), got {probs.shape}"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    print(f"DamageSeverityRouter test passed. Input dim: {cfg.router_input_dim}, output shape: {probs.shape}")


if __name__ == "__main__":
    test_router()
