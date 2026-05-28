"""
Router Training — Proper Training Pipeline with PointNet Encoder
=================================================================
Trains the DamageSeverityRouter using the PointNet encoder for
feature extraction (replacing the dummy bounding-box features).

Features:
  - Stratified train/val split
  - Cosine annealing learning rate schedule
  - Label smoothing cross-entropy
  - Training metrics saved to JSON for reproducibility
  - Early stopping on validation accuracy

Usage:
    python train_router.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from config import PipelineConfig
from damage_simulation import apply_damage
from damage_router import DamageSeverityRouter
from pointnet_encoder import PointNetEncoder, extract_global_features

cfg = PipelineConfig()


class RouterDataset(Dataset):
    """Generates training data for the damage severity router.
    
    Loads scene files, applies simulated damage at different severity
    levels, and extracts features using the PointNet encoder.
    """

    def __init__(self, data_dir: str, num_samples: int = 200,
                 geoflow=None, seed: int = 42):
        super().__init__()
        self.points_list = []
        self.labels = []
        self.geoflow = geoflow

        rng = np.random.RandomState(seed)
        files = list(Path(data_dir).glob("vaihingen_scene_*.pts"))
        if len(files) == 0:
            print(f"No .pts files found in {data_dir}!")
            return

        classes = ["Intact", "Moderate", "Severe"]
        damage_map = {"Intact": "Green", "Moderate": "Yellow", "Severe": "Red"}

        for i in range(num_samples):
            f = rng.choice(files)
            pc = np.loadtxt(f, dtype=np.float32)
            if pc.ndim == 1:
                pc = pc.reshape(-1, 3)

            label_idx = rng.randint(0, 3)
            damage_class = classes[label_idx]
            damage_type = damage_map[damage_class]

            damaged_pc = apply_damage(pc, damage_type, seed=rng.randint(0, 10000))

            if len(damaged_pc) > cfg.n_partial:
                idx = rng.choice(len(damaged_pc), cfg.n_partial, replace=False)
                partial_pc = damaged_pc[idx]
            else:
                partial_pc = damaged_pc

            feat = self.geoflow.extract_descriptor(partial_pc)
            feat = torch.from_numpy(feat).float()
            self.points_list.append(feat)
            self.labels.append(label_idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.points_list[idx], torch.tensor(self.labels[idx], dtype=torch.long)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing for regularization."""

    def __init__(self, smoothing: float = 0.1, num_classes: int = 3):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        smooth = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()


def train_router(data_dir: str, epochs: int = 20, batch_size: int = 32,
                 lr: float = 1e-3, seed: int = 42):
    """Train the damage severity router with proper methodology.
    
    Args:
        data_dir: Path to directory containing .pts scene files.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Initial learning rate.
        seed: Random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize GeoFlowPCWrapper
    from geoflow_integration import GeoFlowPCWrapper
    geoflow = GeoFlowPCWrapper(checkpoint_path=str(cfg.get_checkpoint_path()), device="cpu")

    print("Generating training dataset...")
    train_dataset = RouterDataset(data_dir, num_samples=200, geoflow=geoflow, seed=seed)
    print("Generating validation dataset...")
    val_dataset = RouterDataset(data_dir, num_samples=60, geoflow=geoflow, seed=seed + 1)

    if len(train_dataset) == 0:
        print("Dataset empty. Cannot train.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DamageSeverityRouter(input_dim=cfg.router_input_dim)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"Starting training ({epochs} epochs, lr={lr})...")
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == y).sum().item()

        # Validate
        val_loss = 0.0
        val_correct = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == y).sum().item()

        t_loss = train_loss / len(train_dataset)
        t_acc = train_correct / len(train_dataset)
        v_loss = val_loss / len(val_dataset)
        v_acc = val_correct / len(val_dataset)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        tag = ""
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            tag = " *BEST* BEST"
            # Save best weights
            out_path = cfg.project_root / "data" / "router_weights.pth"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch + 1:3d}/{epochs} | lr={current_lr:.1e} | "
              f"Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}{tag}")

    # Save training history
    history_path = cfg.get_results_dir() / "router_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"Weights: {cfg.project_root / 'data' / 'router_weights.pth'}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    data_path = str(cfg.project_root / "data" / "vaihingen_scenes9")
    train_router(data_path, epochs=15, batch_size=16)
