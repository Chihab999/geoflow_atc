"""
Diagnostic Checks — Pipeline Component Verification
=====================================================
Verifies that all pipeline components are working correctly:
  1. Damage simulation produces expected Z-statistics
  2. Feature extraction generates valid descriptions
  3. Router produces sensible predictions
  4. All imports resolve without sys.path hacks

Run this before the ablation study to catch configuration issues.
"""

import numpy as np
import torch
from pathlib import Path

from config import PipelineConfig, SEVERITY_BRANCHES
from damage_simulation import apply_damage
from geometric_features import compute_features
from description_builder_v2 import build_description
from damage_router import DamageSeverityRouter
from pointnet_encoder import PointNetEncoder, extract_global_features

cfg = PipelineConfig()


def run_diagnostics():
    """Execute all diagnostic checks."""
    # Deterministic seeds
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    scene_file = cfg.project_root / "data" / "vaihingen_scenes9" / "vaihingen_scene_1.pts"

    if not scene_file.exists():
        print(f"[ERR] Scene file not found: {scene_file}")
        print("   Run build_scene_index.py first.")
        return

    print("=" * 50)
    print("CHECK 1: Damage Simulation Z-Statistics")
    print("=" * 50 + "\n")

    pc_orig = np.loadtxt(scene_file, dtype=np.float32)
    if pc_orig.ndim == 1:
        pc_orig = pc_orig.reshape(-1, 3)

    descriptions = {}
    green_pc = None

    for damage_class in ["Green", "Yellow", "Red"]:
        pc_damaged = apply_damage(pc_orig, damage_class, seed=42)

        print(f"--- {damage_class} ---")
        print(f"  Point count: {len(pc_damaged)}")
        if len(pc_damaged) > 0:
            print(f"  Z range: {pc_damaged[:, 2].min():.2f} to {pc_damaged[:, 2].max():.2f}")
            print(f"  Z std: {pc_damaged[:, 2].std():.2f}")
            print(f"  Mean Z of top 25%: {np.percentile(pc_damaged[:, 2], 75):.2f}")
        else:
            print("  Empty point cloud!")

        if len(pc_damaged) > cfg.n_partial:
            idx = np.random.choice(len(pc_damaged), cfg.n_partial, replace=False)
            pc_feat = pc_damaged[idx]
        else:
            pc_feat = pc_damaged

        if damage_class == "Green":
            green_pc = pc_feat

        mock_completed = np.concatenate([pc_feat, pc_feat, pc_feat, pc_feat])
        partial_feats = compute_features(pc_feat, partial_pc_size=len(pc_feat))
        feats = compute_features(mock_completed, partial_pc_size=len(pc_feat))
        descriptions[damage_class] = build_description(feats, partial_feats=partial_feats)

    print("\n" + "=" * 50)
    print("CHECK 2: Natural Language Descriptions")
    print("=" * 50 + "\n")

    for damage_class, desc in descriptions.items():
        print(f"--- Description for {damage_class} ---")
        print(desc)
        print()

    print("=" * 50)
    print("CHECK 3: PointNet Encoder Output")
    print("=" * 50 + "\n")

    encoder = PointNetEncoder(d_out=cfg.router_input_dim)
    feat = extract_global_features(green_pc, encoder=encoder)
    print(f"  PointNet output shape: {feat.shape}")
    print(f"  Feature norm: {torch.norm(feat).item():.4f}")
    print(f"  Feature range: [{feat.min().item():.4f}, {feat.max().item():.4f}]")

    print("\n" + "=" * 50)
    print("CHECK 4: Router Prediction")
    print("=" * 50 + "\n")

    router = DamageSeverityRouter(input_dim=cfg.router_input_dim)
    weights_path = cfg.get_router_weights_path()
    if weights_path.exists():
        try:
            router.load_state_dict(torch.load(weights_path, weights_only=True))
            print("  -> Router weights loaded successfully.")
        except RuntimeError as e:
            print(f"  -> Warning: Router weights incompatible (likely old input_dim=512).")
            print(f"     Re-train with: python train_router.py")
            print(f"     Using random weights for this check.")
    else:
        print("  -> Warning: router_weights.pth not found! Using random weights.")

    router.eval()
    with torch.no_grad():
        logits = router(feat)
        probs = torch.softmax(logits, dim=-1)[0]
        branch_idx = torch.argmax(logits, dim=-1).item()

    print(f"  Scene 1 (Truth = Green) with PointNet features:")
    print(f"  Predicted branch: {SEVERITY_BRANCHES[branch_idx]} "
          f"(probs: {[f'{p:.3f}' for p in probs.tolist()]})")

    print("\n" + "=" * 50)
    print("CHECK 5: Import Verification")
    print("=" * 50 + "\n")

    # Verify no sys.path usage
    import importlib
    modules = [
        "config", "geoflow_model", "geoflow_integration",
        "pointnet_encoder", "damage_router", "damage_simulation",
        "geometric_features", "description_builder_v2",
        "unified_retrieval", "reasoning_agent", "evaluation",
        "pipeline_integrated", "run_ablation_final",
    ]
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            print(f"  [OK] {mod_name}")
        except ImportError as e:
            print(f"  [ERR] {mod_name}: {e}")

    print("\n" + "=" * 50)
    print("ALL DIAGNOSTICS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    run_diagnostics()
