"""
Integrated Pipeline — End-to-End Structural Triage
====================================================
Orchestrates the complete pipeline:
  1. Load scene point cloud
  2. Simulate earthquake damage
  3. Route through damage severity classifier
  4. Complete partial cloud via GeoFlow-PC
  5. Extract geometric features
  6. Build natural language description
  7. Execute ReAct agent for ATC-20 classification

All imports are local — no sys.path hacks required.
"""

import json
import numpy as np
import torch
from pathlib import Path

from config import PipelineConfig, SEVERITY_BRANCHES
from damage_simulation import apply_damage
from geometric_features import compute_features
from description_builder_v2 import build_description
from damage_router import DamageSeverityRouter
from pointnet_encoder import PointNetEncoder, extract_global_features
from reasoning_agent import TriageReActAgent
from unified_retrieval import UnifiedKnowledgeBase
from geoflow_integration import GeoFlowPCWrapper

cfg = PipelineConfig()


def run_pipeline(scene_file: str, damage_class: str = "Yellow",
                 checkpoint: str = None, config: PipelineConfig = None):
    """Execute the full triage pipeline on a single scene.
    
    Args:
        scene_file: Path to the .pts point cloud file.
        damage_class: Ground-truth damage class for simulation.
        checkpoint: Path to GeoFlow checkpoint (auto-resolved if None).
        config: Pipeline configuration.
    
    Returns:
        Natural language description string, or None on failure.
    """
    config = config or cfg

    # 1. Load scene
    pc = np.loadtxt(scene_file, dtype=np.float32)
    if pc.ndim == 1:
        pc = pc.reshape(-1, 3)

    # 2. Simulate Damage
    damaged_pc = apply_damage(pc, damage_class)
    if len(damaged_pc) > config.n_partial:
        idx = np.random.choice(len(damaged_pc), config.n_partial, replace=False)
        partial_pc = damaged_pc[idx]
    else:
        partial_pc = damaged_pc

    print(f"[{Path(scene_file).name}] Intact: {len(pc)}, "
          f"Damaged: {len(damaged_pc)}, Partial: {len(partial_pc)}")

    # 3. Router — using PointNet encoder for meaningful features
    router = DamageSeverityRouter(input_dim=config.router_input_dim)
    weights_path = config.get_router_weights_path()
    if weights_path.exists():
        try:
            router.load_state_dict(torch.load(weights_path, weights_only=True))
        except RuntimeError:
            print("  [WARN] Router weights incompatible. Using random weights.")
    router.eval()

    with torch.no_grad():
        router_feat = extract_global_features(partial_pc)
        logits = router(router_feat)
        probs = torch.softmax(logits, dim=-1)[0]

    branch = torch.argmax(probs).item()
    print(f"Router predicted branch: {SEVERITY_BRANCHES[branch]} "
          f"(probs: {[f'{p:.3f}' for p in probs.tolist()]})")

    # 4. Completion (Branch A/B/C)
    checkpoint_path = checkpoint or str(config.get_checkpoint_path())
    geoflow = GeoFlowPCWrapper(checkpoint_path=checkpoint_path)

    try:
        completed_pc = geoflow.complete_point_cloud(partial_pc)
        completed_used = True
    except Exception as e:
        print(f"Completion failed, using partial. Error: {e}")
        completed_pc = partial_pc
        completed_used = False

    # 5. Features
    try:
        used_pc = completed_pc if completed_used else partial_pc
        feats = compute_features(used_pc, partial_pc_size=len(partial_pc))
        partial_feats = compute_features(partial_pc, partial_pc_size=len(partial_pc))
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

    # 6. Description
    desc = build_description(feats, partial_features=partial_feats)
    print("\n--- Description ---")
    print(desc)
    print("-------------------\n")

    # 7. Agentic Reasoning (LLM)
    print("Initializing Knowledge Base and LLM Agent...")
    try:
        kb = UnifiedKnowledgeBase(config=config)
        agent = TriageReActAgent(
            kb_interface=kb,
            perception_model=geoflow,
            config=config,
        )

        print("Sending to LLM for ReAct traversal...")
        final_classification, reasoning_trace = agent.run(
            description=desc,
            point_cloud_payload=feats,
        )

        print(f"\n[FINAL CLASSIFICATION]: {final_classification}")
    except Exception as e:
        print(f"\nAgent execution failed. Make sure your LLM server "
              f"(vLLM/Ollama) is running on port 11434.\nError: {e}")

    return desc


if __name__ == "__main__":
    test_scene = str(cfg.project_root / "data" / "scene_index.json")
    with open(test_scene, "r") as f:
        idx = json.load(f)["vaihingen"]

    # Run on one sample
    sample_path = str(cfg.project_root / idx[0])
    run_pipeline(sample_path, damage_class="Red")
