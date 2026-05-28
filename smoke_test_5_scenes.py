import sys
from pathlib import Path
import numpy as np

from config import PipelineConfig
from geoflow_integration import GeoFlowPCWrapper
from reasoning_agent import TriageReActAgent
from run_ablation_final import process_single_scene
from unified_retrieval import UnifiedKnowledgeBase

def run_smoke_test():
    cfg = PipelineConfig()
    geoflow = GeoFlowPCWrapper(checkpoint_path=str(cfg.get_checkpoint_path()), device="cpu")
    kb = UnifiedKnowledgeBase(cfg)
    agent = TriageReActAgent(kb_interface=kb)
    rng = np.random.default_rng(42)
    
    # We want 2 Green, 1 Yellow, 2 Red
    scenes_to_test = [
        ("data/vaihingen_scenes9/vaihingen_scene_1.pts", "Green"),
        ("data/vaihingen_scenes9/vaihingen_scene_4.pts", "Green"),
        ("data/vaihingen_scenes9/vaihingen_scene_2.pts", "Yellow"),
        ("data/vaihingen_scenes9/vaihingen_scene_3.pts", "Red"),
        ("data/vaihingen_scenes9/vaihingen_scene_6.pts", "Red"),
    ]
    
    # We'll test with "Single-Branch" so we bypass the router completely. 
    # The goal is to see if the agent interprets the features properly.
    arch = "Single-Branch"
    
    for scene_rel, damage_class in scenes_to_test:
        scene_abs = str(cfg.project_root / scene_rel)
        print(f"\n{'='*60}")
        print(f"Testing Scene: {Path(scene_abs).name} | Truth: {damage_class}")
        res = process_single_scene(scene_abs, damage_class, arch, geoflow, agent, rng)
        
        print("\n--- RESULTS ---")
        if "error" in res:
            print(f"ERROR: {res['error']}")
        else:
            print(f"Predicted Class: {res['predicted_class']}")
            print(f"Confidence: {res['confidence']:.2f}")

if __name__ == "__main__":
    run_smoke_test()
