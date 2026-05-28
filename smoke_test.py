import numpy as np
import time
from pathlib import Path
from config import PipelineConfig
from geoflow_integration import GeoFlowPCWrapper
from reasoning_agent import TriageReActAgent
from unified_retrieval import UnifiedKnowledgeBase
from run_ablation_final import process_single_scene

cfg = PipelineConfig()
geoflow = GeoFlowPCWrapper(checkpoint_path=str(cfg.get_checkpoint_path()))
kb = UnifiedKnowledgeBase(config=cfg)
agent = TriageReActAgent(kb_interface=kb, perception_model=geoflow, config=cfg)
rng = np.random.default_rng(42)

scene_abs = str(cfg.project_root / "data" / "vaihingen_scenes9" / "vaihingen_scene_1.pts")
res = process_single_scene(scene_abs, "Green", "Single-Branch", geoflow, agent, rng)

print("SMOKE TEST RESULTS:")
for k, v in res.get("diagnostics", {}).items():
    if k in ["partial_z_range", "used_z_range", "partial_x_range", "used_x_range", "bbox_x", "height_std"]:
        print(f"{k}: {v}")
