import os
import json
from pathlib import Path

def build_scene_index():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / "vaihingen_scenes9"
    out_file = base_dir / "data" / "scene_index.json"
    
    out_file.parent.mkdir(exist_ok=True, parents=True)
    
    files = list(data_dir.glob("vaihingen_scene_*.pts"))
    
    # Use paths relative to the project root
    index = {
        "vaihingen": [str(f.relative_to(base_dir).as_posix()) for f in files]
    }
    
    with open(out_file, "w") as f:
        json.dump(index, f, indent=2)
        
    print(f"Wrote scene_index.json with {len(files)} files.")

if __name__ == "__main__":
    build_scene_index()
