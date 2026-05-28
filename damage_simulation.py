import numpy as np

def simulate_wall_collapse(pc: np.ndarray, frac: float = 0.65, seed: int = 0):
    """
    Yellow damage: remove ~65% of points from one randomly-chosen facade side,
    apply structural leaning (out-of-plumb), and add scattered debris near the base.
    Building remains standing but is clearly damaged/leaning.
    """
    rng = np.random.RandomState(seed)
    if pc.size == 0:
        return pc.copy()
    
    # 1. Identify "tall" points (likely walls/roof) vs "low" points (likely ground)
    z_min = pc[:, 2].min()
    z_threshold = z_min + 2.0  # Points more than 2m above ground are structure
    is_structure = pc[:, 2] > z_threshold
    
    structure_pts = pc[is_structure]
    ground_pts = pc[~is_structure]
    
    if len(structure_pts) == 0:
        return pc.copy()
    
    # 2. Compute centroid of structure points (not whole scene)
    centroid = structure_pts[:, :2].mean(axis=0)
    
    # 3. Compute angle of each structure point from centroid
    vecs = structure_pts[:, :2] - centroid[None, :]
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])  # range: [-pi, pi]
    
    # 4. Pick a random "facade direction" to damage
    cut_angle = rng.uniform(-np.pi, np.pi)
    
    # 5. Proper angular distance: shortest arc between two angles
    angular_dist = np.abs((angles - cut_angle + np.pi) % (2 * np.pi) - np.pi)
    
    # 6. Remove points within a wedge of width (frac * 2 * pi)
    wedge_half_width = np.pi * frac
    keep_mask = angular_dist > wedge_half_width
    
    kept_structure = structure_pts[keep_mask]
    
    if len(kept_structure) > 0:
        # Apply structural tilt (3 to 5 degrees) to simulate out-of-plumb leaning
        tilt_angle = rng.uniform(3.0, 5.0) * np.pi / 180.0
        # Determine tilt axis randomly
        axis_angle = rng.uniform(-np.pi, np.pi)
        ux, uy = np.cos(axis_angle), np.sin(axis_angle)
        
        c, s = np.cos(tilt_angle), np.sin(tilt_angle)
        # Rotation matrix around arbitrary axis (ux, uy, 0)
        R = np.array([
            [c + ux*ux*(1-c), ux*uy*(1-c),   uy*s],
            [uy*ux*(1-c),     c + uy*uy*(1-c), -ux*s],
            [-uy*s,           ux*s,            c]
        ])
        
        # Shift so base is at z=0, apply rotation, shift back
        kept_shifted = kept_structure.copy()
        kept_shifted[:, :2] -= centroid
        kept_shifted[:, 2] -= z_threshold
        kept_rotated = kept_shifted @ R.T
        kept_rotated[:, :2] += centroid
        kept_rotated[:, 2] += z_threshold
        kept_structure = kept_rotated
    
    # 7. Add debris near the base (scattered points at ground level near centroid)
    n_debris = int(len(structure_pts) * 0.30)  # Increased from 15% to 30%
    if n_debris > 0:
        debris_xy = centroid + rng.normal(scale=3.5, size=(n_debris, 2)) # Wider spread
        debris_z = z_min + np.abs(rng.normal(scale=0.8, size=(n_debris, 1))) # Higher debris piles
        debris = np.concatenate([debris_xy, debris_z], axis=1)
        result = np.vstack([ground_pts, kept_structure, debris])
    else:
        result = np.vstack([ground_pts, kept_structure])
    
    return result


def simulate_roof_damage(pc: np.ndarray, frac: float = 0.3, seed: int = 0):
    rng = np.random.RandomState(seed)
    if pc.size == 0:
        return pc.copy()
    zth = np.percentile(pc[:,2], 100 * (1.0 - frac))
    keep = pc[:,2] <= zth
    return pc[keep]


def simulate_full_collapse(pc: np.ndarray, seed: int = 0):
    """
    Red damage: completely collapse the building structure to debris field.
    Ground points are preserved. Structure points are flattened with noise.
    """
    rng = np.random.RandomState(seed)
    if pc.size == 0:
        return pc.copy()
    
    z_min = pc[:, 2].min()
    z_threshold = z_min + 2.0
    is_structure = pc[:, 2] > z_threshold
    
    structure_pts = pc[is_structure]
    ground_pts = pc[~is_structure]
    
    if len(structure_pts) == 0:
        return pc.copy()
    
    # Flatten structure to a debris field
    collapsed = structure_pts.copy()
    # Crush Z down to ground completely
    collapsed[:, 2] = z_min + np.abs(rng.normal(scale=0.1, size=collapsed.shape[0]))
    # Add horizontal spread (collapsed debris travels outward)
    horizontal_spread = rng.normal(scale=2.5, size=(collapsed.shape[0], 2))
    collapsed[:, :2] += horizontal_spread
    # Subsample to 30% (severe debris is very sparse)
    n_keep = int(len(collapsed) * 0.3)
    idx = rng.choice(len(collapsed), size=n_keep, replace=False)
    collapsed = collapsed[idx]
    
    return np.vstack([ground_pts, collapsed])


def apply_damage(point_cloud: np.ndarray, damage_type: str, seed: int = 0):
    if damage_type == "Green":
        return point_cloud.copy()
    if damage_type == "Yellow":
        return simulate_wall_collapse(point_cloud, frac=0.4, seed=seed)
    if damage_type == "Yellow_roof":
        return simulate_roof_damage(point_cloud, frac=0.3, seed=seed)
    if damage_type == "Red":
        return simulate_full_collapse(point_cloud, seed=seed)
    # default: return copy
    return point_cloud.copy()

