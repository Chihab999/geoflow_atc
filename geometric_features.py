"""
Geometric Feature Extraction — PhD-Level Structural Descriptors
================================================================
Computes geometric and statistical features from 3D point clouds
for structural damage assessment. Features are designed to capture
damage-indicative geometric signatures.

Feature Categories:
  1. Bounding Box & Height Statistics
  2. Completeness Ratio (post-completion vs pre-completion)
  3. Top-Surface Planarity (PCA-based)
  4. Vertical Density Distribution (low/mid/high tertiles)
  5. Surface Normal Consistency (local PCA normals)
  6. Roughness Index (local distance variance)
  7. Symmetry Score (bilateral symmetry along principal axes)
  8. Volume Estimation (convex hull)
  9. Aspect Ratios (height/footprint)

References:
  - Weinmann et al., "Semantic 3D point cloud interpretation" (ISPRS 2015)
  - Hackel et al., "Fast semantic segmentation of 3D point clouds" (2016)
"""

import numpy as np
from scipy.spatial import cKDTree, ConvexHull


def _percentile(vals, p):
    """Safe percentile computation."""
    return float(np.percentile(vals, p))


def _estimate_normals(pc: np.ndarray, k: int = 20) -> np.ndarray:
    """Estimate surface normals via local PCA (k nearest neighbors).
    
    Args:
        pc: (N, 3) point cloud.
        k: Number of neighbors for local plane fitting.
    
    Returns:
        (N, 3) unit normal vectors.
    """
    n_points = pc.shape[0]
    if n_points < k:
        return np.zeros((n_points, 3), dtype=np.float32)

    tree = cKDTree(pc)
    _, indices = tree.query(pc, k=min(k, n_points))
    normals = np.zeros((n_points, 3), dtype=np.float32)

    for i in range(n_points):
        neighbors = pc[indices[i]]
        centered = neighbors - neighbors.mean(axis=0)
        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Normal is the eigenvector with smallest eigenvalue
            normals[i] = eigenvectors[:, 0]
        except np.linalg.LinAlgError:
            normals[i] = [0, 0, 1]  # Default to vertical

    # Orient normals consistently (pointing upward)
    flip_mask = normals[:, 2] < 0
    normals[flip_mask] *= -1

    return normals


def _normal_consistency(normals: np.ndarray) -> float:
    """Compute normal consistency score.
    
    Measures how uniform the surface normals are across the point cloud.
    Intact buildings have highly consistent normals (flat surfaces);
    damaged/collapsed structures have scattered normal directions.
    
    Returns:
        Float in [0, 1]. Higher = more consistent (more intact).
    """
    if normals.shape[0] < 3:
        return 0.0
    # Compute mean normal direction
    mean_normal = normals.mean(axis=0)
    norm = np.linalg.norm(mean_normal)
    if norm < 1e-8:
        return 0.0
    mean_normal /= norm
    # Consistency = average cosine similarity to mean normal
    dots = np.abs(normals @ mean_normal)
    return float(dots.mean())


def _roughness_index(pc: np.ndarray, k: int = 10) -> float:
    """Compute roughness index (local distance variance).
    
    High roughness indicates debris or irregular surfaces (damage).
    Low roughness indicates smooth, intact structural surfaces.
    
    Returns:
        Float >= 0. Higher = rougher surface.
    """
    if pc.shape[0] < k + 1:
        return 0.0
    tree = cKDTree(pc)
    dists, _ = tree.query(pc, k=min(k + 1, pc.shape[0]))
    # Skip self-distance (column 0)
    neighbor_dists = dists[:, 1:]
    # Roughness = mean of local distance standard deviations
    local_stds = neighbor_dists.std(axis=1)
    return float(local_stds.mean())


def _symmetry_score(pc: np.ndarray) -> float:
    """Compute bilateral symmetry score along the two principal horizontal axes.
    
    Intact buildings tend to be symmetric. Partially collapsed structures
    show asymmetry due to missing walls or tilted segments.
    
    Returns:
        Float in [0, 1]. Higher = more symmetric.
    """
    if pc.shape[0] < 10:
        return 0.0

    centered = pc - pc.mean(axis=0)

    # PCA to find principal axes
    try:
        cov = np.cov(centered.T)
        _, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return 0.0

    # Project onto principal axes
    projected = centered @ eigvecs

    # Symmetry score: for each axis, compare point distribution
    # on positive vs negative side
    scores = []
    for axis in range(2):  # Only horizontal axes
        positive = projected[projected[:, axis] > 0, axis]
        negative = -projected[projected[:, axis] < 0, axis]  # Mirror
        if len(positive) < 3 or len(negative) < 3:
            continue
        # Compare distributions via Wasserstein-like metric
        p_sorted = np.sort(positive)
        n_sorted = np.sort(negative)
        # Resample to same length
        n_common = min(len(p_sorted), len(n_sorted))
        p_resampled = np.interp(
            np.linspace(0, 1, n_common),
            np.linspace(0, 1, len(p_sorted)), p_sorted
        )
        n_resampled = np.interp(
            np.linspace(0, 1, n_common),
            np.linspace(0, 1, len(n_sorted)), n_sorted
        )
        # Score = 1 - normalized mean absolute difference
        diff = np.abs(p_resampled - n_resampled).mean()
        scale = max(np.abs(projected[:, axis]).max(), 1e-8)
        scores.append(max(0.0, 1.0 - diff / scale))

    return float(np.mean(scores)) if scores else 0.0


def _volume_estimate(pc: np.ndarray) -> float:
    """Estimate point cloud volume via convex hull.
    
    Collapsed buildings occupy less volume than intact ones (debris
    is compressed vertically).
    
    Returns:
        Volume in cubic units. 0.0 if hull computation fails.
    """
    if pc.shape[0] < 4:
        return 0.0
    try:
        hull = ConvexHull(pc)
        return float(hull.volume)
    except Exception:
        return 0.0


def compute_features(pc: np.ndarray, partial_pc_size: int = None) -> dict:
    """Compute comprehensive geometric features from a point cloud.
    
    Args:
        pc: (N, 3) numpy array of XYZ coordinates.
        partial_pc_size: Size of the original partial input (before completion).
            Used to compute completeness ratio.
    
    Returns:
        Dictionary of named features with float values.
    """
    pc = np.asarray(pc, dtype=np.float32)
    if pc.size == 0:
        return {}

    point_count = pc.shape[0]

    # ── Completeness Ratio ────────────────────────────────────
    if partial_pc_size is not None and partial_pc_size > 0:
        ratio = float(point_count) / float(partial_pc_size)
    else:
        ratio = float(point_count) / 2048.0
    completeness_ratio = min(1.0, ratio / 4.0)

    # ── Bounding Box & Height Statistics ──────────────────────
    zs = pc[:, 2]
    xs = pc[:, 0]
    ys = pc[:, 1]

    max_h = float(zs.max())
    min_h = float(zs.min())
    height_range = max_h - min_h
    hstd = float(zs.std())

    bbox_x = float(xs.max() - xs.min())
    bbox_y = float(ys.max() - ys.min())

    # Height percentiles
    h_p10 = _percentile(zs, 10)
    h_p25 = _percentile(zs, 25)
    h_p50 = _percentile(zs, 50)
    h_p75 = _percentile(zs, 75)
    h_p90 = _percentile(zs, 90)

    # ── Aspect Ratios ─────────────────────────────────────────
    footprint = max(bbox_x, bbox_y)
    aspect_ratio = height_range / max(footprint, 1e-6)

    # ── Top-Surface Planarity (PCA) ───────────────────────────
    z_th = _percentile(zs, 90)
    top_pts = pc[zs >= z_th]
    planarity_top = 0.0
    if len(top_pts) > 3:
        cov = np.cov(top_pts.T)
        eigenvalues, _ = np.linalg.eigh(cov)
        eigenvalues = np.sort(eigenvalues)
        if sum(eigenvalues) > 1e-6:
            planarity_top = (eigenvalues[1] - eigenvalues[0]) / sum(eigenvalues)

    # ── Vertical Density Distribution ─────────────────────────
    low_th = _percentile(zs, 33)
    high_th = _percentile(zs, 66)
    low = np.sum(zs <= low_th) / point_count
    mid = np.sum((zs > low_th) & (zs <= high_th)) / point_count
    high = np.sum(zs > high_th) / point_count

    # ── Advanced Features (PhD-level) ─────────────────────────
    # Surface normal estimation and consistency
    normals = _estimate_normals(pc, k=min(20, point_count - 1))
    normal_consistency = _normal_consistency(normals)

    # Roughness index
    roughness = _roughness_index(pc, k=min(10, point_count - 1))

    # Symmetry score
    symmetry = _symmetry_score(pc)

    # Volume estimation
    volume = _volume_estimate(pc)

    # ── Assemble Feature Dictionary ───────────────────────────
    features = {
        # Basic geometry
        "max_height": max_h,
        "min_height": min_h,
        "height_range": height_range,
        "height_std": hstd,
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "point_count": point_count,
        # Height percentiles
        "height_p10": h_p10,
        "height_p25": h_p25,
        "height_p50": h_p50,
        "height_p75": h_p75,
        "height_p90": h_p90,
        # Ratios
        "aspect_ratio": aspect_ratio,
        "completeness_ratio": completeness_ratio,
        # Planarity & density
        "planarity_top": planarity_top,
        "vertical_density_low": float(low),
        "vertical_density_mid": float(mid),
        "vertical_density_high": float(high),
        # Advanced structural descriptors
        "normal_consistency": normal_consistency,
        "roughness_index": roughness,
        "symmetry_score": symmetry,
        "volume_estimate": volume,
    }
    return features


if __name__ == "__main__":
    print("Testing geometric feature extraction...")
    # Create a simple building-like point cloud
    rng = np.random.RandomState(42)
    # Floor
    floor = rng.uniform(-5, 5, (200, 2))
    floor_z = np.zeros((200, 1))
    floor_pts = np.hstack([floor, floor_z])
    # Walls
    wall_pts = []
    for _ in range(400):
        side = rng.randint(4)
        h = rng.uniform(0, 10)
        if side == 0:
            wall_pts.append([-5, rng.uniform(-5, 5), h])
        elif side == 1:
            wall_pts.append([5, rng.uniform(-5, 5), h])
        elif side == 2:
            wall_pts.append([rng.uniform(-5, 5), -5, h])
        else:
            wall_pts.append([rng.uniform(-5, 5), 5, h])
    wall_pts = np.array(wall_pts)
    # Roof
    roof = rng.uniform(-5, 5, (200, 2))
    roof_z = np.full((200, 1), 10.0) + rng.normal(0, 0.1, (200, 1))
    roof_pts = np.hstack([roof, roof_z])

    pc = np.vstack([floor_pts, wall_pts, roof_pts]).astype(np.float32)
    feats = compute_features(pc, partial_pc_size=512)

    print(f"  Point count: {feats['point_count']}")
    print(f"  Height range: {feats['height_range']:.2f} m")
    print(f"  Planarity (top): {feats['planarity_top']:.3f}")
    print(f"  Normal consistency: {feats['normal_consistency']:.3f}")
    print(f"  Roughness index: {feats['roughness_index']:.4f}")
    print(f"  Symmetry score: {feats['symmetry_score']:.3f}")
    print(f"  Volume estimate: {feats['volume_estimate']:.1f}")
    print(f"  Aspect ratio: {feats['aspect_ratio']:.3f}")
    print(f"  Total features: {len(feats)}")
    print("  [OK] All features computed successfully!")
