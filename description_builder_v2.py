"""
Description Builder v2 — Structured Natural Language Payload
=============================================================
Translates computed geometric features into a structured natural-language
description for the LLM ReAct agent. Includes severity indicators with
literature-grounded thresholds for context.
"""


def build_description(features: dict) -> str:
    """Build a structured textual description from geometric features.
    
    Args:
        features: Dictionary from geometric_features.compute_features().
    
    Returns:
        Multi-line string describing the building's geometric state.
    """
    if not features:
        return "No point-cloud features available."

    lines = []
    lines.append("Building geometric analysis (post-completion):")

    # ── Basic Geometry ────────────────────────────────────────
    max_h = features.get("max_height", 0)
    height_range = features.get("height_range", 0)
    hstd = features.get("height_std", 0)
    lines.append(f"- Maximum height: {max_h:.2f} m")
    lines.append(f"- Height range: {height_range:.2f} m")
    lines.append(f"- Height variability (std): {hstd:.2f} m")

    # Height std interpretation
    if hstd < 1.0:
        lines.append("  -> Low height variability suggests collapsed or single-story structure")
    elif hstd > 3.0:
        lines.append("  -> High height variability consistent with intact multi-story building")

    # ── Footprint & Aspect Ratio ──────────────────────────────
    bbox_x = features.get("bbox_x", 0)
    bbox_y = features.get("bbox_y", 0)
    aspect = features.get("aspect_ratio", 0)
    lines.append(f"- Footprint dimensions: {bbox_x:.2f} x {bbox_y:.2f} m")
    lines.append(f"- Height/footprint aspect ratio: {aspect:.3f}")

    if aspect < 0.3:
        lines.append("  -> Very low aspect ratio may indicate partial height loss (collapse)")

    # ── Planarity ─────────────────────────────────────────────
    pt = features.get("planarity_top", 0.0)
    lines.append(f"- Top-surface planarity: {pt:.3f}")

    if pt > 0.3:
        lines.append("  -> High planarity suggests intact roof surface")
    elif pt < 0.1:
        lines.append("  -> Low planarity indicates irregular top surface (potential roof damage)")

    # ── Vertical Density Distribution ─────────────────────────
    low = features.get("vertical_density_low", 0.0)
    mid = features.get("vertical_density_mid", 0.0)
    high = features.get("vertical_density_high", 0.0)
    lines.append(f"- Point density distribution (low/mid/high): {low:.2f}/{mid:.2f}/{high:.2f}")

    if low > 0.50:
        lines.append("  -> Concentration at base level suggests collapsed upper structure")
    elif high > 0.40:
        lines.append("  -> Strong upper-level presence suggests roof/upper floors intact")

    # ── Completeness Ratio ────────────────────────────────────
    comp = features.get("completeness_ratio", 0.0)
    lines.append(f"- Completeness ratio: {comp:.3f}")

    # ── Advanced Structural Descriptors ───────────────────────
    nc = features.get("normal_consistency", None)
    if nc is not None:
        lines.append(f"- Surface normal consistency: {nc:.3f}")
        if nc > 0.7:
            lines.append("  -> High normal consistency indicates smooth, intact structural surfaces")
        elif nc < 0.4:
            lines.append("  -> Low normal consistency suggests fragmented or rubble-like geometry")

    ri = features.get("roughness_index", None)
    if ri is not None:
        lines.append(f"- Roughness index: {ri:.4f}")
        if ri > 0.5:
            lines.append("  -> High roughness consistent with debris field or severe structural damage")
        elif ri < 0.1:
            lines.append("  -> Low roughness consistent with clean structural surfaces")

    sym = features.get("symmetry_score", None)
    if sym is not None:
        lines.append(f"- Bilateral symmetry score: {sym:.3f}")
        if sym > 0.7:
            lines.append("  -> High symmetry consistent with structurally intact building")
        elif sym < 0.4:
            lines.append("  -> Low symmetry suggests asymmetric damage (partial wall collapse)")

    vol = features.get("volume_estimate", None)
    if vol is not None:
        lines.append(f"- Convex hull volume: {vol:.1f} m³")

    # ── Height Percentiles ────────────────────────────────────
    h_p50 = features.get("height_p50", None)
    h_p90 = features.get("height_p90", None)
    if h_p50 is not None and h_p90 is not None:
        lines.append(f"- Height percentiles (P50/P90): {h_p50:.2f}/{h_p90:.2f} m")

    return "\n".join(lines)
