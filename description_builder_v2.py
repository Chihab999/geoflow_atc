"""
Description Builder v2 — Structured Natural Language Payload
=============================================================
Translates computed geometric features into a structured natural-language
description for the LLM ReAct agent. Includes severity indicators with
literature-grounded thresholds for context.
"""

def build_description(features: dict, partial_features: dict = None) -> str:
    """Build a structured textual description translated for ATC-20 reasoning.
    
    Args:
        features: Dictionary from geometric_features.compute_features() on completed point cloud.
        partial_features: Dictionary of features on the partial point cloud (ground truth geometry).
                          If provided, this is used for the ATC-20 geometric assessment to avoid completion artifacts.
    """
    if not features:
        return "No point-cloud features available."

    # Use partial features as the "ground truth" for geometric assessment if available
    assess_feats = partial_features if partial_features else features

    lines = []
    lines.append("DAMAGE INDICATORS (Geometric Analysis):")

    # 1. Categorize height_range
    hr = assess_feats.get("height_range", 0)
    if hr < 3.0:
        lines.append(f"- Vertical extent: {hr:.2f}m — BUILDING COLLAPSED (intact buildings have 15-50m range)")
    elif hr < 15.0:
        lines.append(f"- Vertical extent: {hr:.2f}m — Reduced vertical extent — possible partial collapse or low building")
    else:
        lines.append(f"- Vertical extent: {hr:.2f}m — Full building height preserved (Green/Yellow indicator)")

    # 2. Categorize height_std
    hstd = assess_feats.get("height_std", 0)
    if hstd < 1.0:
        lines.append(f"- Surface variation: {hstd:.2f} — Geometry is FLAT (strong collapse indicator)")
    elif hstd < 4.0:
        lines.append(f"- Surface variation: {hstd:.2f} — Moderate vertical variation — possible damage")
    else:
        lines.append(f"- Surface variation: {hstd:.2f} — Normal vertical variation — intact building structure")

    # 3. Categorize roughness_index
    ri = assess_feats.get("roughness_index", 3.0)
    if ri < 1.5:
        lines.append(f"- Roughness: {ri:.2f} — Smooth surface — intact roof")
    elif ri <= 3.0:
        lines.append(f"- Roughness: {ri:.2f} — Moderate roughness — minor damage or normal texture")
    else:
        lines.append(f"- Roughness: {ri:.2f} — High roughness — possible damaged surface OR sparse sampling artifact")

    nc = assess_feats.get("normal_consistency", 0.0)
    vol = assess_feats.get("volume_estimate", 0.0)

    # 4. Decision Boundary Crossings
    lines.append("\nINTACT-BUILDING SIGNATURE CHECK:")
    lines.append(f"[{'✓' if hr > 15.0 else '✗'}] Height range > 15m (indicates full structure): {hr:.2f}m")
    lines.append(f"[{'✓' if hstd > 4.0 else '✗'}] Height std > 4.0 (indicates vertical complexity): {hstd:.2f}")
    lines.append(f"[{'✓' if 0.85 <= nc <= 0.97 else '✗'}] Normal consistency in 0.85-0.97 range (intact roof, not debris): {nc:.2f}")
    lines.append(f"[{'✓' if ri < 3.0 else '✗'}] Roughness < 3.0 (smooth surface, not rubble): {ri:.2f}")

    lines.append("\nDEBRIS-FIELD SIGNATURE CHECK:")
    lines.append(f"[{'✓' if hr < 3.0 else '✗'}] Height range < 3m (flat geometry): {hr:.2f}m")
    lines.append(f"[{'✓' if hstd < 1.0 else '✗'}] Height std < 1.0 (collapsed): {hstd:.2f}")
    lines.append(f"[{'✓' if nc > 0.99 else '✗'}] Normal consistency > 0.99 (flat surface): {nc:.2f}")
    lines.append(f"[{'✓' if vol < 50000 else '✗'}] Volume estimate < 50000 (very small enclosed volume): {vol:.1f}")

    # 5. Preliminary Assessment Summary
    lines.append("\nGEOMETRIC EVIDENCE SUMMARY:")
    
    collapse_indicator = "STRONG" if (hr < 3.0 and hstd < 1.0) else ("MODERATE" if hr < 15.0 else "ABSENT")
    intact_indicator = "STRONG" if (hr > 15.0 and hstd > 4.0) else "ABSENT"
    
    lines.append(f"- Vertical collapse indicator: {collapse_indicator} (height_range = {hr:.2f}m)")
    lines.append(f"- Structural intactness indicator: {intact_indicator} (height_std = {hstd:.2f})")
    
    if nc > 0.99:
        lines.append(f"- Surface uniformity: extreme (normal_consistency = {nc:.4f}, consistent with flat debris)")
    elif nc > 0.85:
        lines.append(f"- Surface uniformity: normal (normal_consistency = {nc:.4f}, consistent with intact roof)")
"""
Description Builder v2 — Structured Natural Language Payload
=============================================================
Translates computed geometric features into a structured natural-language
description for the LLM ReAct agent. Includes severity indicators with
literature-grounded thresholds for context.
"""

def build_description(features: dict, partial_features: dict = None) -> str:
    """Build a structured textual description translated for ATC-20 reasoning.
    
    Args:
        features: Dictionary from geometric_features.compute_features() on completed point cloud.
        partial_features: Dictionary of features on the partial point cloud (ground truth geometry).
                          If provided, this is used for the ATC-20 geometric assessment to avoid completion artifacts.
    """
    if not features:
        return "No point-cloud features available."

    # Use partial features as the "ground truth" for geometric assessment if available
    assess_feats = partial_features if partial_features else features

    lines = []
    lines.append("DAMAGE INDICATORS (Geometric Analysis):")

    # 1. Categorize height_range
    hr = assess_feats.get("height_range", 0)
    if hr < 3.0:
        lines.append(f"- Vertical extent: {hr:.2f}m — BUILDING COLLAPSED (intact buildings have 15-50m range)")
    elif hr < 15.0:
        lines.append(f"- Vertical extent: {hr:.2f}m — Reduced vertical extent — possible partial collapse or low building")
    else:
        lines.append(f"- Vertical extent: {hr:.2f}m — Full building height preserved (Green/Yellow indicator)")

    # 2. Categorize height_std
    hstd = assess_feats.get("height_std", 0)
    if hstd < 1.0:
        lines.append(f"- Surface variation: {hstd:.2f} — Geometry is FLAT (strong collapse indicator)")
    elif hstd < 4.0:
        lines.append(f"- Surface variation: {hstd:.2f} — Moderate vertical variation — possible damage")
    else:
        lines.append(f"- Surface variation: {hstd:.2f} — Normal vertical variation — intact building structure")

    # 3. Categorize roughness_index
    ri = assess_feats.get("roughness_index", 3.0)
    if ri < 1.5:
        lines.append(f"- Roughness: {ri:.2f} — Smooth surface — intact roof")
    elif ri <= 3.0:
        lines.append(f"- Roughness: {ri:.2f} — Moderate roughness — minor damage or normal texture")
    else:
        lines.append(f"- Roughness: {ri:.2f} — High roughness — possible damaged surface OR sparse sampling artifact")

    nc = assess_feats.get("normal_consistency", 0.0)
    vol = assess_feats.get("volume_estimate", 0.0)

    # 4. Decision Boundary Crossings
    lines.append("\nINTACT-BUILDING SIGNATURE CHECK:")
    lines.append(f"[{'✓' if hr > 15.0 else '✗'}] Height range > 15m (indicates full structure): {hr:.2f}m")
    lines.append(f"[{'✓' if hstd > 4.0 else '✗'}] Height std > 4.0 (indicates vertical complexity): {hstd:.2f}")
    lines.append(f"[{'✓' if 0.85 <= nc <= 0.97 else '✗'}] Normal consistency in 0.85-0.97 range (intact roof, not debris): {nc:.2f}")
    lines.append(f"[{'✓' if ri < 3.0 else '✗'}] Roughness < 3.0 (smooth surface, not rubble): {ri:.2f}")

    lines.append("\nDEBRIS-FIELD SIGNATURE CHECK:")
    lines.append(f"[{'✓' if hr < 3.0 else '✗'}] Height range < 3m (flat geometry): {hr:.2f}m")
    lines.append(f"[{'✓' if hstd < 1.0 else '✗'}] Height std < 1.0 (collapsed): {hstd:.2f}")
    lines.append(f"[{'✓' if nc > 0.99 else '✗'}] Normal consistency > 0.99 (flat surface): {nc:.2f}")
    lines.append(f"[{'✓' if vol < 50000 else '✗'}] Volume estimate < 50000 (very small enclosed volume): {vol:.1f}")

    # 5. Preliminary Assessment Summary
    lines.append("\nGEOMETRIC EVIDENCE SUMMARY:")
    
    collapse_indicator = "STRONG" if (hr < 3.0 and hstd < 1.0) else ("MODERATE" if hr < 15.0 else "ABSENT")
    intact_indicator = "STRONG" if (hr > 15.0 and hstd > 4.0) else "ABSENT"
    
    lines.append(f"- Vertical collapse indicator: {collapse_indicator} (height_range = {hr:.2f}m)")
    lines.append(f"- Structural intactness indicator: {intact_indicator} (height_std = {hstd:.2f})")
    
    if nc > 0.99:
        lines.append(f"- Surface uniformity: extreme (normal_consistency = {nc:.4f}, consistent with flat debris)")
    elif nc > 0.85:
        lines.append(f"- Surface uniformity: normal (normal_consistency = {nc:.4f}, consistent with intact roof)")
    else:
        lines.append(f"- Surface uniformity: low (normal_consistency = {nc:.4f}, highly irregular)")

    lines.append("\nThese indicators provide characteristic structural context.")
    lines.append("The agent should consult ATC-20 placard criteria and verify against the retrieved text.")

    # ── Height Percentiles & Completeness ────────────────────────────────────
    h_p50 = features.get("height_p50", None)
    h_p90 = features.get("height_p90", None)
    if h_p50 is not None and h_p90 is not None:
        lines.append(f"- Height percentiles (P50/P90): {h_p50:.2f}/{h_p90:.2f} m")
        
    comp = features.get("completeness_ratio", 1.0)
    lines.append(f"- Completeness ratio: {comp:.2f} (1.0 = fully intact/completed, < 1.0 = missing sections)")

    return "\n".join(lines)
