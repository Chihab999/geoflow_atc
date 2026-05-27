"""
Centralized Configuration for the GeoFlow-PC + ReAct Agent Pipeline.
=====================================================================
All pipeline-wide constants, paths, and hyperparameters are defined here.
Import this module instead of scattering magic strings across scripts.

Usage:
    from config import PipelineConfig, LABEL_MAP, SEVERITY_BRANCHES
    cfg = PipelineConfig()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import json


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# ATC-20 placard classes (ground-truth label vocabulary)
PLACARD_CLASSES = ["Green", "Yellow", "Red"]
LABEL_MAP = {"Green": 0, "Yellow": 1, "Red": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

# Damage severity branches for the router
SEVERITY_BRANCHES = {0: "Intact", 1: "Moderate", 2: "Severe"}

# Project root (resolved relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────
# Pipeline Configuration Dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Central configuration for the full triage pipeline."""

    # ── Paths ──────────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    checkpoint_path: str = "best_v8.pth"
    router_weights: str = "data/router_weights.pth"
    kb_index_dir: str = "kb_index"
    kb_documents_dir: str = "kb_documents"
    data_dir: str = "data"
    results_dir: str = "results"
    logs_dir: str = "logs"

    # ── Point Cloud ────────────────────────────────────────────
    n_partial: int = 512
    n_complete: int = 2048

    # ── Router ─────────────────────────────────────────────────
    router_input_dim: int = 384  # Matches GeoFlow d_model & PointNet output
    router_hidden_dim: int = 128
    router_num_classes: int = 3

    # ── LLM / Agent ────────────────────────────────────────────
    api_base: str = "http://localhost:11434/v1"
    model_name: str = "qwen2.5"
    api_key: str = "local-key"
    max_react_steps: int = 16
    llm_temperature: float = 0.1
    llm_max_tokens: int = 300

    # ── Evidence / Verification ────────────────────────────────
    evidence_overlap_threshold: float = 0.40
    evidence_keyword_overlap_threshold: float = 0.15
    kb_retrieval_top_k: int = 5
    verifier_revision_patience: int = 3
    confidence_threshold: float = 0.70

    # ── Evaluation ─────────────────────────────────────────────
    random_seed: int = 42
    bootstrap_n_iterations: int = 1000
    bootstrap_confidence_level: float = 0.95
    f_score_threshold: float = 0.01  # For point cloud F-Score

    # ── Ablation Architectures ─────────────────────────────────
    ablation_architectures: List[str] = field(
        default_factory=lambda: ["NoCompletion", "Single-Branch", "GeoFlow+Router", "PoinTr"]
    )

    def resolve_path(self, relative: str) -> Path:
        """Resolve a relative path against the project root."""
        p = self.project_root / relative
        if p.exists():
            return p
        # Fallback: check parent directory (for shared assets like checkpoints)
        parent_p = self.project_root.parent / relative
        if parent_p.exists():
            return parent_p
        return p  # Return original (caller handles missing)

    def get_checkpoint_path(self) -> Path:
        """Resolve checkpoint path with fallback to parent directory."""
        return self.resolve_path(self.checkpoint_path)

    def get_router_weights_path(self) -> Path:
        """Resolve router weights path."""
        return self.resolve_path(self.router_weights)

    def get_kb_index_dir(self) -> Path:
        """Resolve KB index directory."""
        return self.project_root / self.kb_index_dir

    def get_kb_documents_dir(self) -> Path:
        """Resolve KB documents directory."""
        return self.project_root / self.kb_documents_dir

    def get_results_dir(self) -> Path:
        """Resolve and create results directory."""
        p = self.project_root / self.results_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_logs_dir(self) -> Path:
        """Resolve and create logs directory."""
        p = self.project_root / self.logs_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def to_dict(self) -> dict:
        """Serialize config to a JSON-safe dictionary for provenance logging."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, list):
                d[k] = v
            else:
                d[k] = v
        return d

    def save(self, path: Optional[str] = None):
        """Save configuration to JSON."""
        out = Path(path) if path else self.get_results_dir() / "pipeline_config.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ─────────────────────────────────────────────────────────────
# Triage Result Dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class TriageResult:
    """Structured output from the ReAct triage agent."""
    predicted_class: str = "INCONCLUSIVE"
    confidence: float = 0.0
    citations: List[str] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    raw_output: str = ""
    num_steps: int = 0
    latency_seconds: float = 0.0
    log_path: str = ""
    is_refused: bool = False
    refuse_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "citations": self.citations,
            "reasoning_trace": self.reasoning_trace,
            "raw_output": self.raw_output,
            "num_steps": self.num_steps,
            "latency_seconds": self.latency_seconds,
            "log_path": self.log_path,
            "is_refused": self.is_refused,
            "refuse_reason": self.refuse_reason,
        }


if __name__ == "__main__":
    cfg = PipelineConfig()
    print("Pipeline Configuration:")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    print(f"\nCheckpoint: {cfg.get_checkpoint_path()}")
    print(f"Router weights: {cfg.get_router_weights_path()}")
