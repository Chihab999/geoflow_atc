# GeoFlow-ATC: Autonomous Structural Triage via Point Cloud Completion and Agentic Reasoning

> **A novel pipeline fusing 3D geometric deep learning with LLM-based ReAct reasoning for post-earthquake ATC-20 building safety classification.**

---

## Abstract

This repository implements **GeoFlow-ATC**, an autonomous structural triage system that processes partial 3D LiDAR point clouds of earthquake-damaged buildings and produces ATC-20 safety placard classifications (Green / Yellow / Red) with full explainability. The system combines:

1. **GeoFlow-PC v8** — A neural point cloud completion network (Multi-scale DGCNN encoder + Transformer decoder + Folding refinement) that reconstructs missing geometry from partial scans.
2. **Damage Severity Router** — A PointNet-based classifier that routes inputs to appropriate completion branches based on inferred damage severity.
3. **TriageReActAgent** — A ReAct (Reasoning and Acting) LLM agent grounded in an ATC-20 knowledge base, with evidence verification sub-agents to prevent hallucination in safety-critical decisions.

The pipeline is evaluated via a comprehensive ablation study across 4 architectures, with publication-grade metrics including per-class F1, Cohen's κ, MCC, bootstrap confidence intervals, and McNemar's statistical significance tests.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    GeoFlow-ATC Pipeline                              │
│                                                                      │
│  ┌─────────┐    ┌──────────┐    ┌───────────┐    ┌──────────────┐   │
│  │ Partial  │───▶│ PointNet │───▶│  Damage   │───▶│  GeoFlow-PC  │   │
│  │ 3D Scan  │    │ Encoder  │    │  Router   │    │  Completion  │   │
│  └─────────┘    └──────────┘    └───────────┘    └──────┬───────┘   │
│                                    │                     │           │
│                              ┌─────┴─────┐               │           │
│                              │ Branch A/B/C│               │           │
│                              └───────────┘               ▼           │
│                                               ┌──────────────────┐   │
│                                               │ Geometric Feature│   │
│                                               │   Extraction     │   │
│                                               └────────┬─────────┘   │
│                                                        │             │
│  ┌──────────────────┐    ┌────────────┐               ▼             │
│  │  ATC-20 / FEMA   │───▶│   FAISS +  │    ┌──────────────────┐    │
│  │  Knowledge Base   │    │   BM25     │───▶│  ReAct LLM Agent │    │
│  └──────────────────┘    │   Hybrid   │    │  (Triage Engine) │    │
│                          │   Search   │    └────────┬─────────┘    │
│                          └────────────┘             │              │
│                                                     ▼              │
│                                           ┌──────────────────┐     │
│                                           │  ATC-20 Placard  │     │
│                                           │  Classification  │     │
│                                           │  + Confidence    │     │
│                                           │  + Citations     │     │
│                                           └──────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Module | File | Description |
|--------|------|-------------|
| Configuration | `config.py` | Centralized pipeline configuration via dataclasses |
| GeoFlow-PC v8 | `geoflow_model.py` | Multi-scale DGCNN + Transformer decoder architecture |
| GeoFlow Wrapper | `geoflow_integration.py` | Inference wrapper with normalization/denormalization |
| PointNet Encoder | `pointnet_encoder.py` | Feature extraction for damage severity routing |
| Damage Router | `damage_router.py` | 4-layer MLP routing to completion branches |
| Geometric Features | `geometric_features.py` | 22 structural descriptors (normals, roughness, symmetry) |
| Description Builder | `description_builder_v2.py` | Feature → natural language for LLM consumption |
| Knowledge Base | `unified_retrieval.py` | Hybrid FAISS + BM25 retrieval with re-ranking |
| ReAct Agent | `reasoning_agent.py` | KB-driven triage with evidence verification |
| Evaluation | `evaluation.py` | Metrics, bootstrap CI, McNemar's test, LaTeX tables |
| Ablation Study | `run_ablation_final.py` | Full orchestration across 4 architectures |

---

## Installation

### Prerequisites
- Python ≥ 3.9
- PyTorch ≥ 2.0 (CUDA optional but recommended)
- Ollama with `qwen2.5` model pulled

### Setup

```bash
# Clone and navigate to the project
cd paper2_option2_FINAL

# Install dependencies
pip install -r requirements.txt

# Pull the LLM model (required for agent inference)
ollama pull qwen2.5

# Build the knowledge base index (from raw ATC-20 documents)
python build_kb_index.py

# Build the scene index (from point cloud data)
python build_scene_index.py
```

---

## Quick Start

### 1. Verify Setup (30 seconds)

```bash
python diagnostic_checks.py
```

This validates damage simulation, feature extraction, router, and all imports.

### 2. Run Single Scene

```bash
python pipeline_integrated.py
```

### 3. Run Full Ablation Study

```bash
python run_ablation_final.py
```

Results are saved to `results/ablation_run/`:
- `ablation_orchestration_results.json` — Raw per-scene results
- `ablation_metrics.json` — Comprehensive metrics with bootstrap CIs
- `ablation_table.tex` — LaTeX table ready for paper insertion
- `provenance.json` — Full reproducibility metadata

---

## Evaluation Protocol

### Ablation Architectures

| Architecture | Completion | Routing | Description |
|-------------|-----------|---------|-------------|
| NoCompletion | ✗ | ✗ | Raw partial cloud baseline |
| Single-Branch | ✓ (GeoFlow) | ✗ | Always uses Branch A |
| **GeoFlow+Router** | ✓ (GeoFlow) | ✓ | **Proposed: severity-adaptive** |
| PoinTr | ✓ (proxy) | ✗ | Alternative completion baseline |

### Metrics Computed

- **Per-class**: Precision, Recall, F1-Score (for Green, Yellow, Red)
- **Aggregate**: Macro-F1, Weighted-F1, Accuracy
- **Agreement**: Cohen's κ (inter-rater reliability)
- **Correlation**: Matthews Correlation Coefficient (MCC)
- **Confidence**: 95% Bootstrap Confidence Intervals (1000 iterations)
- **Significance**: McNemar's test (pairwise architecture comparison)

---

## Knowledge Base

The system is grounded in official structural engineering criteria:
- **ATC-20**: Post-earthquake safety evaluation procedures
- **FEMA P-154**: Rapid visual screening for potential seismic hazards

Documents are stored in `kb_documents/` and indexed into FAISS vectors via `build_kb_index.py`. The retrieval system uses hybrid dense (semantic) + sparse (BM25) search for superior recall on both paraphrased queries and exact ATC-20 terminology.

---

## Geometric Features

The pipeline extracts 22 structural descriptors from point clouds:

| Feature | Type | Damage Indicator |
|---------|------|------------------|
| Height range / std | Basic | Collapsed = low range |
| Planarity (top) | PCA | Intact roof = high planarity |
| Normal consistency | Surface | Rubble = low consistency |
| Roughness index | Local | Debris = high roughness |
| Symmetry score | Global | Wall collapse = asymmetry |
| Volume estimate | Hull | Collapse = less volume |
| Vertical density | Distribution | Collapse = base-heavy |
| Aspect ratio | Ratio | Tilt = unusual ratio |

---

## Project Structure

```
paper2_option2_FINAL/
├── config.py                    # Centralized configuration
├── geoflow_model.py             # GeoFlow-PC v8 architecture
├── geoflow_integration.py       # GeoFlow inference wrapper
├── pointnet_encoder.py          # PointNet feature encoder
├── damage_router.py             # Severity routing MLP
├── damage_simulation.py         # Earthquake damage simulation
├── geometric_features.py        # 22 structural descriptors
├── description_builder_v2.py    # Feature → NL description
├── unified_retrieval.py         # Hybrid FAISS + BM25 retrieval
├── reasoning_agent.py           # ReAct triage agent
├── pipeline_integrated.py       # End-to-end pipeline
├── evaluation.py                # Publication-grade metrics
├── run_ablation_final.py        # Full ablation orchestration
├── train_router.py              # Router training pipeline
├── train_lora_adapters.py       # LoRA fine-tuning (experimental)
├── build_kb_index.py            # KB index builder
├── build_scene_index.py         # Scene index builder
├── diagnostic_checks.py         # Component verification
├── requirements.txt             # Pinned dependencies
├── data/
│   ├── vaihingen_scenes9/       # 9 Vaihingen test scenes (.pts)
│   ├── router_weights.pth       # Trained router weights
│   └── scene_index.json         # Scene file index
├── kb_documents/
│   ├── ATC_20_Criteria.txt      # ATC-20 criteria manual
│   └── FEMA_P154_*.pdf          # FEMA rapid screening guide
├── kb_index/
│   ├── text_kb.faiss            # FAISS vector index
│   └── metadata.json            # Chunk metadata
├── results/
│   └── ablation_run/            # Ablation study outputs
└── logs/                        # Agent reasoning traces (JSON)
```

---

## Citation

```bibtex
@article{geoflow_atc_2026,
  title={GeoFlow-ATC: Autonomous Structural Triage via Point Cloud 
         Completion and Agentic Reasoning},
  author={},
  journal={},
  year={2026},
  note={Combining 3D geometric deep learning with LLM-based ReAct 
        reasoning for post-earthquake ATC-20 building safety 
        classification}
}
```

---

## License

This project is provided for academic research purposes. See the parent repository for licensing details.
