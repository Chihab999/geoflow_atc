"""
Evaluation Module — Publication-Grade Metrics & Statistical Tests
==================================================================
Computes comprehensive classification metrics for the ATC-20 triage
pipeline ablation study.

Metrics:
  - Per-class Precision, Recall, F1-Score (macro & weighted)
  - Overall Accuracy
  - Confusion Matrix (3×3: Green/Yellow/Red)
  - Cohen's Kappa (inter-rater reliability)
  - Matthews Correlation Coefficient (MCC)
  - Bootstrap Confidence Intervals (95% CI for all metrics)
  - McNemar's Test (pairwise architecture comparison)
  - LaTeX table generation for direct paper insertion

Usage:
    from evaluation import Evaluator
    ev = Evaluator(class_names=["Green", "Yellow", "Red"])
    ev.add_predictions(y_true, y_pred, architecture="GeoFlow+Router")
    report = ev.compute_all_metrics()
    ev.save_report("results/ablation_metrics.json")
    ev.to_latex("results/ablation_table.tex")
"""

import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# Core Metrics (no sklearn dependency)
# ─────────────────────────────────────────────────────────────

def confusion_matrix(y_true: List[int], y_pred: List[int], n_classes: int = 3) -> np.ndarray:
    """Compute confusion matrix. Rows = true, Cols = predicted."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t][p] += 1
    return cm


def precision_recall_f1(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-class and aggregated metrics from confusion matrix."""
    n_classes = cm.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    support = cm.sum(axis=1)

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision[i] = tp / max(tp + fp, 1)
        recall[i] = tp / max(tp + fn, 1)
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    # Macro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    # Weighted averages
    total = support.sum()
    if total > 0:
        weighted_precision = (precision * support).sum() / total
        weighted_recall = (recall * support).sum() / total
        weighted_f1 = (f1 * support).sum() / total
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    return {
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "support": support,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


def accuracy(cm: np.ndarray) -> float:
    """Overall accuracy from confusion matrix."""
    total = cm.sum()
    return float(np.trace(cm) / max(total, 1))


def cohens_kappa(cm: np.ndarray) -> float:
    """Cohen's Kappa — measures inter-rater agreement beyond chance.
    
    Kappa = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement, p_e = expected agreement.
    """
    total = cm.sum()
    if total == 0:
        return 0.0
    p_o = np.trace(cm) / total
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    p_e = (row_sums * col_sums).sum() / (total ** 2)
    if p_e == 1.0:
        return 1.0
    return float((p_o - p_e) / (1 - p_e))


def matthews_corrcoef(cm: np.ndarray) -> float:
    """Matthews Correlation Coefficient for multi-class.
    
    MCC ranges from -1 (total disagreement) to +1 (perfect prediction).
    Recommended for imbalanced multi-class classification.
    """
    total = cm.sum()
    if total == 0:
        return 0.0

    # Convert to float for numerical stability
    cm_f = cm.astype(np.float64)
    t_k = cm_f.sum(axis=1)  # row sums (true class counts)
    p_k = cm_f.sum(axis=0)  # col sums (predicted class counts)

    c = np.trace(cm_f)  # correctly classified
    s = total

    # MCC for multi-class (Gorodkin, 2004)
    numerator = c * s - np.dot(t_k, p_k)
    denominator = np.sqrt(s ** 2 - np.dot(p_k, p_k)) * np.sqrt(s ** 2 - np.dot(t_k, t_k))

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


# ─────────────────────────────────────────────────────────────
# Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────────

def bootstrap_ci(y_true: List[int], y_pred: List[int],
                 metric_fn, n_iterations: int = 1000,
                 confidence: float = 0.95, seed: int = 42,
                 n_classes: int = 3) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a classification metric.
    
    Args:
        y_true, y_pred: Ground-truth and predicted labels.
        metric_fn: Function(confusion_matrix) -> float.
        n_iterations: Number of bootstrap samples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.
        n_classes: Number of classes.
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    # Point estimate
    cm_full = confusion_matrix(y_true.tolist(), y_pred.tolist(), n_classes)
    point = metric_fn(cm_full)

    # Bootstrap
    scores = []
    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        cm_boot = confusion_matrix(y_true[idx].tolist(), y_pred[idx].tolist(), n_classes)
        scores.append(metric_fn(cm_boot))

    alpha = 1 - confidence
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))

    return (float(point), lower, upper)


# ─────────────────────────────────────────────────────────────
# McNemar's Test (Pairwise Architecture Comparison)
# ─────────────────────────────────────────────────────────────

def mcnemar_test(y_true: List[int], y_pred_a: List[int],
                 y_pred_b: List[int]) -> Dict[str, float]:
    """McNemar's test for comparing two classifiers.
    
    Tests whether the two classifiers have the same error rate.
    Uses the chi-squared approximation with continuity correction.
    
    Returns:
        {"b": b_count, "c": c_count, "chi2": statistic, "p_value": p_value}
    """
    # b = A correct, B wrong;  c = A wrong, B correct
    b = sum(1 for t, a, bb in zip(y_true, y_pred_a, y_pred_b) if a == t and bb != t)
    c = sum(1 for t, a, bb in zip(y_true, y_pred_a, y_pred_b) if a != t and bb == t)

    # Chi-squared with continuity correction
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # Approximate p-value using chi2 distribution with 1 df
    # Using the survival function approximation
    p_value = _chi2_sf(chi2, df=1)

    return {"b": b, "c": c, "chi2": float(chi2), "p_value": float(p_value)}


def _chi2_sf(x, df=1):
    """Survival function for chi-squared distribution (approximate).
    Uses the incomplete gamma function approximation.
    """
    if x <= 0:
        return 1.0
    # For df=1: P(X > x) ≈ erfc(sqrt(x/2))
    from math import erfc, sqrt
    return erfc(sqrt(x / 2))


# ─────────────────────────────────────────────────────────────
# Evaluator Class
# ─────────────────────────────────────────────────────────────

@dataclass
class ArchitectureResults:
    """Stores predictions and metrics for one architecture."""
    name: str
    y_true: List[int] = field(default_factory=list)
    y_pred: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


class Evaluator:
    """Comprehensive evaluation engine for the triage pipeline.
    
    Usage:
        ev = Evaluator(class_names=["Green", "Yellow", "Red"])
        ev.add_predictions(y_true, y_pred, architecture="GeoFlow+Router",
                          confidences=[0.9, 0.7, ...], latencies=[1.2, 0.8, ...])
        report = ev.compute_all_metrics()
        ev.save_report("results/metrics.json")
        print(ev.to_latex())
    """

    def __init__(self, class_names: List[str] = None,
                 n_bootstrap: int = 1000,
                 confidence_level: float = 0.95,
                 seed: int = 42):
        self.class_names = class_names or ["Green", "Yellow", "Red"]
        self.n_classes = len(self.class_names)
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.seed = seed
        self.architectures: Dict[str, ArchitectureResults] = {}

    def add_predictions(self, y_true: List[int], y_pred: List[int],
                        architecture: str,
                        confidences: List[float] = None,
                        latencies: List[float] = None):
        """Register predictions for an architecture."""
        ar = ArchitectureResults(name=architecture)
        ar.y_true = list(y_true)
        ar.y_pred = list(y_pred)
        ar.confidences = list(confidences or [])
        ar.latencies = list(latencies or [])
        self.architectures[architecture] = ar

    def compute_all_metrics(self) -> Dict:
        """Compute all metrics for all registered architectures."""
        report = {}

        for arch_name, ar in self.architectures.items():
            cm = confusion_matrix(ar.y_true, ar.y_pred, self.n_classes)
            prf = precision_recall_f1(cm)
            acc = accuracy(cm)
            kappa = cohens_kappa(cm)
            mcc = matthews_corrcoef(cm)

            # Bootstrap CIs for key metrics
            acc_ci = bootstrap_ci(ar.y_true, ar.y_pred, accuracy,
                                  self.n_bootstrap, self.confidence_level, self.seed, self.n_classes)
            kappa_ci = bootstrap_ci(ar.y_true, ar.y_pred, cohens_kappa,
                                    self.n_bootstrap, self.confidence_level, self.seed, self.n_classes)
            mcc_ci = bootstrap_ci(ar.y_true, ar.y_pred, matthews_corrcoef,
                                  self.n_bootstrap, self.confidence_level, self.seed, self.n_classes)
            f1_ci = bootstrap_ci(ar.y_true, ar.y_pred,
                                 lambda cm_: precision_recall_f1(cm_)["macro_f1"],
                                 self.n_bootstrap, self.confidence_level, self.seed, self.n_classes)

            metrics = {
                "confusion_matrix": cm.tolist(),
                "accuracy": acc,
                "accuracy_95ci": list(acc_ci),
                "macro_f1": prf["macro_f1"],
                "macro_f1_95ci": list(f1_ci),
                "weighted_f1": prf["weighted_f1"],
                "cohens_kappa": kappa,
                "cohens_kappa_95ci": list(kappa_ci),
                "mcc": mcc,
                "mcc_95ci": list(mcc_ci),
                "per_class": {},
                "n_samples": len(ar.y_true),
            }

            for i, cls_name in enumerate(self.class_names):
                metrics["per_class"][cls_name] = {
                    "precision": float(prf["per_class_precision"][i]),
                    "recall": float(prf["per_class_recall"][i]),
                    "f1": float(prf["per_class_f1"][i]),
                    "support": int(prf["support"][i]),
                }

            # Timing statistics
            if ar.latencies:
                metrics["latency_mean_s"] = float(np.mean(ar.latencies))
                metrics["latency_std_s"] = float(np.std(ar.latencies))
                metrics["latency_median_s"] = float(np.median(ar.latencies))

            # Confidence calibration
            if ar.confidences:
                metrics["mean_confidence"] = float(np.mean(ar.confidences))
                metrics["confidence_std"] = float(np.std(ar.confidences))

            ar.metrics = metrics
            report[arch_name] = metrics

        # Pairwise McNemar's tests
        arch_names = list(self.architectures.keys())
        mcnemar_results = {}
        for i in range(len(arch_names)):
            for j in range(i + 1, len(arch_names)):
                a, b = arch_names[i], arch_names[j]
                ar_a = self.architectures[a]
                ar_b = self.architectures[b]
                if len(ar_a.y_true) == len(ar_b.y_true):
                    result = mcnemar_test(ar_a.y_true, ar_a.y_pred, ar_b.y_pred)
                    key = f"{a} vs {b}"
                    mcnemar_results[key] = result
        report["__mcnemar_tests__"] = mcnemar_results

        return report

    def save_report(self, path: str):
        """Save the full evaluation report to JSON."""
        report = self.compute_all_metrics()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[METRICS] Evaluation report saved to {path}")

    def print_summary(self):
        """Print a formatted summary table to stdout."""
        if not self.architectures:
            print("\n[WARN] No architectures with valid predictions to evaluate.")
            print("  This usually means all predictions were INCONCLUSIVE.")
            print("  Check that the LLM agent is returning valid Green/Yellow/Red classes.")
            return

        report = self.compute_all_metrics()
        header = f"{'Architecture':<20} {'Acc':>6} {'F1-M':>6} {'F1-W':>6} {'Kappa':>6} {'MCC':>6}"
        print("\n" + "=" * len(header))
        print("ABLATION STUDY -- CLASSIFICATION METRICS")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for arch_name in self.architectures:
            m = report[arch_name]
            print(f"{arch_name:<20} "
                  f"{m['accuracy']:>5.3f} "
                  f"{m['macro_f1']:>5.3f} "
                  f"{m['weighted_f1']:>5.3f} "
                  f"{m['cohens_kappa']:>5.3f} "
                  f"{m['mcc']:>5.3f}")
        print("-" * len(header))

        # Per-class breakdown for best architecture
        best_arch = max(self.architectures, key=lambda a: report[a]["macro_f1"])
        print(f"\nPer-class breakdown ({best_arch}):")
        for cls_name in self.class_names:
            pc = report[best_arch]["per_class"][cls_name]
            print(f"  {cls_name:<8} P={pc['precision']:.3f}  R={pc['recall']:.3f}  "
                  f"F1={pc['f1']:.3f}  N={pc['support']}")

        # McNemar tests
        if report.get("__mcnemar_tests__"):
            print(f"\nStatistical significance (McNemar's test):")
            for pair, result in report["__mcnemar_tests__"].items():
                sig = "***" if result["p_value"] < 0.001 else \
                      "**" if result["p_value"] < 0.01 else \
                      "*" if result["p_value"] < 0.05 else "ns"
                print(f"  {pair}: chi2={result['chi2']:.2f}, p={result['p_value']:.4f} {sig}")
        print()

    def to_latex(self, path: str = None) -> str:
        """Generate a LaTeX table for the ablation results.
        
        Returns the LaTeX string and optionally saves to file.
        """
        report = self.compute_all_metrics()
        lines = []
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(r"\caption{Ablation study results for ATC-20 triage classification. "
                     r"Best results in \textbf{bold}. 95\% bootstrap CI in parentheses.}")
        lines.append(r"\label{tab:ablation}")
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{l" + "c" * 5 + "}")
        lines.append(r"\toprule")
        lines.append(r"Architecture & Accuracy & Macro-F1 & Weighted-F1 & Cohen's $\kappa$ & MCC \\")
        lines.append(r"\midrule")

        # Find best values for bolding
        metrics_keys = ["accuracy", "macro_f1", "weighted_f1", "cohens_kappa", "mcc"]
        best_vals = {}
        if self.architectures:
            for k in metrics_keys:
                best_vals[k] = max(report[a][k] for a in self.architectures)
        else:
            for k in metrics_keys:
                best_vals[k] = 0.0

        for arch_name in self.architectures:
            m = report[arch_name]
            vals = []
            for k in metrics_keys:
                v = m[k]
                ci_key = f"{k}_95ci"
                s = f"{v:.3f}"
                if ci_key in m:
                    ci = m[ci_key]
                    s += f" ({ci[1]:.3f}–{ci[2]:.3f})"
                if v == best_vals[k]:
                    s = r"\textbf{" + s + "}"
                vals.append(s)
            name_escaped = arch_name.replace("+", r"\texttt{+}")
            lines.append(f"{name_escaped} & " + " & ".join(vals) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")
        lines.append(r"\end{table}")

        latex = "\n".join(lines)

        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(latex)
            print(f"[FILE] LaTeX table saved to {path}")

        return latex

    def confusion_matrix_str(self, architecture: str) -> str:
        """Generate a formatted confusion matrix string."""
        ar = self.architectures.get(architecture)
        if not ar:
            return f"Architecture '{architecture}' not found."

        cm = confusion_matrix(ar.y_true, ar.y_pred, self.n_classes)
        lines = [f"\nConfusion Matrix — {architecture}"]
        lines.append(f"{'':>10} " + " ".join(f"{c:>8}" for c in self.class_names) + "  (predicted)")
        for i, cls in enumerate(self.class_names):
            row = " ".join(f"{cm[i][j]:>8}" for j in range(self.n_classes))
            lines.append(f"{cls:>10} {row}")
        lines.append(f"{'(true)':>10}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Self-Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Evaluation Module...")

    # Synthetic data: 30 samples per class, simulate different architectures
    rng = np.random.RandomState(42)
    n_per_class = 30
    y_true = [0] * n_per_class + [1] * n_per_class + [2] * n_per_class

    # Architecture A: good (90% accuracy)
    y_pred_a = list(y_true)
    for i in rng.choice(len(y_true), size=9, replace=False):
        y_pred_a[i] = (y_pred_a[i] + 1) % 3

    # Architecture B: worse (70% accuracy)
    y_pred_b = list(y_true)
    for i in rng.choice(len(y_true), size=27, replace=False):
        y_pred_b[i] = (y_pred_b[i] + 1) % 3

    ev = Evaluator(class_names=["Green", "Yellow", "Red"], n_bootstrap=500)
    ev.add_predictions(y_true, y_pred_a, "GeoFlow+Router",
                       confidences=[rng.uniform(0.7, 1.0) for _ in y_true],
                       latencies=[rng.uniform(0.5, 2.0) for _ in y_true])
    ev.add_predictions(y_true, y_pred_b, "NoCompletion",
                       confidences=[rng.uniform(0.4, 0.8) for _ in y_true],
                       latencies=[rng.uniform(0.3, 1.5) for _ in y_true])

    ev.print_summary()
    print(ev.confusion_matrix_str("GeoFlow+Router"))
    latex = ev.to_latex()
    print(f"\nLaTeX preview (first 200 chars):\n{latex[:200]}...")

    print("\n[OK] All evaluation tests passed!")
