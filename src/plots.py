"""Figures for the model comparison, inference examples and validation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

COLORS = ["#4C72B0", "#ff7f0e", "#2ca02c", "#d62728"]


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(table: pd.DataFrame, metric_names, path: Path):
    """Grouped bar chart of the performance metrics of each model (Figure 6)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    width = 0.8 / len(table)
    index = np.arange(len(metric_names))
    for i, (model, row) in enumerate(table.iterrows()):
        ax.bar(index + i * width, row[metric_names].values, width=width,
               label=model, color=COLORS[i % len(COLORS)])
    ax.set_xticks(index + width * (len(table) - 1) / 2, metric_names)
    ax.set_xlabel("Metrics", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Performance Metrics Comparison for Different Models", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(fontsize=10)
    _save(fig, path)


def plot_total_time(table: pd.DataFrame, column: str, path: Path):
    """Bar chart of training + prediction time of each model (Figure 7)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (model, row) in enumerate(table.iterrows()):
        ax.bar(i, row[column], label=model, color=COLORS[i % len(COLORS)])
    ax.set_xticks(range(len(table)), table.index, rotation=10)
    ax.set_ylabel("Time (seconds)", fontsize=14)
    ax.set_title("Total Time Taken (Training + Prediction) for Different Models",
                 fontsize=16, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    _save(fig, path)


def plot_roc(fpr, tpr, roc_auc, path: Path):
    """ROC curve of the Bayesian Network (Figure 9)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_inference_table(evidence: dict, posterior: dict, path: Path, target="corona_result"):
    """Table of P(target | evidence) for one evidence set (Figure 4)."""
    given = ", ".join(f"{k}={v}" for k, v in evidence.items())
    cells = [[f"{target}({state})", f"{prob * 100:.2f}%"] for state, prob in posterior.items()]
    fig, ax = plt.subplots(figsize=(6, 2.2))
    table = ax.table(cellText=cells, colLabels=[target, f"P({target} | evidence)"],
                     cellLoc="center", loc="center", colColours=["#d9d9d9", "#d9d9d9"])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.6)
    ax.axis("off")
    ax.set_title(f"P({target} | {given})", fontsize=11)
    _save(fig, path)


def plot_evidence_sets(evidence_sets, posteriors, path: Path, target="corona_result"):
    """Posterior probabilities for several evidence sets (Figure 5)."""
    states = sorted(posteriors[0])
    x = np.arange(len(states))
    width = 0.8 / len(evidence_sets)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (evidence, posterior) in enumerate(zip(evidence_sets, posteriors)):
        label = ", ".join(f"{k}={v}" for k, v in evidence.items())
        values = [posterior[s] for s in states]
        bars = ax.bar(x + i * width, values, width, label=label,
                      color=COLORS[i % len(COLORS)], edgecolor="black")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.2f}", ha="center", va="bottom")
    ax.set_xticks(x + width * (len(evidence_sets) - 1) / 2, states)
    ax.set_xlabel("Corona Result")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_training_vs_independent(train_scores: dict, independent_scores: dict, path: Path):
    """Training vs independent dataset performance (Figure 8)."""
    names = list(train_scores)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.175, [train_scores[n] for n in names], 0.35, label="Training Dataset", color="#4C72B0")
    ax.bar(x + 0.175, [independent_scores[n] for n in names], 0.35, label="Independent Dataset", color="#55A868")
    ax.set_xticks(x, names)
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Scores")
    ax.set_title("Performance Comparison (Training vs Independent Dataset)")
    ax.set_ylim(0, 1)
    ax.legend()
    _save(fig, path)


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1], cmap=plt.cm.Blues, ax=ax)
    ax.set_title(title)
    _save(fig, path)
