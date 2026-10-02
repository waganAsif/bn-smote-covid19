"""Performance metrics (Equations 3-8 of the paper) and ROC analysis."""

import time

import numpy as np
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)

METRIC_NAMES = ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1-score"]


def specificity_score(y_true, y_pred) -> float:
    """Specificity = TN / (TN + FP)."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def classification_metrics(y_true, y_pred, average: str = "weighted") -> dict:
    """Accuracy, precision, recall, specificity and F1-score.

    `average="weighted"` is used for the model comparison on the test data and
    `average="binary"` (positive class = 1) for the independent validation.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_pred, average=average, zero_division=0),
        "Specificity": specificity_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def roc_from_labels(y_true, y_pred):
    """ROC curve and AUC computed from the predicted class labels."""
    fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_pred))
    return fpr, tpr, auc(fpr, tpr)


class Timer:
    """Context manager that stores the elapsed wall-clock time in seconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start
        return False
