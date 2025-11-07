"""
Core Logic: Evaluation
"""

import numpy as np
from typing import Dict
from models.validation_models import Metrics


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """
    Calculates accuracy, precision, recall, F1, and confusion matrix manually.
    Assumes labels are -1 (Negative) and +1 (Positive).
    """
    # Epsilon for safe division
    epsilon = 1e-9

    # Confusion Matrix components
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == -1) & (y_pred == -1)).sum())
    fp = int(((y_true == -1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == -1)).sum())

    cm = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    # Metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / (total + epsilon)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    return Metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        confusion_matrix=cm,
    )