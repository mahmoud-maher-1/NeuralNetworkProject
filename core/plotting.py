"""
Core Logic: Plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from models.validation_models import TrainingParams, ModelWeights


def plot_decision_boundary(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        weights: ModelWeights,
        params: TrainingParams,
) -> Figure:
    """
    Generates a scatter plot of data points and the linear decision boundary.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- Plot Data Points ---
    # Training data
    ax.scatter(
        X_train[y_train == -1, 0],
        X_train[y_train == -1, 1],
        marker="o",
        label=f"Class {params.class1} (Train)",
        alpha=0.7,
        edgecolors="k",
    )
    ax.scatter(
        X_train[y_train == 1, 0],
        X_train[y_train == 1, 1],
        marker="s",
        label=f"Class {params.class2} (Train)",
        alpha=0.7,
        edgecolors="k",
    )

    # Test data
    ax.scatter(
        X_test[y_test == -1, 0],
        X_test[y_test == -1, 1],
        marker="o",
        label=f"Class {params.class1} (Test)",
        facecolors='none',
        edgecolors='b',
        s=100
    )
    ax.scatter(
        X_test[y_test == 1, 0],
        X_test[y_test == 1, 1],
        marker="s",
        label=f"Class {params.class2} (Test)",
        facecolors='none',
        edgecolors='orange',
        s=100
    )

    # --- Plot Decision Boundary ---
    w = np.array(weights.w_)
    b = weights.b_ if params.include_bias else 0.0

    if np.linalg.norm(w) > 0:
        x_min = min(X_train[:, 0].min(), X_test[:, 0].min()) - 1
        x_max = max(X_train[:, 0].max(), X_test[:, 0].max()) + 1

        x_vals = np.array([x_min, x_max])

        if abs(w[1]) > 1e-12:
            # Standard case: w0*x + w1*y + b = 0  => y = -(w0/w1)x - b/w1
            y_vals = -(w[0] / w[1]) * x_vals - (b / w[1])
            ax.plot(x_vals, y_vals, "r--", linewidth=2, label="Decision Boundary")
        else:
            # Vertical line case: w0*x + b = 0 => x = -b/w0
            x_intercept = -b / (w[0] + 1e-12)
            ax.axvline(x_intercept, color="r", linestyle="--", linewidth=2, label="Decision Boundary")

    ax.set_xlabel(params.feature1)
    ax.set_ylabel(params.feature2)
    ax.set_title(
        f"{params.algorithm} Decision Boundary\n"
        f"({params.class1} vs. {params.class2}) | Epochs: {params.epochs}"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()

    return fig