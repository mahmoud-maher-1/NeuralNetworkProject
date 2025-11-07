"""
Core Logic: Prediction
"""

import numpy as np
from models.validation_models import ModelWeights


def predict(
        X: np.ndarray, weights: ModelWeights, include_bias: bool
) -> np.ndarray:
    """
    Generates class predictions (-1 or +1) given features and trained weights.
    """
    w_array = np.array(weights.w_)

    net_input = X @ w_array + (weights.b_ if include_bias else 0.0)

    predictions = np.where(net_input >= 0.0, 1, -1)
    return predictions