"""
Core Logic: Training
"""

import numpy as np
from typing import List, Optional
from models.validation_models import TrainingParams, ModelWeights

class Perceptron:
    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 50,
        include_bias: bool = True,
        random_seed: int = 42,
    ):
        self.lr = learning_rate
        self.epochs = epochs
        self.include_bias = include_bias
        self.random_seed = random_seed
        self.w_: np.ndarray = np.array([])
        self.b_: float = 0.0
        self.loss_history_: List[int] = []

    def _init_params(self, n_features: int):
        """Initializes weights and bias."""
        rng = np.random.default_rng(self.random_seed)
        self.w_ = rng.normal(0, 0.01, size=(n_features,))
        self.b_ = rng.normal(0, 0.01) if self.include_bias else 0.0

    def _net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculates net input."""
        return X @ self.w_ + (self.b_ if self.include_bias else 0.0)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels (-1 or +1)."""
        return np.where(self._net_input(X) >= 0.0, 1, -1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the Perceptron model."""
        _, n_features = X.shape
        self._init_params(n_features)
        self.loss_history_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = self._predict(xi.reshape(1, -1))[0] # Signum activation
                update = target - pred # Error calculation

                if update != 0:
                    self.w_ += self.lr * update * xi
                    if self.include_bias:
                        self.b_ += self.lr * update
                    # ---------------------------------------------
                    errors += 1
            self.loss_history_.append(errors)
        return self


class AdalineGD:
    """
    Implements the Adaline (Adaptive Linear Neuron) algorithm with Gradient Descent.
    Labels are expected to be -1 or +1.
    """
    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 50,
        include_bias: bool = True,
        mse_threshold: Optional[float] = None,
        random_seed: int = 42,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.include_bias = include_bias
        self.mse_threshold = mse_threshold
        self.random_seed = random_seed
        self.w_: np.ndarray = np.array([])
        self.b_: float = 0.0
        self.loss_history_: List[float] = []

    def _init_params(self, n_features: int):
        """Initializes weights and bias."""
        rng = np.random.default_rng(self.random_seed)
        self.w_ = rng.normal(0, 0.01, size=(n_features,))
        self.b_ = rng.normal(0, 0.01) if self.include_bias else 0.0

    def _net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculates net input (which is also the activation for Adaline)."""
        return X @ self.w_ + (self.b_ if self.include_bias else 0.0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the Adaline model using batch gradient descent."""
        n_samples, n_features = X.shape
        self._init_params(n_features)
        self.loss_history_ = []

        for _ in range(self.epochs):
            net = self._net_input(X) # Linear Activation
            errors = y - net
            cost = (errors**2).mean()  # Mean Squared Error (MSE)
            self.loss_history_.append(cost)

            if self.mse_threshold is not None and cost <= self.mse_threshold:
                break  # Stop early if threshold is met

            grad_w = -2.0 * (X.T @ errors) / n_samples
            grad_b = -2.0 * errors.mean() if self.include_bias else 0.0

            # Weight and bias update
            self.w_ -= self.learning_rate * grad_w
            if self.include_bias:
                self.b_ -= self.learning_rate * grad_b
        return self

def train_model(
    X_train: np.ndarray, y_train: np.ndarray, params: TrainingParams
) -> ModelWeights:
    """
    Factory function to instantiate and train the selected model.
    """
    if params.algorithm == "Perceptron":
        model = Perceptron(
            learning_rate=params.learning_rate,
            epochs=params.epochs,
            include_bias=params.include_bias,
            random_seed=params.random_seed,
        )
    elif params.algorithm == "Adaline":
        model = AdalineGD(
            learning_rate=params.learning_rate,
            epochs=params.epochs,
            include_bias=params.include_bias,
            mse_threshold=params.mse_threshold,
            random_seed=params.random_seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {params.algorithm}")

    model.fit(X_train, y_train)

    return ModelWeights(w_=model.w_.tolist(), b_=model.b_)