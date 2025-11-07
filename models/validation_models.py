"""
Models: Pydantic Validation

Defines Pydantic models for type-safe parameter validation and
structured data passing.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
import matplotlib.pyplot as plt

class AlgorithmType(str, Enum):
    """Enum for algorithm selection."""
    PERCEPTRON = "Perceptron"
    ADALINE = "Adaline"

class TrainingParams(BaseModel):
    """
    Validates and stores all user-configurable training parameters.
    """
    algorithm: AlgorithmType
    feature1: str = Field(..., min_length=1)
    feature2: str = Field(..., min_length=1)
    class1: str = Field(..., min_length=1)
    class2: str = Field(..., min_length=1)
    learning_rate: float = Field(..., gt=0.0)
    epochs: int = Field(..., gt=0)
    mse_threshold: Optional[float] = Field(None, gt=0.0)
    include_bias: bool
    random_seed: int

class ModelWeights(BaseModel):
    """
    Stores the trained model weights.
    """
    w_: List[float]
    b_: float

class Metrics(BaseModel):
    """
    Stores the calculated performance metrics.
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Dict[str, int]

class TrainingResult(BaseModel):
    """
    A container for all training outputs to be passed to the UI.
    """
    params: TrainingParams
    weights: ModelWeights
    metrics: Metrics
    plot_figure: Any  # Store the Matplotlib figure

    class Config:
        # Allow Matplotlib Figure object
        arbitrary_types_allowed = True