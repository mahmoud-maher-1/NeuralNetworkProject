"""
Core Logic: Preprocessing
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Union, Optional
from models.validation_models import TrainingParams
from utils.constants import (
    TRAIN_SAMPLES_PER_CLASS,
    TEST_SAMPLES_PER_CLASS,
    RANDOM_SEED,
    DATA_FILE_PATH,
    PROCESSED_DATA_FILE_PATH,
    ALL_FEATURES
)

class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    (z = (x - u) / s)
    """
    def __init__(self):
        self.mean_: np.ndarray = np.array([])
        self.std_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray):
        """
        Compute the mean and standard deviation to be used for later scaling.

        Args:
            X (np.ndarray): The data used to compute the mean and std.
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_ = np.where(self.std_ == 0, 1e-8, self.std_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.

        Args:
            X (np.ndarray): The data to transform.

        Returns:
            np.ndarray: The transformed data.
        """
        if self.mean_.size == 0 or self.std_.size == 0:
            raise RuntimeError("Must fit scaler before transforming data.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Args:
            X (np.ndarray): The data to fit and transform.

        Returns:
            np.ndarray: The transformed data.
        """
        self.fit(X)
        return self.transform(X)


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercases, trims, and replaces spaces/dashes with underscores in column names."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    return df

def _encode_gender(series: pd.Series) -> pd.Series:
    """Encodes gender to numeric."""
    mapping = {"male": 1.0, "female": 0.0}
    return series.map(mapping)

def _process_and_save_data() -> Union[pd.DataFrame, str]:
    """
    Internal function to load raw data, clean, impute, encode,
    and save it to the processed file path.
    """
    try:
        df_raw = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        return f"Error: Raw dataset not found at {DATA_FILE_PATH}."

    df = _normalize_cols(df_raw)

    # Rename target column for clarity
    if "bird_category" in df.columns:
        df = df.rename(columns={"bird_category": "target"})
    else:
        return "Error: 'bird category' column not found in dataset."

    rng_impute = np.random.default_rng(RANDOM_SEED)

    # --- Handle Missing Gender Data ---
    if "gender" in df.columns:
        df["gender"] = df["gender"].replace("NA", np.nan)
        mask = df["gender"].isna()
        if mask.sum() > 0:
            df.loc[mask, "gender"] = rng_impute.choice(
                ["male", "female"], size=mask.sum()
            )
        df["gender"] = _encode_gender(df["gender"])

    # --- Handle Missing Numeric Data (Median Imputation) ---
    numeric_features = [f for f in ALL_FEATURES if f != 'gender']
    for col in numeric_features:
        if col in df.columns and df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Save the fully processed file
    try:
        df.to_csv(PROCESSED_DATA_FILE_PATH, index=False)
    except OSError as e:
        return f"Error: Could not save processed file. Check permissions. {e}"

    return df


def load_and_split_data(
    params: TrainingParams,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], str]:
    """
    Loads, filters, splits, and scales the data based on user parameters.
    """

    # --- Caching Logic ---
    if os.path.exists(PROCESSED_DATA_FILE_PATH):
        try:
            df_final = pd.read_csv(PROCESSED_DATA_FILE_PATH)
        except Exception as e:
            return f"Error loading processed file: {e}. Try deleting data/processed.csv."
    else:
        # File not found, so we process and save it
        result = _process_and_save_data()
        if isinstance(result, str):
            return result  # Return error message
        df_final = result

    # 1. Filter for the two selected classes
    if params.class1 not in df_final["target"].unique() or params.class2 not in df_final["target"].unique():
        return f"Error: Classes {params.class1} or {params.class2} not in dataset."

    df_filtered = df_final[df_final["target"].isin([params.class1, params.class2])].copy()

    # 2. Select the two features
    features = [params.feature1, params.feature2]

    df_split_data = df_filtered[["target"] + features].dropna(subset=features)

    # 3. Stratified split
    rng = np.random.default_rng(params.random_seed)
    train_parts, test_parts = [], []

    total_needed_per_class = TRAIN_SAMPLES_PER_CLASS + TEST_SAMPLES_PER_CLASS

    for cls in [params.class1, params.class2]:
        cls_df = df_split_data[df_split_data["target"] == cls]

        if len(cls_df) < total_needed_per_class:
            return (
                f"Error: Class '{cls}' has only {len(cls_df)} valid samples "
                f"(after imputation). Need at least {total_needed_per_class} "
                "for train/test split."
            )

        idx = cls_df.index.to_numpy()
        rng.shuffle(idx)

        train_idx = idx[:TRAIN_SAMPLES_PER_CLASS]
        test_idx = idx[
            TRAIN_SAMPLES_PER_CLASS : (TRAIN_SAMPLES_PER_CLASS + TEST_SAMPLES_PER_CLASS)
        ]

        train_parts.append(df_split_data.loc[train_idx])
        test_parts.append(df_split_data.loc[test_idx])

    train_df = pd.concat(train_parts, axis=0)
    test_df = pd.concat(test_parts, axis=0)

    # 4. Prepare data for models (using -1, +1 labels)
    X_train_raw = train_df[features].to_numpy(dtype=float)
    X_test_raw = test_df[features].to_numpy(dtype=float)

    y_train = np.where(train_df["target"].values == params.class1, -1, 1)
    y_test = np.where(test_df["target"].values == params.class1, -1, 1)

    # --- 5. Scale the feature data ---
    # StandardScaler is now part of this module
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return X_train_scaled, y_train, X_test_scaled, y_test
