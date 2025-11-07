"""
UI Layer: Home Page (ui/home_page.py)
"""

import streamlit as st
import pandas as pd
from models.validation_models import TrainingParams, TrainingResult
from utils.constants import (
    ALL_FEATURES,
    ALL_CLASSES,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_MSE_THRESH,
    RANDOM_SEED,
    TRAIN_SAMPLES_PER_CLASS,
    TEST_SAMPLES_PER_CLASS,
)
from core.preprocessing import load_and_split_data
from core.training import train_model
from core.prediction import predict
from core.evaluation import calculate_metrics
from core.plotting import plot_decision_boundary

def render():
    """
    Renders the Home page for user input.
    """
    st.title("Train Perceptron or Adaline Model")
    st.markdown(
        f"""
    Configure your model parameters below. The model will be trained on
    **{TRAIN_SAMPLES_PER_CLASS * 2} random samples**
    (_{TRAIN_SAMPLES_PER_CLASS} from each class_)
    and tested on **{TEST_SAMPLES_PER_CLASS * 2} random samples**
    (_{TEST_SAMPLES_PER_CLASS} from each class_).
    """
    )

    with st.form(key="training_form"):
        st.subheader("1. Data Selection")
        c1, c2 = st.columns(2)
        with c1:
            feature1 = st.selectbox(
                "Select Feature 1 (X-axis)",
                ALL_FEATURES,
                index=2,  # 'beak_length'
            )
            class1 = st.selectbox(
                "Select Class 1 (Target: -1)", ALL_CLASSES, index=0  # 'A'
            )
        with c2:
            feature2 = st.selectbox(
                "Select Feature 2 (Y-axis)",
                ALL_FEATURES,
                index=3,  # 'beak_depth'
            )
            class2 = st.selectbox(
                "Select Class 2 (Target: +1)", ALL_CLASSES, index=1  # 'B'
            )

        st.subheader("2. Algorithm Selection")
        algorithm = st.radio(
            "Choose Model",
            ("Perceptron", "Adaline"),
            horizontal=True,
        )

        st.subheader("3. Hyperparameter Configuration")
        c3, c4, c5 = st.columns(3)
        with c3:
            epochs = st.number_input(
                "Number of Epochs",
                min_value=1,
                max_value=1000,
                value=DEFAULT_EPOCHS,
                step=10,
            )
            include_bias = st.checkbox("Include Bias Term", value=True)

        with c4:
            learning_rate = st.number_input(
                "Learning Rate (eta)",
                min_value=0.0001,
                max_value=1.0,
                value=DEFAULT_LR,
                step=0.001,
                format="%.4f",
                help="Used for both Perceptron and Adaline update rules.",
            )

        with c5:
            mse_threshold = st.number_input(
                "MSE Threshold (Optional)",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_MSE_THRESH,
                step=0.001,
                format="%.4f",
                help="Stops Adaline training early if MSE drops below this.",
            )

        submit_button = st.form_submit_button(label="Train Model")

    if submit_button:
        # --- Input Validation ---
        if feature1 == feature2:
            st.error("Please select two different features.")
            return
        if class1 == class2:
            st.error("Please select two different target classes.")
            return

        # Use Pydantic model for validation and type conversion
        try:
            params = TrainingParams(
                algorithm=algorithm,
                feature1=feature1,
                feature2=feature2,
                class1=class1,
                class2=class2,
                learning_rate=learning_rate,
                epochs=epochs,
                mse_threshold=mse_threshold if mse_threshold > 0 else None,
                include_bias=include_bias,
                random_seed=RANDOM_SEED,
            )
        except Exception as e:
            st.error(f"Parameter validation failed: {e}")
            return

        # --- Orchestration ---
        try:
            with st.spinner("Training model..."):
                # 1. Preprocessing
                data_load_result = load_and_split_data(params)
                if isinstance(data_load_result, str):
                    st.error(data_load_result)
                    return

                X_train, y_train, X_test, y_test = data_load_result

                # 2. Training
                weights = train_model(X_train, y_train, params)

                # 3. Prediction
                y_pred_test = predict(X_test, weights, params.include_bias)

                # 4. Evaluation
                metrics = calculate_metrics(y_test, y_pred_test)

                # 5. Plotting
                fig = plot_decision_boundary(
                    X_train, y_train, X_test, y_test, weights, params
                )

                # Store result in session state
                st.session_state.training_result = TrainingResult(
                    params=params, weights=weights, metrics=metrics, plot_figure=fig
                )

                # Navigate to Results page
                st.session_state.page = "Results"
                st.rerun()

        except ValueError as ve:
            st.error(f"An error occurred during training: {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")