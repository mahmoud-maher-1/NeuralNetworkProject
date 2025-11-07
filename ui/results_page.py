"""
UI Layer: Results Page
"""

import streamlit as st
import pandas as pd
from models.validation_models import TrainingResult

def render():
    """
    Renders the Results page.
    """
    st.title("Training Results")

    # Check if results exist in session state
    if "training_result" not in st.session_state or st.session_state.training_result is None:
        st.error("No training results found. Please go back and train a model.")
        if st.button("Back to Home"):
            st.session_state.page = "Home"
            st.rerun()
        return

    result: TrainingResult = st.session_state.training_result

    # --- Display Parameters ---
    st.subheader("1. Training Configuration")
    params_df = pd.DataFrame(
        [result.params.model_dump()],
    ).T.rename(columns={0: "Value"})

    params_df["Value"] = params_df["Value"].astype(str)

    st.dataframe(params_df, use_container_width=True)

    # --- Display Metrics ---
    st.subheader("2. Model Performance (on Test Set)")
    metrics = result.metrics
    cm = metrics.confusion_matrix

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy", f"{metrics.accuracy:.2%}")
        st.metric("Precision", f"{metrics.precision:.2%}")
        st.metric("Recall", f"{metrics.recall:.2%}")
        st.metric("F1-Score", f"{metrics.f1_score:.2%}")

    with c2:
        st.markdown("**Confusion Matrix**")
        cm_df = pd.DataFrame(
            [
                [f"TP: {cm['TP']}", f"FP: {cm['FP']}"],
                [f"FN: {cm['FN']}", f"TN: {cm['TN']}"],
            ],
            columns=["Predicted +1", "Predicted -1"],
            index=["Actual +1", "Actual -1"],
        )
        st.dataframe(cm_df, use_container_width=True)

    # --- Display Plot ---
    st.subheader("3. Decision Boundary")
    st.pyplot(result.plot_figure)

    # --- Navigation ---
    if st.button("Train a New Model"):
        st.session_state.page = "Home"
        st.session_state.training_result = None
        st.session_state.training_params = None
        st.rerun()
