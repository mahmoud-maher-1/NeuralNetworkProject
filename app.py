"""
Streamlit App Entrypoint (app.py)

This file acts as the main router for the multi-page Streamlit application.
It uses st.session_state to manage the current page and pass data between pages.
"""

import streamlit as st
from ui import home_page, results_page
from utils.constants import APP_TITLE, APP_ICON

def main():
    """
    Main function to control page navigation.
    """
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

    # --- Initialize Session State ---
    # This is the primary mechanism for page routing and state management
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "training_params" not in st.session_state:
        st.session_state.training_params = None
    if "training_result" not in st.session_state:
        st.session_state.training_result = None

    # --- Page Routing ---
    if st.session_state.page == "Home":
        home_page.render()
    elif st.session_state.page == "Results":
        results_page.render()

if __name__ == "__main__":
    main()