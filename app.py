import streamlit as st
from pathlib import Path

# Main application configuration
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Load original application as the only Streamlit page.
# This keeps its existing 15-option sidebar navigation.
main_page = st.Page(
    str(BASE_DIR / "src" / "app.py"),
    title="FraudShield AI",
    default=True
)

navigation = st.navigation(
    [main_page],
    position="hidden"
)

navigation.run()
