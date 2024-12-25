import streamlit as st
from scripts.dashboard_utils import *

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["User Overview", "User Engagement", "Experience Analysis", "Satisfaction Analysis"]
    )
    
    if page == "User Overview":
        create_overview_page()
    elif page == "User Engagement":
        create_engagement_page()
    elif page == "Experience Analysis":
        create_experience_page()
    else:
        create_satisfaction_page()
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Dashboard created with Streamlit")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Telecom Analytics Dashboard",
        page_icon="📊",
        layout="wide"
    )
    main() 