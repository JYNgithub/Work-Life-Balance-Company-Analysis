import streamlit as st

# Run in terminal: streamlit run "Home Page.py"

# Data source: https://www.kaggle.com/datasets/noeyislearning/it-job-market-insights

# Add a random statistics about work life balance?

# Add a lottie animation

# Configuration: Setting Up Main Page
st.set_page_config(
    page_title="Company Analysis",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"  # Collapse the sidebar by default
)

# Configuration: Page List
pages = {
    "Data Cleaning": "./pages/1_DataCleaning.py",
    "Choose Model": "./pages/2_ChooseModel.py",
}
    
# Main title
st.title("Company Analysis")

# Subheader
st.subheader("Does your company have work-life balance?")
st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True)

if st.button("Let's find out!"):
    st.switch_page(pages["Choose Model"])
    
if st.button("Or click here to learn how we clean our data", type="tertiary"):
    st.switch_page(pages["Data Cleaning"])

st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
st.markdown("---")
st.markdown('<div style="height: 25px;"></div>', unsafe_allow_html=True)