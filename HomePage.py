import requests
import streamlit as st
from streamlit_lottie import st_lottie

# Run in terminal: streamlit run "Home Page.py"

# Data source: https://www.kaggle.com/datasets/noeyislearning/it-job-market-insights

# Add a random statistics about work life balance?

# Add a lottie animation

# Configuration: Setting Up Main Page
st.set_page_config(
    page_title="Company Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapse the sidebar by default
)

# Configuration: Page List
pages = {
    "Data Cleaning": "./pages/1_DataCleaning.py",
    "Choose Model": "./pages/2_ChooseModel.py",
}

# Load lottie animations
def get_lottie(url):
    link = requests.get(url)
    return link.json()
animation = get_lottie("https://lottie.host/dc99f77d-f2ff-45ea-896a-bcf55b4ded78/6Mk6BHTG44.json")


with st.container():
    margin1, col1, col2, margin2 = st.columns([2, 5, 3, 2])
    with margin1:
        st.empty()
        
    with col1:
        # Main title
        st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True)
        st.title("Company Analysis")

        # Subheader
        st.subheader("Does your company have work-life balance?")
        st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True)

        if st.button("Let's find out!"):
            st.switch_page(pages["Choose Model"])
            
        if st.button("Or click here to learn how we clean our data", type="tertiary"):
            st.switch_page(pages["Data Cleaning"])
    
    with col2:
        st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True)
        st_lottie(animation, width = 350)
        
    with margin2:
        st.empty()
        
st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
st.markdown("---")
st.markdown('<div style="height: 25px;"></div>', unsafe_allow_html=True)