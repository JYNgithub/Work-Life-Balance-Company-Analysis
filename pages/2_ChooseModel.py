import streamlit as st
import pandas as pd

# Configuration: Page
st.set_page_config(
    initial_sidebar_state="collapsed",
    layout='wide'
)

# Configuration: Page List
pages = {
    "Home Page": "./HomePage.py",
    "Logistic Regression": "./pages/3_LogisticRegression.py",
    "Decision Tree": "./pages/4_DecisionTree.py",
    "Support Vector Machine": "./pages/5_SupportVectorMachine.py",
}
    
# Page container
with st.container():
    margin1, col1, margin2 = st.columns([1, 5, 1])

    with margin1:
        st.empty()
        
    with col1:
        if st.button("Back"):
            st.switch_page(pages["Home Page"])
        
        st.write("Choose a model")
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)

        # Logistic Regression Section
        row1_col1, row1_col2 = st.columns([4, 1])  # Subheader and button on the same row
        with row1_col1:
            st.subheader("Logistic Regression")
            st.write("A statistical model that predicts the probability of a binary outcome by fitting data to a logistic function.")
        with row1_col2:
            if st.button("Select", key=1):
                st.switch_page(pages["Logistic Regression"])
        
        #st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("---")
        #st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)

        # Decision Tree Section
        row2_col1, row2_col2 = st.columns([4, 1])  # Subheader and button on the same row
        with row2_col1:
            st.subheader("Decision Tree")
            st.write("A flowchart-like model that splits data into subsets based on feature values, recursively building a tree to make decisions.")
        with row2_col2:
            if st.button("Select", key=2):
                st.switch_page(pages["Decision Tree"])
        
        #st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("---")
        #st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)

        # Support Vector Machine Section
        row3_col1, row3_col2 = st.columns([4, 1])  # Subheader and button on the same row
        with row3_col1:
            st.subheader("Support Vector Machine")
            st.write("A supervised learning algorithm that finds the hyperplane that best separates data points of different classes by maximizing the margin between them.")
        with row3_col2:
            if st.button("Select", key=3):
                st.switch_page(pages["Support Vector Machine"])
                
        #st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("---")


    with margin2:
        st.empty()


     
    
    
