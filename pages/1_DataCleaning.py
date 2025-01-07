import streamlit as st
import pandas as pd

# Configuration: Page
st.set_page_config(
    initial_sidebar_state="collapsed",
    layout='wide'
)

# Configuration: Page List
pages = {
    "Home Page": "./HomePage.py"
}

# Read data
df = pd.read_csv("./data/companies.csv")

# Initialize and set session state (for button press counter and dataset)
if 'press_count' not in st.session_state:
    st.session_state.press_count = 0
    st.session_state.df = df.copy()  # Store the original dataset in session state
    
# Operation: Process 'k' values
def process_k_columns(df, columns):
    processed_df = df.copy()
    for col in columns:
        processed_df[col] = processed_df[col].astype(str)
        
        def convert_k_value(x):
            try:
                if 'k' in str(x).lower():
                    return float(x.replace('k', '', 1)) * 1000
                else:
                    return float(x)
            except (ValueError, TypeError):
                return pd.NA
        
        processed_df[col] = processed_df[col].apply(convert_k_value)
    return processed_df

# Operation: Handle missing values and process numeric columns
def process_jobs_and_interviews(df):
    df['jobs'] = pd.to_numeric(df['jobs'], errors='coerce')
    df['interviews'] = pd.to_numeric(df['interviews'], errors='coerce')
    df = df.dropna(subset=['jobs'])
    df['interviews'] = df['interviews'].fillna(df['interviews'].median())
    df['rating'] = df['rating'].astype(float)
    df['reviews'] = df['reviews'].astype(float)
    df['jobs'] = df['jobs'].astype(float)
    df['interviews'] = df['interviews'].astype(float)
    return df

# Operation: Feature engineering
def feature_engineering(df):
    df['Work_Life_Balance'] = df['highly_rated_for'].str.contains('Work Life Balance', case=False, na=False).map({True: 1, False: 0})
    df['Job_Security'] = df['highly_rated_for'].str.contains('Job Security', case=False, na=False).map({True: 1, False: 0})
    df['Salary_&_Benefits'] = df['highly_rated_for'].str.contains('Salary & Benefits', case=False, na=False).map({True: 1, False: 0})
    df['Skill_Development_/_Learning'] = df['highly_rated_for'].str.contains('Skill Development / Learning', case=False, na=False).map({True: 1, False: 0})
    df['Company_Culture'] = df['highly_rated_for'].str.contains('Company Culture', case=False, na=False).map({True: 1, False: 0})
    df['Promotions_/_Appraisal'] = df['critically_rated_for'].str.contains('Promotions / Appraisal', case=False, na=False).map({True: 1, False: 0})
    return df

# Operation: Drop features
def drop_irrelevant_features(df):
    return df.drop(columns=['highly_rated_for', 'critically_rated_for'])

# Operation: Remove outliers by capping
def cap_outliers(df, float_columns):
    for col in float_columns:
        lower_bound = df[col].quantile(0.05)
        upper_bound = df[col].quantile(0.95)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

# Operation list
operations = [
    {"action": "Dropped irrelevant column 'company_name'", 
     "operation": lambda: st.session_state.df.drop(columns=['company_name'])},
    {"action": "Cleaned numeric columns by changing 'k' values to numeric", 
     "operation": lambda: process_k_columns(st.session_state.df, ['reviews', 'jobs', 'interviews'])},
    {"action": "Dropped missing values in 'jobs' and filled missing values in 'interviews' with median", 
     "operation": lambda: process_jobs_and_interviews(st.session_state.df)},
    {"action": "Created new features from 'highly_rated_for' and 'critically_rated_for'", 
     "operation": lambda: feature_engineering(st.session_state.df)},
    {"action": "Dropped column 'highly_rated_for' and 'critically_rated_for'", 
     "operation": lambda: drop_irrelevant_features(st.session_state.df)},
    {"action": "Removed outliers by capping at the 5th and 95th percentiles", 
     "operation": lambda: cap_outliers(st.session_state.df, ['reviews', 'jobs', 'interviews', 'rating'])}
]

# Function: Apply and print operations
def apply_operation():
    current_operation = operations[st.session_state.press_count]
    st.write(f"{current_operation['action']}")
    return current_operation['operation']()

# Function to reset session state
def reset_session_state():
    # Reset press count to 0 and reload the original dataset
    st.session_state.press_count = 0
    st.session_state.df = df.copy()

# Streamlit container
with st.container():
    
    margin, col1, col2 = st.columns([1, 7, 1])  
    with margin:
        st.empty()
        
    with col1:
        st.subheader("Data Cleaning")
        table_placeholder = st.empty()
        if st.session_state.press_count < len(operations):
            if st.button("Next Step", key="next_step", type="secondary"):
                # Perform the current operation
                st.session_state.df = apply_operation()

                # Update the table in the placeholder
                table_placeholder.write(st.session_state.df)

                # Increment the press count (after the render cycle)
                st.session_state.press_count += 1
                
                # Handle the final operation output
        if st.session_state.press_count == len(operations):
            st.write("Final operation reached! Dataset is fully cleaned.")

            # Save the DataFrame as a CSV file in the 'data' folder
            output_path = './data/companies_cleaned.csv'
            st.session_state.df.to_csv(output_path, index=False)

            # Add a button to navigate to the "2_ChooseModel" page
            if st.button("Back"):
                st.switch_page(pages["Home Page"])
                st.rerun()  # Trigger a re-run to reflect the state change
                
    with col2:
        # "Refresh" button on the right
        if st.button("Refresh", type="tertiary"):
            reset_session_state()  # Reset session state
            st.rerun()  # Trigger a re-run to reflect the state change

    # Display the dataset initially in the placeholder
    table_placeholder.write(st.session_state.df)

    







