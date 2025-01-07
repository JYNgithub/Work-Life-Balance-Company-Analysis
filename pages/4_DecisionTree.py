import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier

# Configuration: Page
st.set_page_config(
    initial_sidebar_state="collapsed",
    layout='wide'
)

# Configuration: Page List
pages = {
    "Choose Model": "./pages/2_ChooseModel.py",
}

# Read data
df = pd.read_csv("./data/companies_cleaned.csv")

# This part is messy, but it ensures it runs everytime the page loads to update the contents
# ------------------------------------------------------------------------------------------
# Set seed
seed_value = 9029
# Define input features (X) and target feature (y)
X = df.drop(columns=['Work_Life_Balance'])
y = df['Work_Life_Balance']
# Perform a stratified train-test split to maintain the target class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed_value)
# Apply RobustScaler and update the DataFrame
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Initialize SMOTE
smote = SMOTE(random_state=seed_value)
X, y = smote.fit_resample(X_train, y_train)

# Define the model with the best parameters
dt_model = DecisionTreeClassifier(
    class_weight=None,
    criterion='gini',
    max_depth=10,
    max_features=None,
    min_samples_leaf=2,
    min_samples_split=20,
    splitter='random',
    random_state=seed_value
)
# Train the model
dt_model.fit(X_train, y_train)

# Evaluation
# Model Accuracy
accuracy = dt_model.score(X_test, y_test)
# Confusion Matrix
y_pred = dt_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 4))  
sns.heatmap(cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=dt_model.classes_, 
            yticklabels=dt_model.classes_,
            annot_kws={"size": 14} )
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
cm_plot_path = "./plots/dt_confusion_matrix.jpg"  # Define the path to save the image
plt.savefig(cm_plot_path, bbox_inches='tight')  # Save the plot with tight layout to avoid cropping
plt.close()  
# Generate the classification report
report = classification_report(y_test, y_pred)
# AUC plot
roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(4, 4))  
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guess Reference Line")  
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
roc_plot_path = "./plots/dt_roc_curve.jpg"  # Define the path to save the image
plt.savefig(roc_plot_path, bbox_inches='tight')  # Save the plot with tight layout to avoid cropping
plt.close()  
# ------------------------------------------------------------------------------------------

# Page container
with st.container():
    margin1, col1, margin2 = st.columns([1, 16, 1])

    with margin1:
        st.empty()
        
    with col1:
        if st.button("Back"):
            st.switch_page(pages["Choose Model"])
        
        st.title("Decision Tree")
        st.markdown('<p style="font-size: 16px;">Scroll down to use the model</p>', unsafe_allow_html=True)

        st.markdown("---")

        
        # Add three columns inside col1
        inner_col1, inner_col2, inner_col3 = st.columns([1.5,1,1])
        with inner_col1:
            # Display accuracy
            st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")
            # Display classification report
            st.write("Classification Report")
            st.code(report)  
        with inner_col2:
            # Display confusion matrix
            st.write("Confusion Matrix")
            st.image(cm_plot_path, width=350)
        with inner_col3:
            # Display confusion matrix
            st.write("ROC Curve")
            st.image(roc_plot_path, width=350) 
        
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)

        # Create a form for user input
        with st.form("prediction_form"):
            st.subheader("Prediction")
            st.markdown("You may find your company's data from [AmbitionBox](https://ambitionbox.com) to complete the form.")
            # Input fields for features
            rating = st.number_input("How is your company rated? (1-5)", min_value=0.0, max_value=5.0, step=0.1)
            reviews = st.number_input("How many reviews have your company received?", min_value=0, step=1)
            jobs = st.number_input("How many job postings from your company?", min_value=0, step=1)
            interviews = st.number_input("How many interviews shared by candidates?", min_value=0, step=1)
            job_security = st.radio(
                "Is your company highly rated for Job Security?", 
                options=[1, 0], 
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            salary_benefits = st.radio(
                "Is your company highly rated for Salary & Benefits?", 
                options=[1, 0], 
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            skill_development = st.radio(
                "Is your company highly rated for Skill Development & Learning?", 
                options=[1, 0], 
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            company_culture = st.radio(
                "Is your company highly rated for Company Culture?", 
                options=[1, 0], 
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            promotions = st.radio(
                "Is your company critically rated for Promotions & Appraisal?", 
                options=[1, 0], 
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            # Submit button
            submit_button = st.form_submit_button(label="Predict")
        
            # Display prediction result
            if submit_button:
                # Collect the input data into a NumPy array
                input_data = np.array([[rating, reviews, jobs, interviews, job_security, salary_benefits, 
                                        skill_development, company_culture, promotions]])
                
                # Scale the input data using the same scaler used for training
                input_data_scaled = scaler.transform(input_data)
                
                # Make a prediction using the trained model 
                prediction = dt_model.predict(input_data_scaled)[0]
                
                message = {
                1: "Your company is predicted to have good work-life balance! ",
                0: "Your company is predicted to be lacking in work-life balance..."
                }
                if prediction == 1:
                    st.success(message[prediction])
                else:
                    st.error(message[prediction])
                    
    with margin2:
        st.empty()