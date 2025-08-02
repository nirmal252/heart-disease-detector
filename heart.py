import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open('models/random_model.pkl', 'rb') as f:
    model = pickle.load(f)

# App config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# Custom style
st.markdown(
    """
    <style>
    .main {
        background-color: #000000;
        color: #FF0000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("❤️ Heart Disease Detector")
st.subheader("Powered by Random Forest Classifier")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["EDA", "Risk Calculator", "Recommendations"])

# EDA Page
if page == "EDA":
    st.header("Exploratory Data Analysis")

    # Load the dataset
    df = pd.read_csv("heart.csv")

    # Basic info
    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Data Info")
    st.write(df.describe())

    # Correlation heatmap (only numeric columns)
    st.subheader("Correlation Heatmap")
    import seaborn as sns
    import matplotlib.pyplot as plt

    numeric_df = df.select_dtypes(include=['number'])

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='Reds', ax=ax1)
    st.pyplot(fig1)

    # Age distribution
    st.subheader("Age Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Age'], kde=True, color='red', ax=ax2)
    st.pyplot(fig2)

    # Heart disease by sex
    st.subheader("Heart Disease by Sex")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Sex', hue='HeartDisease', data=df, palette='Set1', ax=ax3)
    st.pyplot(fig3)

    # Cholesterol vs Heart Disease
    st.subheader("Cholesterol Levels vs Heart Disease")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='HeartDisease', y='Cholesterol', data=df, ax=ax4)
    st.pyplot(fig4)

elif page== "Risk Calculator":
    st.header("Risk Calculator")
    st.write("Please enter your details:")
    
    # Age
    age = st.slider("Age", 20, 100, 50)
    
    #  Sex (M/F -> 1/0)
    sex = st.selectbox("Sex", ["Male", "Female"])
    sex_encoded = 1 if sex == "Male" else 0
    
    # Mappings must be defined BEFORE they are used!
    # Chest Pain Type map
    chest_pain_map = {"Typical Angina": 3, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 0}
    
    # Resting ECG map
    resting_ecg_map = {"Normal": 1, "ST": 2, "LVH": 0}
    
    # ST Slope map
    st_slope_map = {"Up": 2, "Flat": 1, "Down": 0}
    
    # Chest Pain Type (uses chest_pain_map)
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    chest_pain_encoded = chest_pain_map[chest_pain]
    
    #  Resting Blood Pressure
    resting_bp = st.slider("Resting BP (mm Hg)", 50, 200, 120)
    
    #  Cholesterol
    cholesterol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    
    # Fasting Blood Sugar (>120 mg/dl)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["Yes", "No"])
    fasting_bs_encoded = 1 if fasting_bs == "Yes" else 0
    
    # Resting ECG (uses resting_ecg_map)
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    resting_ecg_encoded = resting_ecg_map[resting_ecg]
    
    # Max Heart Rate
    max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
    
    # Exercise Induced Angina (Y/N)
    exercise_angina = st.selectbox("Exercise Induced Angina?", ["Y", "N"])
    exercise_angina_encoded = 1 if exercise_angina == "Y" else 0
    
    # Oldpeak (ST depression)
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 7.0, 1.0, step=0.1)
    
    # ST Slope (uses st_slope_map)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    st_slope_encoded = st_slope_map[st_slope]
    
    # Final input array in correct order
    input_data = np.array([[
        age,
        sex_encoded,
        chest_pain_encoded,
        resting_bp,
        cholesterol,
        fasting_bs_encoded,
        resting_ecg_encoded,
        max_hr,
        exercise_angina_encoded,
        oldpeak,
        st_slope_encoded
    ]])
    
    # Predict button
    if st.button("Predict"):
        result = model.predict(input_data)
        prob = model.predict_proba(input_data)[0][1]
    
        if result[0] == 1:
            st.error(f"⚠️ High risk of heart disease. Probability: {prob:.2%}")
        else:
            st.success(f"✅ Low risk of heart disease. Probability: {prob:.2%}")
# Recommendations Page
else:
    st.header("Recommendations")
    st.write("""
    - Maintain healthy cholesterol & BP levels.
    - Regular check-ups for high-risk age groups.
    - Lifestyle modifications: exercise, diet.
    - Consult your doctor for personalized advice.
    """)


