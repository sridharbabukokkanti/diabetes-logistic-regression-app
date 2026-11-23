
cat << 'EOF' > app.py
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

@st.cache_resource
def load_or_train_model():
    df = pd.read_csv("diabetes.csv")
    zero_missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_missing_cols] = df[zero_missing_cols].replace(0, np.nan)

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

model = load_or_train_model()

st.title("🩺 Diabetes Risk Prediction App")
st.markdown("Enter patient diagnostic measurements below to estimate the probability of **Type 2 Diabetes**.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=250, value=120, step=1)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=72, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=29, step=1)

with col2:
    insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=125, step=1)
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=32.0, step=0.1, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=2.50, value=0.47, step=0.01, format="%.3f")
    age = st.number_input("Age (years)", min_value=21, max_value=100, value=33, step=1)

threshold = st.sidebar.slider("Classification Threshold", 0.10, 0.90, 0.50, 0.05)

if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }])
    zero_missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    input_df[zero_missing_cols] = input_df[zero_missing_cols].replace(0, np.nan)

    prob = model.predict_proba(input_df)[0, 1]
    pred = int(prob >= threshold)

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Estimated Diabetes Probability", f"{prob:.1%}")
    c2.metric("Decision Threshold", f"{threshold:.0%}")
    st.progress(float(prob))

    if pred == 1:
        st.error(f"**High Risk (Outcome = 1):** Probability ({prob:.1%}) >= {threshold:.0%}")
    else:
        st.success(f"**Low Risk (Outcome = 0):** Probability ({prob:.1%}) < {threshold:.0%}")
EOF
