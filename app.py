import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Page settings
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load the real dataset
data = pd.read_csv("student-mat.csv", sep=";")

# Create Pass/Fail result
data["Result"] = data["G3"].apply(
    lambda x: "Pass" if x >= 10 else "Fail"
)

# Select features
X = data[["studytime", "failures", "absences", "G1", "G2"]]
y = data["Result"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Title
st.title("🎓 Student Performance Predictor")

st.write(
    "Predict whether a student is likely to **Pass or Fail** "
    "based on their academic information."
)

st.info(
    "This application uses Machine Learning with Logistic Regression "
    "and the UCI Student Performance dataset."
)

# Student details
st.header("📋 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    studytime = st.number_input(
        "Study Time (1–4)",
        min_value=1,
        max_value=4,
        value=2
    )

    failures = st.number_input(
        "Previous Failures",
        min_value=0,
        max_value=4,
        value=0
    )

    absences = st.number_input(
        "Absences",
        min_value=0,
        max_value=100,
        value=5
    )

with col2:
    g1 = st.number_input(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=10
    )

    g2 = st.number_input(
        "Second Period Grade (G2)",
        min_value=0,
        max_value=20,
        value=10
    )

# Prediction section
st.header("🔮 Prediction")

if st.button("Predict Result", use_container_width=True):

    new_student = pd.DataFrame(
        [[studytime, failures, absences, g1, g2]],
        columns=["studytime", "failures", "absences", "G1", "G2"]
    )

    prediction = model.predict(new_student)
    probability = model.predict_proba(new_student)

    result = prediction[0]

    if result == "Pass":
        pass_probability = probability[0][1] * 100

        st.success(f"🎉 Predicted Result: {result}")
        st.metric("Pass Probability", f"{pass_probability:.2f}%")
        st.progress(pass_probability / 100)

    else:
        fail_probability = probability[0][0] * 100

        st.error(f"⚠️ Predicted Result: {result}")
        st.metric("Fail Probability", f"{fail_probability:.2f}%")
        st.progress(fail_probability / 100)

# Footer
st.divider()

st.caption(
    "Student Performance Predictor | Python • Pandas • Scikit-learn • Streamlit"
)