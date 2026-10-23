
import streamlit as st
import pandas as pd
import joblib

# Load the complete trained pipeline
model = joblib.load("insurance_model.pkl")

st.title("Medical Insurance Charges Prediction")
st.write("Enter the person's details to estimate insurance charges.")

age = st.number_input("Age", min_value=18, max_value=100, value=25)
sex = st.selectbox("Sex", ["female", "male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker?", ["no", "yes"])
region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

if st.button("Predict Insurance Charges"):
    person = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    prediction = model.predict(person)[0]

    st.success(f"Estimated charges: {prediction:,.2f}")
    st.caption("This will be your estimated charges .")
