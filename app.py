import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set Streamlit page config
st.set_page_config(page_title="TurboCare - AI Car Price Predictor", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl")

model = load_model()

# Load logo
logo_path = "turbocare_logo.jpg"
if os.path.exists(logo_path):
    st.image(logo_path, width=150)

st.title("🚗 TurboCare - AI Car Price Prediction")
st.markdown("Enter car details to estimate its **selling price** and get useful analytics.")

# Sidebar input form
with st.sidebar:
    st.header("📋 Car Details")

    price = st.number_input("💰 Present Price (in Lakhs)", min_value=0.0, format="%.2f")
    kms = st.number_input("🛣️ Kilometers Driven", min_value=0)
    owner = st.selectbox("👤 Number of Previous Owners", [0, 1, 2, 3])
    age = st.slider("📅 Car Age (in years)", 0, 30, 5)
    fuel = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller = st.selectbox("🏷️ Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("⚙️ Transmission", ["Manual", "Automatic"])

    uploaded_img = st.file_uploader("📷 Upload car image (optional)", type=["jpg", "jpeg", "png"])
    compare = st.checkbox("🔁 Compare with another car")
    submit = st.button("🔮 Predict Price")

if submit:
    input_data = pd.DataFrame({
        "Present_Price": [price],
        "Kms_Driven": [kms],
        "Owner": [owner],
        "Car_Age": [age],
        "Fuel_Type": [fuel],
        "Seller_Type": [seller],
        "Transmission": [transmission]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"💸 **Estimated Selling Price: ₹ {prediction:.2f} Lakhs**")

    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded Car Image", width=250)

    # 📉 Depreciation curve
    st.subheader("📉 Estimated Depreciation Curve")
    depreciation_years = list(range(0, 11))
    depreciation_prices = [prediction * (0.85 ** y) for y in depreciation_years]
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(depreciation_years, depreciation_prices, marker='o', color='blue')
    ax1.set_xlabel("Years from now")
    ax1.set_ylabel("Estimated Price (₹ Lakhs)")
    ax1.set_title("Depreciation Curve")
    st.pyplot(fig1)

    # 📊 Feature importance
    st.subheader("📊 Feature Importance")
    if os.path.exists("feature_importance.csv"):
        imp_df = pd.read_csv("feature_importance.csv")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x="importance", y="feature", data=imp_df.sort_values(by="importance", ascending=False), ax=ax2)
        ax2.set_title("Model Feature Importance", fontsize=10)
        ax2.set_xlabel("Importance", fontsize=8)
        ax2.set_ylabel("Feature", fontsize=8)
        st.pyplot(fig2)

    # 📝 Log prediction
    log = input_data.copy()
    log["Predicted_Price"] = prediction
    log["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.to_csv("prediction_log.csv", mode="a", header=not os.path.exists("prediction_log.csv"), index=False)

# 🔁 Comparison
if compare:
    st.subheader("🔁 Compare Two Cars")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Car A")
        a_price = st.number_input("A - Price", key="a_price")
        a_kms = st.number_input("A - Kms Driven", key="a_kms")
        a_owner = st.selectbox("A - Owners", [0, 1, 2, 3], key="a_owner")
        a_age = st.slider("A - Age", 0, 30, 5, key="a_age")
        a_fuel = st.selectbox("A - Fuel", ["Petrol", "Diesel", "CNG"], key="a_fuel")
        a_seller = st.selectbox("A - Seller", ["Dealer", "Individual"], key="a_seller")
        a_trans = st.selectbox("A - Transmission", ["Manual", "Automatic"], key="a_trans")

    with col2:
        st.markdown("### Car B")
        b_price = st.number_input("B - Price", key="b_price")
        b_kms = st.number_input("B - Kms Driven", key="b_kms")
        b_owner = st.selectbox("B - Owners", [0, 1, 2, 3], key="b_owner")
        b_age = st.slider("B - Age", 0, 30, 5, key="b_age")
        b_fuel = st.selectbox("B - Fuel", ["Petrol", "Diesel", "CNG"], key="b_fuel")
        b_seller = st.selectbox("B - Seller", ["Dealer", "Individual"], key="b_seller")
        b_trans = st.selectbox("B - Transmission", ["Manual", "Automatic"], key="b_trans")

    if st.button("Compare Cars"):
        car_a = pd.DataFrame({
            "Present_Price": [a_price],
            "Kms_Driven": [a_kms],
            "Owner": [a_owner],
            "Car_Age": [a_age],
            "Fuel_Type": [a_fuel],
            "Seller_Type": [a_seller],
            "Transmission": [a_trans]
        })

        car_b = pd.DataFrame({
            "Present_Price": [b_price],
            "Kms_Driven": [b_kms],
            "Owner": [b_owner],
            "Car_Age": [b_age],
            "Fuel_Type": [b_fuel],
            "Seller_Type": [b_seller],
            "Transmission": [b_trans]
        })

        pred_a = model.predict(car_a)[0]
        pred_b = model.predict(car_b)[0]

        st.markdown(f"**Car A Price:** ₹ {pred_a:.2f} Lakhs")
        st.markdown(f"**Car B Price:** ₹ {pred_b:.2f} Lakhs")

# 📬 Contact/feedback form
st.sidebar.title("📬 Contact Us")
name = st.sidebar.text_input("Your Name")
message = st.sidebar.text_area("Feedback / Suggestions")
if st.sidebar.button("Send"):
    st.sidebar.success("✅ Thank you for your feedback!")
