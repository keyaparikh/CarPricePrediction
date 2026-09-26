import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗"
)

# Load model and preprocessor
model = joblib.load("model/random_forest_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")

# App title
st.title("🚗 Used Car Price Prediction")
st.write("Enter the details of the car to estimate its resale price.")

# Sidebar inputs
st.sidebar.header("Car Details")

kms_driven = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=300000,
    value=50000,
    step=1000
)

mileage = st.sidebar.number_input(
    "Mileage (kmpl)",
    min_value=5.0,
    max_value=50.0,
    value=18.0,
    step=0.5
)

engine = st.sidebar.number_input(
    "Engine (cc)",
    min_value=500,
    max_value=6000,
    value=1200,
    step=100
)

manufacturing_year = st.sidebar.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2025,
    value=2018,
    step=1
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG", "Electric"]
)

transmission = st.sidebar.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

ownsership = st.sidebar.selectbox(
    "Ownership",
    [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth Owner"
    ]
)

# Prediction button
if st.button("Predict Car Price 🚘"):

    # Prepare input data
    car_data = pd.DataFrame({
        "kms_driven": [kms_driven],
        "mileage(kmpl)": [mileage],
        "engine(cc)": [engine],
        "manufacturing_year": [manufacturing_year],
        "fuel_type": [fuel_type],
        "transmission": [transmission],
        "ownsership": [ownsership]
    })

    # Preprocess the input
    encoded_data = preprocessor.transform(car_data)

    # Predict current used-car price
    predicted_price = model.predict(encoded_data)[0]

    st.success(
        f"Estimated Used Car Price: ₹{predicted_price:.2f} Lakhs"
    )

    # -----------------------------
    # Depreciation Analysis
    # -----------------------------

    st.subheader("📉 Depreciation Analysis")

    current_year = 2026

    # Calculate car age
    car_age = current_year - manufacturing_year

    # Estimate depreciation based on car age
    if car_age <= 0:
        depreciation_rate = 0

    elif car_age == 1:
        depreciation_rate = 25

    elif car_age <= 3:
        depreciation_rate = 42.5

    elif car_age <= 5:
        depreciation_rate = 55

    else:
        depreciation_rate = 60

    # Calculate reduced future value
    remaining_value = predicted_price * (
        1 - depreciation_rate / 100
    )

    # Calculate value lost
    value_lost = predicted_price - remaining_value

    # Display depreciation information
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Car Age", f"{car_age} years")
        st.metric(
            "Depreciation Rate",
            f"{depreciation_rate:.1f}%"
        )

    with col2:
        st.metric(
            "Current Predicted Price",
            f"₹{predicted_price:.2f} Lakhs"
        )
        st.metric(
            "Value After Depreciation",
            f"₹{remaining_value:.2f} Lakhs"
        )

    st.metric(
        "Estimated Value Lost",
        f"₹{value_lost:.2f} Lakhs"
    )

    # Depreciation timeline
    st.write("### 📊 General Market Depreciation Timeline")

    depreciation_data = pd.DataFrame({
        "Car Age": [
            "First Year",
            "Years 2–3",
            "Years 4–5",
            "After 5 Years"
        ],
        "Estimated Depreciation": [
            "20%–30%",
            "Additional 15%–20%",
            "8%–12% annually",
            "Approximately 50%–60%"
        ]
    })

    st.table(depreciation_data)

    st.info(
        "The machine-learning model predicts the current used-car price. "
        "The depreciation calculation is an approximate market-based "
        "estimate. Actual value depends on condition, maintenance, "
        "mileage, brand, and market demand."
    )

    # Display entered car details
    st.subheader("🚘 Entered Car Details")
    st.dataframe(car_data)

    # Create downloadable report
    report = car_data.copy()

    report["predicted_price(in lakhs)"] = predicted_price
    report["car_age"] = car_age
    report["depreciation_rate(%)"] = depreciation_rate
    report["value_after_depreciation(in lakhs)"] = remaining_value
    report["estimated_value_lost(in lakhs)"] = value_lost

    st.download_button(
        label="Download Prediction Report",
        data=report.to_csv(index=False),
        file_name="car_price_prediction.csv",
        mime="text/csv"
    )