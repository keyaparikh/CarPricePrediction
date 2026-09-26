import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model/random_forest_model.pkl")

# Enter details of a car
car_data = pd.DataFrame({
    "kms_driven": [40000],
    "mileage(kmpl)": [20],
    "engine(cc)": [1197]
})

# Predict the price
predicted_price = model.predict(car_data)

print("Predicted Car Price:", round(predicted_price[0], 2), "lakhs")