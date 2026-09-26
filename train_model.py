import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load the original dataset
df = pd.read_csv("data/Used Car Dataset.csv")

# 2. Remove unnecessary index column
df = df.drop(columns=["Unnamed: 0"])


# 3. Select the features and target
features = [
    "kms_driven",
    "mileage(kmpl)",
    "engine(cc)",
    "manufacturing_year",
    "fuel_type",
    "transmission",
    "ownsership"
]

target = "price(in lakhs)"


# 4. Keep only the required columns
df = df[features + [target]].copy()


# 5. Define numerical columns
numeric_columns = [
    "kms_driven",
    "mileage(kmpl)",
    "engine(cc)",
    "manufacturing_year",
    "price(in lakhs)"
]


# 6. Convert numerical columns into numbers
for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# 7. Remove rows with missing values
df = df.dropna()


# 8. Remove unreasonable values
df = df[
    (df["kms_driven"] >= 0) &
    (df["kms_driven"] <= 300000) &
    (df["mileage(kmpl)"] >= 5) &
    (df["mileage(kmpl)"] <= 50) &
    (df["engine(cc)"] >= 500) &
    (df["engine(cc)"] <= 6000) &
    (df["manufacturing_year"] >= 1990) &
    (df["manufacturing_year"] <= 2025) &
    (df["price(in lakhs)"] > 0) &
    (df["price(in lakhs)"] < 100)
]


# 9. Separate input features and target
X = df[features]
y = df[target]


# 10. Define categorical columns
categorical_columns = [
    "fuel_type",
    "transmission",
    "ownsership"
]


# 11. Encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# 12. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 13. Apply encoding
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)


# 14. Create the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# 15. Train the model
model.fit(X_train_encoded, y_train)


# 16. Make predictions
y_pred = model.predict(X_test_encoded)


# 17. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Final Dataset Shape:", df.shape)

print("\nRandom Forest Model Results")
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


# 18. Save the model and preprocessor
joblib.dump(
    model,
    "model/random_forest_model.pkl"
)

joblib.dump(
    preprocessor,
    "model/preprocessor.pkl"
)

print("\nModel and preprocessor saved successfully!")