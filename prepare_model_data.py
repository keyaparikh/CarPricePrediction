import pandas as pd

# Load original dataset
df = pd.read_csv("data/Used Car Dataset.csv")

# Remove unnecessary column
df = df.drop(columns=["Unnamed: 0"])

# Select simpler and more reliable columns
features = [
    "kms_driven",
    "mileage(kmpl)",
    "engine(cc)"
]

target = "price(in lakhs)"

# Convert selected columns into numbers
for column in features + [target]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows where important values are missing
df = df.dropna(subset=features + [target])

# Keep only reasonable car prices
df = df[
    (df["price(in lakhs)"] > 0) &
    (df["price(in lakhs)"] < 100)
]

# Remove duplicate rows
df = df.drop_duplicates()

# Display result
print("Model Dataset Shape:")
print(df.shape)

print("\nFirst Five Rows:")
print(df[features + [target]].head())

# Save model-ready dataset
df[features + [target]].to_csv(
    "data/model_ready_data.csv",
    index=False
)

print("\nModel-ready dataset saved successfully!")