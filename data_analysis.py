# ==========================================
# CAR PRICE PREDICTION PROJECT
# SIMPLE DATA CLEANING AND ANALYSIS
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/Used Car Dataset.csv")

print("Original Dataset Shape:")
print(df.shape)

print("\nFirst Five Rows:")
print(df.head())


# ==========================================
# 2. REMOVE UNNECESSARY COLUMN
# ==========================================

if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)


# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())


# ==========================================
# 4. CHECK DUPLICATE ROWS
# ==========================================

print("\nDuplicate Rows Before Cleaning:")
print(df.duplicated().sum())


# ==========================================
# 5. DEFINE NUMERICAL COLUMNS
# ==========================================

numerical_columns = [
    "seats",
    "kms_driven",
    "manufacturing_year",
    "mileage(kmpl)",
    "engine(cc)",
    "max_power(bhp)",
    "torque(Nm)",
    "price(in lakhs)"
]


# ==========================================
# 6. CONVERT NUMERICAL COLUMNS
# ==========================================

for column in numerical_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ==========================================
# 7. FILL MISSING NUMERICAL VALUES
# ==========================================

for column in numerical_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )


# ==========================================
# 8. REMOVE DUPLICATE ROWS
# ==========================================

df = df.drop_duplicates()


# ==========================================
# 9. REMOVE CLEARLY INCORRECT VALUES
# ==========================================

df = df[
    (df["seats"].between(2, 10)) &
    (df["kms_driven"] > 0) &
    (df["manufacturing_year"].between(1990, 2025)) &
    (df["mileage(kmpl)"].between(5, 50)) &
    (df["engine(cc)"].between(500, 6000)) &
    (df["max_power(bhp)"].between(20, 1000)) &
    (df["torque(Nm)"].between(20, 1500)) &
    (df["price(in lakhs)"].between(0.5, 500))
]


# ==========================================
# 10. CHECK DATA AFTER CLEANING
# ==========================================

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows After Cleaning:")
print(df.duplicated().sum())

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Data Types:")
print(df.dtypes)


# ==========================================
# 11. BASIC STATISTICS
# ==========================================

print("\nBasic Statistics:")
print(df.describe())


# ==========================================
# 12. CATEGORICAL DATA ANALYSIS
# ==========================================

categorical_columns = [
    "insurance_validity",
    "fuel_type",
    "ownsership",
    "transmission"
]

for column in categorical_columns:
    print(f"\nValues in {column}:")
    print(df[column].value_counts())


# ==========================================
# 13. PRICE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="price(in lakhs)",
    kde=True
)

plt.title("Distribution of Car Prices")
plt.xlabel("Price in Lakhs")
plt.ylabel("Number of Cars")
plt.show()


# ==========================================
# 14. PRICE VS KILOMETERS DRIVEN
# ==========================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="kms_driven",
    y="price(in lakhs)"
)

plt.title("Car Price vs Kilometers Driven")
plt.xlabel("Kilometers Driven")
plt.ylabel("Price in Lakhs")
plt.show()


# ==========================================
# 15. PRICE VS MANUFACTURING YEAR
# ==========================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="manufacturing_year",
    y="price(in lakhs)"
)

plt.title("Car Price vs Manufacturing Year")
plt.xlabel("Manufacturing Year")
plt.ylabel("Price in Lakhs")
plt.show()


# ==========================================
# 16. PRICE BY FUEL TYPE
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="fuel_type",
    y="price(in lakhs)"
)

plt.title("Car Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Price in Lakhs")
plt.xticks(rotation=30)
plt.show()


# ==========================================
# 17. CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(10, 7))

correlation = df[numerical_columns].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Numerical Features")
plt.show()


# ==========================================
# 18. SAVE CLEANED DATASET
# ==========================================

df.to_csv(
    "data/cleaned_used_car_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
print("File: data/cleaned_used_car_data.csv")

print("Final dataset shape:", df.shape)
print(df.head())