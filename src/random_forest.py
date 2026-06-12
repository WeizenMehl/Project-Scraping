import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("../laptops.csv")

# -----------------------------
# Cleaning prices
# -----------------------------
def clean_price(price):
    if pd.isna(price) or price == "":
        return np.nan

    price = str(price)
    price = price.replace("€", "")
    price = price.replace(" ", "")
    price = price.replace(".", "")
    price = price.replace(",", ".")

    return float(price)

df["price_max"] = df["price_max"].apply(clean_price)
df["price_min"] = df["price_min"].apply(clean_price)

df = df.dropna(subset=["price_max"])

# -----------------------------
# Extract RAM in GB
# -----------------------------
df["ram_gb"] = df["ram"].str.extract(r"(\d+)").astype(float)

# -----------------------------
# Extract CPU cores
# -----------------------------
def extract_cpu_cores(cpu):
    if pd.isna(cpu):
        return np.nan

    patterns = [
        r"(\d+)\s*Core",
        r"(\d+)C\/",
        r"(\d+)x"
    ]

    for pattern in patterns:
        match = re.search(pattern, str(cpu))
        if match:
            return float(match.group(1))
    return np.nan

df["cpu_cores"] = df["cpu"].apply(extract_cpu_cores)

# -----------------------------
# Extract max CPU GHz
# -----------------------------
def extract_max_ghz(cpu):
    if pd.isna(cpu):
        return np.nan

    values = re.findall(r"(\d+\.\d+)GHz", str(cpu))

    if not values:
        return np.nan

    return max(map(float, values))

df["cpu_max_ghz"] = df["cpu"].apply(extract_max_ghz)

# -----------------------------
# Extract GPU cores
# -----------------------------
def extract_gpu_cores(gpu):
    if pd.isna(gpu):
        return np.nan

    patterns = [
        r"(\d+)-Core",
        r"(\d+)Xe",
        r"(\d+)CU"
    ]

    for pattern in patterns:
        match = re.search(pattern, str(gpu))
        if match:
            return float(match.group(1))
    return np.nan

df["gpu_cores"] = df["gpu"].apply(extract_gpu_cores)

# -----------------------------
# Clean ppi
# -----------------------------
df["ppi"] = df["ppi"].str.replace("ppi", "", regex=False).astype(float)

# -----------------------------
# Extract refresh rate Hz
# -----------------------------
def extract_hz(value):
    if pd.isna(value):
        return np.nan

    value = str(value).replace("Hz", "")
    numbers = re.findall(r"\d+", value)

    if not numbers:
        return np.nan

    return float(max(map(int, numbers)))

df["hz"] = df["hz"].apply(extract_hz)

# -----------------------------
# Extract resolution width and height
# -----------------------------
df[["res_width", "res_height"]] = df["resolution"].str.split("x", expand=True).astype(float)

# -----------------------------
# Convert aspect ratio to float
# -----------------------------
def aspect_to_float(ratio):
    if pd.isna(ratio):
        return np.nan

    w, h = ratio.split(":")
    return float(w) / float(h)

df["aspect_ratio_num"] = df["aspect_ratio"].apply(aspect_to_float)

# -----------------------------
# Extract screen size
# -----------------------------
df["screen_size"] = df["dimension"].str.extract(r"(\d+\.?\d*)").astype(float)

# -----------------------------
# Encode OS
# -----------------------------
le_os = LabelEncoder()
df["os_enc"] = le_os.fit_transform(df["os"])

# -----------------------------
# Extract brand from name
# -----------------------------
df["brand"] = df["name"].str.split(' ').str[0]
le_brand = LabelEncoder()
df["brand_enc"] = le_brand.fit_transform(df["brand"])

# -----------------------------
# Features for model
# -----------------------------
features = [
    "ram_gb",
    "cpu_cores",
    "cpu_max_ghz",
    "gpu_cores",
    "ppi",
    "hz",
    "res_width",
    "res_height",
    "aspect_ratio_num",
    "screen_size",
    "os_enc",
    "brand_enc"  # <-- NEW
]

X = df[features]
y = df["price_max"]

# Fill missing values with median
X = X.fillna(X.median())

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Random Forest
# -----------------------------
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
print("\nRANDOM FOREST")
print(f"MSE: {mse:.2f}")

print("\nFeature Importances:")
for feat, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"{feat:20} {imp:.3f}")

# -----------------------------
# Example prediction
# -----------------------------
sample = X.iloc[[0]]
prediction = rf.predict(sample)[0]

print("\nEXAMPLE PREDICTION")
print(f"Actual price: {y.iloc[0]:.2f} €")
print(f"Prediction:   {prediction:.2f} €")