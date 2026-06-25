import pandas as pd
import numpy as np
import re

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("../laptops.csv")

# =====================================================
# CLEAN PRICE COLUMNS
# =====================================================

def clean_price(price):
    if pd.isna(price) or price == "":
        return np.nan

    price = str(price)
    price = price.replace("€", "")
    price = price.replace(" ", "")
    price = price.replace(".", "")
    price = price.replace(",", ".")

    try:
        return float(price)
    except:
        return np.nan

df["price_max"] = df["price_max"].apply(clean_price)
df["price_min"] = df["price_min"].apply(clean_price)

df = df.dropna(subset=["price_max"])

# =====================================================
# RAM (GB)
# =====================================================

df["ram_gb"] = (
    df["ram"]
    .str.extract(r"(\d+)")
    .astype(float)
)

# =====================================================
# CPU CORES
# =====================================================

def extract_cpu_cores(cpu):
    if pd.isna(cpu):
        return np.nan

    cpu = str(cpu)

    # Apple
    match = re.search(r"(\d+)\s*Core", cpu, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # AMD style
    match = re.search(r"(\d+)C\/\d+T", cpu)
    if match:
        return float(match.group(1))

    # Intel hybrid
    matches = re.findall(r"(\d+)(?:P|E|LPE)", cpu)

    if matches:
        return float(sum(map(int, matches)))

    # Qualcomm
    match = re.search(r"(\d+)x\s+Oryon", cpu)
    if match:
        return float(match.group(1))

    return np.nan
df["cpu_cores"] = df["cpu"].apply(extract_cpu_cores)

# =====================================================
# CPU MAX GHZ
# =====================================================

def extract_max_ghz(cpu):
    if pd.isna(cpu):
        return np.nan

    values = re.findall(
        r"(\d+\.\d+)GHz",
        str(cpu)
    )

    if not values:
        return np.nan

    return max(map(float, values))

df["cpu_max_ghz"] = df["cpu"].apply(extract_max_ghz)

# =====================================================
# GPU FEATURES (CORES + VRAM)
# =====================================================

def extract_gpu_cores(gpu):
    if pd.isna(gpu):
        return np.nan

    gpu = str(gpu)

    patterns = [
        r"(\d+)-Core",   # Apple style
        r"(\d+)Xe",      # Intel
        r"(\d+)CU"       # AMD
    ]

    for pattern in patterns:
        match = re.search(pattern, gpu)
        if match:
            return float(match.group(1))

    return np.nan


def extract_vram(gpu):
    """
    Extract VRAM in GB (ONLY for discrete GPUs like NVIDIA/AMD dGPU).
    iGPU + Apple GPU => 0
    """

    if pd.isna(gpu):
        return np.nan

    gpu = str(gpu)

    # NVIDIA / AMD dedicated VRAM (8GB, 12GB, etc.)
    match = re.search(r"(\d+)\s*GB", gpu, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Some formats like "8GB GDDR6"
    match = re.search(r"(\d+)\s*GDDR", gpu, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Apple / iGPU assumed no VRAM
    if "Apple" in gpu or "iGPU" in gpu or "Intel Graphics" in gpu:
        return 0.0

    return np.nan


df["gpu_cores"] = df["gpu"].apply(extract_gpu_cores)
df["vram_gb"] = df["gpu"].apply(extract_vram)

df["gpu_cores"] = df["gpu"].apply(extract_gpu_cores)

# =====================================================
# PPI
# =====================================================

df["ppi"] = (
    df["ppi"]
    .str.replace("ppi", "", regex=False)
    .astype(float)
)

# =====================================================
# REFRESH RATE
# =====================================================

def extract_hz(value):
    if pd.isna(value):
        return np.nan

    value = str(value).replace("Hz", "")

    numbers = re.findall(r"\d+", value)

    if not numbers:
        return np.nan

    return float(max(map(int, numbers)))

df["hz"] = df["hz"].apply(extract_hz)

# =====================================================
# RESOLUTION
# =====================================================

df[["res_width", "res_height"]] = (
    df["resolution"]
    .str.split("x", expand=True)
    .astype(float)
)

# New feature
df["total_pixels"] = (
    df["res_width"] *
    df["res_height"]
)

# =====================================================
# ASPECT RATIO
# =====================================================

def aspect_to_float(ratio):
    if pd.isna(ratio):
        return np.nan

    try:
        w, h = ratio.split(":")
        return float(w) / float(h)
    except:
        return np.nan

df["aspect_ratio_num"] = (
    df["aspect_ratio"]
    .apply(aspect_to_float)
)

# =====================================================
# SCREEN SIZE
# =====================================================

df["screen_size"] = (
    df["dimension"]
    .str.extract(r"(\d+\.?\d*)")
    .astype(float)
)

# =====================================================
# BRAND
# =====================================================

df["brand"] = (
    df["name"]
    .fillna("")
    .str.split()
    .str[0]
)

# =====================================================
# KEEP ONLY USEFUL COLUMNS
# =====================================================

base_features = [
    "ram_gb",
    "cpu_cores",
    "cpu_max_ghz",
    "gpu_cores",
    "vram_gb",
    "ppi",
    "hz",
    "res_width",
    "res_height",
    "total_pixels",
    "aspect_ratio_num",
    "screen_size"
]

# =====================================================
# ONE HOT ENCODE BRAND + OS
# =====================================================

df = pd.get_dummies(
    df,
    columns=["brand", "os"],
    drop_first=True
)

dummy_columns = [
    col for col in df.columns
    if col.startswith("brand_")
    or col.startswith("os_")
]

features = base_features + dummy_columns

# =====================================================
# TRAINING DATA
# =====================================================

X = df[features]

X = X.fillna(X.median(numeric_only=True))

# explicitly fix GPU missing meaning
X["vram_gb"] = X["vram_gb"].fillna(0)
X["gpu_cores"] = X["gpu_cores"].fillna(0)

y = df["price_max"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# RANDOM FOREST
# =====================================================

rf = RandomForestRegressor(
    n_estimators=1000,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = rf.predict(X_test)

# =====================================================
# METRICS
# =====================================================

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"MAE :  {mae:.2f} €")
print(f"RMSE:  {rmse:.2f} €")
print(f"R²  :  {r2:.4f}")

# =====================================================
# CROSS VALIDATION
# =====================================================

cv_scores = cross_val_score(
    rf,
    X,
    y,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("\n==============================")
print("CROSS VALIDATION")
print("==============================")

print("Fold Scores:")
print(cv_scores)

print(f"\nMean R²: {cv_scores.mean():.4f}")
print(f"Std Dev: {cv_scores.std():.4f}")

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\n==============================")
print("TOP 20 FEATURES")
print("==============================")

print(importance_df.head(20))

# =====================================================
# PREDICTION ERROR ANALYSIS
# =====================================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

results["Error"] = (
    results["Predicted"] -
    results["Actual"]
)

results["Abs_Error"] = (
    results["Error"].abs()
)

print("\n==============================")
print("SAMPLE PREDICTIONS")
print("==============================")

print(results.head(15))

print("\nAverage Error:")
print(results["Abs_Error"].mean())

# =====================================================
# EXAMPLE PREDICTION
# =====================================================

sample = X.iloc[[0]]

prediction = rf.predict(sample)[0]

print("\n==============================")
print("EXAMPLE PREDICTION")
print("==============================")

print(f"Actual Price    : {y.iloc[0]:.2f} €")
print(f"Predicted Price : {prediction:.2f} €")
