import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
df = pd.read_csv("../laptops.csv")


# CLEAN PRICE COLUMNS
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


# RAM (GB)
df["ram_gb"] = (
    df["ram"]
    .str.extract(r"(\d+)")
    .astype(float)
)


# CPU CORES
def extract_cpu_cores(cpu):
    if pd.isna(cpu):
        return np.nan

    cpu = str(cpu)

    match = re.search(r"(\d+)\s*Core", cpu, re.IGNORECASE)
    if match:
        return float(match.group(1))

    match = re.search(r"(\d+)C\/\d+T", cpu)
    if match:
        return float(match.group(1))

    matches = re.findall(r"(\d+)(?:P|E|LPE)", cpu)
    if matches:
        return float(sum(map(int, matches)))

    match = re.search(r"(\d+)x\s+Oryon", cpu)
    if match:
        return float(match.group(1))

    return np.nan


df["cpu_cores"] = df["cpu"].apply(extract_cpu_cores)


# CPU MAX GHZ
def extract_max_ghz(cpu):
    if pd.isna(cpu):
        return np.nan

    values = re.findall(r"(\d+\.\d+)GHz", str(cpu))

    if not values:
        return np.nan

    return max(map(float, values))


df["cpu_max_ghz"] = df["cpu"].apply(extract_max_ghz)


# GPU FEATURES
def extract_gpu_cores(gpu):
    if pd.isna(gpu):
        return np.nan

    gpu = str(gpu)

    patterns = [
        r"(\d+)-Core",
        r"(\d+)Xe",
        r"(\d+)CU"
    ]

    for pattern in patterns:
        match = re.search(pattern, gpu)
        if match:
            return float(match.group(1))

    return np.nan


def extract_vram(gpu):
    if pd.isna(gpu):
        return np.nan

    gpu = str(gpu)

    match = re.search(r"(\d+)\s*GB", gpu, re.IGNORECASE)
    if match:
        return float(match.group(1))

    match = re.search(r"(\d+)\s*GDDR", gpu, re.IGNORECASE)
    if match:
        return float(match.group(1))

    if "Apple" in gpu or "iGPU" in gpu or "Intel Graphics" in gpu:
        return 0.0

    return np.nan


df["gpu_cores"] = df["gpu"].apply(extract_gpu_cores)
df["vram_gb"] = df["gpu"].apply(extract_vram)


# DISPLAY FEATURES
df["ppi"] = df["ppi"].str.replace("ppi", "", regex=False).astype(float)

def extract_hz(value):
    if pd.isna(value):
        return np.nan

    value = str(value).replace("Hz", "")
    numbers = re.findall(r"\d+", value)

    if not numbers:
        return np.nan

    return float(max(map(int, numbers)))


df["hz"] = df["hz"].apply(extract_hz)


df[["res_width", "res_height"]] = (
    df["resolution"]
    .str.split("x", expand=True)
    .astype(float)
)

df["total_pixels"] = df["res_width"] * df["res_height"]


def aspect_to_float(ratio):
    if pd.isna(ratio):
        return np.nan

    try:
        w, h = ratio.split(":")
        return float(w) / float(h)
    except:
        return np.nan


df["aspect_ratio_num"] = df["aspect_ratio"].apply(aspect_to_float)

df["screen_size"] = (
    df["dimension"]
    .str.extract(r"(\d+\.?\d*)")
    .astype(float)
)


df["brand"] = df["name"].fillna("").str.split().str[0]


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


df = pd.get_dummies(df, columns=["brand", "os"], drop_first=True)

dummy_columns = [
    col for col in df.columns
    if col.startswith("brand_") or col.startswith("os_")
]

features = base_features + dummy_columns


X = df[features]
X = X.fillna(X.median(numeric_only=True))

X["vram_gb"] = X["vram_gb"].fillna(0)
X["gpu_cores"] = X["gpu_cores"].fillna(0)

y = df["price_max"]


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)


# MODEL
rf = RandomForestRegressor(
    n_estimators=1000,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)


# METRICS
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MODEL PERFORMANCE")
print(f"MAE :  {mae:.2f} €")
print(f"RMSE:  {rmse:.2f} €")
print(f"R²  :  {r2:.4f}")


# CROSS VALIDATION
cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2", n_jobs=-1)

print("CROSS VALIDATION")

print(cv_scores)
print(f"Mean R²: {cv_scores.mean():.4f}")

# RESULTS TABLE
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

results["Error"] = results["Predicted"] - results["Actual"]
results["Abs_Error"] = results["Error"].abs()

print("\nSAMPLE PREDICTIONS")
print(results.head(15))

sns.set_style("whitegrid")

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--")

plt.xlabel("Actual Price (€)")
plt.ylabel("Predicted Price (€)")
plt.title("Actual vs Predicted Prices")
plt.tight_layout()
plt.savefig("prediction.png", dpi=300, bbox_inches="tight")
plt.close()

# FEATURE IMPORTANCE
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(
    data=importance_df.head(15),
    x="Importance",
    y="Feature"
)
plt.title("Top Feature Importance")
plt.tight_layout()
plt.savefig("features", dpi=300, bbox_inches="tight")
plt.close()


# EXAMPLE PREDICTION
sample = X.iloc[[0]]
prediction = rf.predict(sample)[0]

print("\nEXAMPLE PREDICTION")
print(f"Actual    : {y.iloc[0]:.2f} €")
print(f"Predicted : {prediction:.2f} €")