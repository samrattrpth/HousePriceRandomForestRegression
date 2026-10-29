"""Compare regression models on the same leakage-aware 80/20 split.
Run: python evaluate_models.py
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Bengaluru_House_Data.csv"

def parse_sqft(value):
    if pd.isna(value): return np.nan
    text = str(value).strip().replace(",", "")
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums: return np.nan
    vals = [float(n) for n in nums]
    return sum(vals) / len(vals) if len(vals) > 1 else vals[0]

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["location", "size", "total_sqft", "bath", "price"]).copy()
    df["total_sqft"] = df["total_sqft"].apply(parse_sqft)
    df["BHK"] = pd.to_numeric(df["size"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df["bath"] = pd.to_numeric(df["bath"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["location", "total_sqft", "BHK", "bath", "price"])
    df = df[(df["total_sqft"] > 0) & (df["BHK"] > 0) & (df["bath"] > 0) & (df["price"] > 0)]
    df["location"] = df["location"].astype(str).str.strip()
    df = df[(df["total_sqft"] / df["BHK"] >= 150) & (df["total_sqft"] / df["BHK"] <= 3000)]
    df = df[df["bath"] <= df["BHK"] + 2]
    return df

def main():
    df = load_data()
    X = df[["location", "total_sqft", "bath", "BHK"]].copy()
    y = df["price"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    numeric = ["total_sqft", "bath", "BHK"]
    categorical = ["location"]
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, min_samples_leaf=5),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=2),
    }
    rows = []
    for name, estimator in models.items():
        prep = ColumnTransformer([
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
        ])
        pipe = Pipeline([("preprocessor", prep), ("regressor", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({"Model": name,
                     "MAE (₹ lakh)": mean_absolute_error(y_test, pred),
                     "RMSE (₹ lakh)": np.sqrt(mean_squared_error(y_test, pred)),
                     "R2": r2_score(y_test, pred)})
    result = pd.DataFrame(rows).sort_values("MAE (₹ lakh)")
    result.to_csv(BASE_DIR / "model_comparison_results.csv", index=False)
    print("\\nModel comparison on same held-out test split (prices in ₹ lakh):")
    print(result.to_string(index=False, formatters={"MAE (₹ lakh)": "{:.3f}".format,
        "RMSE (₹ lakh)": "{:.3f}".format, "R2": "{:.4f}".format}))
    print(f"\\nSaved: {BASE_DIR / 'model_comparison_results.csv'}")

if __name__ == "__main__":
    main()
