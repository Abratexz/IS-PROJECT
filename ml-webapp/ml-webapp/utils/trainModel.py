import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, median_absolute_error
import joblib
import os

def main():
    # 1) Read JSON from stdin (already cleaned)
    input_data = sys.stdin.read()
    data = json.loads(input_data)  # e.g. list of dicts

    # 2) Convert to DataFrame
    df = pd.DataFrame(data)

    # (Optional) If needed, do minor numeric coercion
    numerical_cols = ["RAM_GB", "Storage_GB", "Battery_mAh", "Price"]
    df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=numerical_cols, inplace=True)

    # 3) Split
    X = df[["RAM_GB", "Storage_GB", "Battery_mAh"]]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    med_ae = median_absolute_error(y_test, y_pred)
    coefficients = model.coef_.tolist()  # Convert numpy array to list for JSON
    intercept = model.intercept_

    # 6) Save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "linear_regression.pkl")
    joblib.dump(model, model_path)

    # 7) Output
    output = {
        "message": "Model trained successfully",
        "model_path": model_path,
        "MSE": mse,
        "R2": r2,
        "MAE": mae,
        "MedianAE": med_ae,
        "Coefficients": coefficients,
        "Intercept": intercept
        }
    print(json.dumps(output))

if __name__ == "__main__":
    main()
