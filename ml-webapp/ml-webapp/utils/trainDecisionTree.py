# utils/trainDecisionTree.py
import sys
import json
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import os

def main():
    # Read JSON data from stdin
    input_data = sys.stdin.read()
    try:
        data = json.loads(input_data)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"JSON Decode Error: {str(e)}"}))
        sys.exit(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # --- Update these column names based on your worker_dataset.csv ---
    # For example, let’s assume the features are:
    # "Experience" (years), "Education_Level" (numeric score), "Age"
    # and target is "Salary"
    # If your dataset has different columns, change these accordingly.
    numerical_cols = ['Age', 'Salary', 'Years_of_Experience', 'Performance_Score']
    print("✅ Selected Numerical Columns:", numerical_cols, file=sys.stderr)
    sys.stderr.flush()

    df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    # Drop rows with missing values in these columns
    df.dropna(subset=numerical_cols, inplace=True)
    
    # Define features (X) and target (y)
    X = df[["Years_of_Experience", "Performance_Score", "Age"]]
    y = df["Salary"]
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Decision Tree Regressor
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    med_ae = median_absolute_error(y_test, y_pred)
    coefficients = None  
    intercept = None
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "decision_tree_worker.pkl")
    joblib.dump(model, model_path)
    
    # Output the results as JSON
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
    sys.stdout.flush()
    

if __name__ == "__main__":
    main()
