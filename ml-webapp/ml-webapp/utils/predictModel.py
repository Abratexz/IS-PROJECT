import sys
import json
import pandas as pd
import joblib

def main():
    input_data = sys.stdin.read()
    try:
        request = json.loads(input_data)  # e.g. { "RAM_GB": 6, "Storage_GB": 128, ... }
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON"}))
        return

    # Load the trained model
    model_path = "models/linear_regression.pkl"
    model = joblib.load(model_path)

    # Convert input to DataFrame (assuming multiple predictions or single)
    df = pd.DataFrame([request])  # or more rows if needed

    # Predict
    y_pred = model.predict(df)

    # Return result
    output = {
        "prediction": y_pred[0]  # or list(y_pred) if multiple
    }
    print(json.dumps(output))

if __name__ == "__main__":
    main()
