import sys
import json
import pandas as pd
import joblib

def main():
    # Read input JSON from stdin
    input_data = sys.stdin.read()
    try:
        # Expecting a JSON object with keys matching model features, e.g.,
        # {"Years_of_Experience": 5, "Performance_Score": 3, "Age": 30}
        new_data = json.loads(input_data)
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON input"}))
        sys.exit(1)

    # Load the trained model
    model_path = "models/decision_tree_worker.pkl"
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {e}"}))
        sys.exit(1)

    # Prepare input data as a DataFrame;
    # Ensure the keys exactly match those used during training
    df = pd.DataFrame([new_data], columns=["Years_of_Experience", "Performance_Score", "Age"])
    
    # Predict the target variable (for example, Salary)
    try:
        prediction = model.predict(df)[0]
    except Exception as e:
        print(json.dumps({"error": f"Prediction failed: {e}"}))
        sys.exit(1)

    # Return the prediction as JSON
    output = {"prediction": prediction}
    print(json.dumps(output))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
