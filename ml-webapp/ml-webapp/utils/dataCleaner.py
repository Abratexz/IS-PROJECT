import pandas as pd
import sys
import json

# Read JSON from Node.js
input_data = sys.stdin.read()
data = json.loads(input_data)

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Convert empty strings to NaN before dropping missing values
df.replace("", pd.NA, inplace=True)

# Drop missing values
df = df.dropna()

# Convert back to JSON and send to Node.js
print(json.dumps(df.to_dict(orient="records")))
sys.stdout.flush()
