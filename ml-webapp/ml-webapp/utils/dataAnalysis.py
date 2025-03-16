import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

def main():
    # 1) Read JSON from Node.js (or a CSV file path)
    try:
        input_data = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"JSON Decode Error: {str(e)}"}))
        sys.exit(1)

    # 2) Convert JSON data to pandas DataFrame
    df = pd.DataFrame(input_data)

    # 3) Basic EDA: summary stats
    # df.describe() returns a DataFrame, convert to dict
    summary_stats = df.describe(include='all').to_dict()

    # 4) Generate one or more plots
    #    For example: correlation heatmap for numeric columns
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        corr = numeric_df.corr()
        # Plot correlation heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        
        # Save figure to public/images with a unique timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        plot_filename = f"correlation_{timestamp}.png"
        output_path = os.path.join("public", "images", plot_filename)
        plt.savefig(output_path)
        plt.close()

    else:
        plot_filename = None  # No numeric data => no correlation plot

    # 5) Return results to Node: 
    #    - summary stats
    #    - name of the plot image
    output_json = {
        "summary_stats": summary_stats,
        "plot_filename": plot_filename
    }
    print(json.dumps(output_json))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
