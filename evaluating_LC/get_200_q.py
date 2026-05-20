import sys
import pandas as pd

# 1. Get input size argument
if len(sys.argv) < 2:
    print("Error: Please provide a size argument. Example: python script.py small")
    sys.exit(1)

size = sys.argv[1]
input_path = f"evaluating_LC/outputs/su_outputs/su_results_google_{size}.csv"

# 2. Read the data
try:
    su_data = pd.read_csv(input_path, encoding="utf-8")
except FileNotFoundError:
    print(f"Error: File not found at {input_path}")
    sys.exit(1)

# 3. Define the target levels we care about
target_levels = [
    "high level of confidence",
    "moderate level of confidence",
    "low level of confidence",
    "lowest level of confidence",
    "complete uncertainty, reject to reply",
]

# 4. Filter for only rows matching our target levels
filtered_df = su_data[su_data["level"].isin(target_levels)]

# 5. Group by 'level' and take the first 40 rows of each group
questions_df = filtered_df.groupby("level").head(40)

# Optional: If you want random rows instead of the first 40, uncomment below:
# questions_df = filtered_df.groupby('level').apply(lambda x: x.sample(n=min(len(x), 40))).reset_index(drop=True)

# 6. Save the results
save_name = f"evaluating_LC/outputs/200_q_{size}.csv"
questions_df.to_csv(save_name, index=False)
print(f"Successfully saved balanced dataset to {save_name}. Total rows: {len(questions_df)}")