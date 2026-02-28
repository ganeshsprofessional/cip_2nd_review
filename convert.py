import pandas as pd

# Load the giant dataset
giant_df = pd.read_csv("gnn_dqn_ready_dataset.csv")

# Take the first 50,000 rows (Ensure you don't cut off a receiver's timeline mid-way)
small_df = giant_df.head(50000)

# Save as a new file
small_df.to_csv("small_gnn_dqn_dataset.csv", index=False)
print("Saved small dataset!")