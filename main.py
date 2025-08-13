import pandas as pd
import numpy as np
from pysr import PySRRegressor

df = pd.read_csv("formatted_choice_data_with_block4.csv")

print(df.head())
print(df.columns)

def classify_condition(x1):
    if x1 < 0:
        return "loss_only"
    elif x1 == 0:
        return "mixed"
    else:
        return "win_only"

df["condition"] = df["x1"].apply(classify_condition)

gambling_rate_df = (
    df.groupby(["Subject", "block", "condition"])["Choice"]
    .mean()
    .reset_index()
    .rename(columns={"Choice": "gambling_rate"})
)

# Count blocks per subject per condition
check_counts = gambling_rate_df.groupby(["Subject", "condition"])["block"].nunique().reset_index()

print(check_counts.head())
print(check_counts["block"].value_counts())  # Should mostly be 3

# Calculate gambling rate
gambling_rate_df = (
    df.groupby(["Subject", "condition", "block"])
      .agg(gambling_rate=("Choice", lambda x: (x == 1).mean()))
      .reset_index()
)

print(gambling_rate_df.head())

# Split dataset by condition
loss_only_df = gambling_rate_df[gambling_rate_df["condition"] == "loss_only"]
mixed_df     = gambling_rate_df[gambling_rate_df["condition"] == "mixed"]
win_only_df  = gambling_rate_df[gambling_rate_df["condition"] == "win_only"]

# Check sizes
print(len(loss_only_df), len(mixed_df), len(win_only_df))

# Prepare data for PySR (Start from the win_only condition)
## Subset to win_only and keep needed columns
win_label = "win_only"
df_win = (
    gambling_rate_df.loc[gambling_rate_df["condition"] == win_label, ["block", "gambling_rate"]]
    .dropna()
    .copy()
)

## Ensure numeric types
df_win["block"] = pd.to_numeric(df_win["block"], errors="coerce")
df_win["gambling_rate"] = pd.to_numeric(df_win["gambling_rate"], errors="coerce")
df_win.dropna(inplace=True)

## Center block (helps SR find simpler formulas) ---
block_mean = df_win["block"].mean()
df_win["block_c"] = df_win["block"] - block_mean

## Build X (2D) and y (1D)
X = df_win[["block_c"]].values
y = df_win["gambling_rate"].values

print("X shape:", X.shape, "y shape:", y.shape)
if X.shape[0] == 0:
    raise ValueError("win_only subset is empty. Check earlier filtering steps.")

# Run PySR (simple, interpretable search space)
model = PySRRegressor(
    niterations=400,   # the total number of generations the algorithm will evolve equations for
    timeout_in_seconds=180, # stop iteration after 3 minutes
    population_size=200,      # how many candidate equations exist in the "population" at any given time
    binary_operators=["+", "-", "*", "/"], # the allowed binary math operations that take two inputs
    unary_operators=[],               # the allowed unary math operations that take one input (sin, cos, exp, log, sqrt)
    model_selection="best", # how PySR picks the final equation. 'best': pick the one with the lowest loss/error one the training data. 'score': trade-off between accuracy and complexity
    elementwise_loss="(x, y) -> (x - y)^2",   # the loss function that evaluates equation accuracy - here is Mean Squared Error
    maxsize=20,                       # the maximum allowed complexity (the number of nodes in the equation's expression tree) of an equation
    procs=0,                          # number of CPU processes to use
    progress=True,
    verbosity=1, # how much extra text output you get (0,1,2)
    random_state=0, # seed for the random number generator - if different seeds give the same equation, the result is more reliable
)

model.fit(X, y)
print(model)                # table of discovered formulas
best_eq = model.get_best()
print("Best equation:", best_eq)

# predict at block 1-3
blocks = np.array([1, 2, 3])
blocks_c = blocks - block_mean
pred = model.predict(blocks_c.reshape(-1, 1))
print("Blocks:", blocks, "Predicted gambling rate:", pred)

# save equations
model.equations_.to_csv("win_only_equations.csv", index=False)
print("Saved to win_only_equations.csv")