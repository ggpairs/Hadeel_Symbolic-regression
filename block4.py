# block4.py — win_only symbolic regression (PySR-compatible)

import pandas as pd
import numpy as np
from pysr import PySRRegressor

# === Load data ===
df = pd.read_csv("formatted_choice_data_with_block4.csv")
print(df.head())
print(df.columns)

# === Raw x1 sign counts (trial-level) ===
num_neg = (df["x1"] < 0).sum()
num_zero = (df["x1"] == 0).sum()
num_pos = (df["x1"] > 0).sum()
print(f"Negative x1 (loss_only): {num_neg}")
print(f"Zero x1 (mixed): {num_zero}")
print(f"Positive x1 (win_only): {num_pos}")

# === Classify condition from x1 ===
def classify_condition(x1):
    if x1 < 0:
        return "loss_only"
    elif x1 == 0:
        return "mixed"
    else:
        return "win_only"

df["condition"] = df["x1"].apply(classify_condition)

# === Aggregate gambling rate: Subject × condition × block ===
gambling_rate_df = (
    df.groupby(["Subject", "condition", "block"])
      .agg(gambling_rate=("Choice", lambda x: (x == 1).mean()))
      .reset_index()
)

# completeness check
check_counts = gambling_rate_df.groupby(["Subject", "condition"])["block"].nunique().reset_index()
print("Block counts per subject-condition (should mostly be 3):")
print(check_counts["block"].value_counts())

# === Keep only win_only (others are empty in this dataset) ===
win_label = "win_only"
win_only_df = gambling_rate_df.loc[gambling_rate_df["condition"] == win_label].copy()
if win_only_df.empty:
    raise ValueError("No rows found for win_only after aggregation.")

print(f"win_only aggregated rows: {len(win_only_df)}")

# === Prepare X (block) and y (gambling_rate) ===
df_win = win_only_df[["block", "gambling_rate"]].dropna().copy()
df_win["block"] = pd.to_numeric(df_win["block"], errors="coerce")
df_win["gambling_rate"] = pd.to_numeric(df_win["gambling_rate"], errors="coerce")
df_win.dropna(inplace=True)

# Center block (helps search)
block_mean = df_win["block"].mean()
df_win["block_c"] = df_win["block"] - block_mean

X = df_win[["block_c"]].values   # 2D
y = df_win["gambling_rate"].values  # 1D
print("X shape:", X.shape, "y shape:", y.shape)
if X.shape[0] == 0:
    raise ValueError("win_only subset became empty after cleaning.")

# === Run PySR ===
model = PySRRegressor(
    niterations=400,              # the total number of generations the algorithm will evolve equations for
    timeout_in_seconds=180,             # stop iteration after 3 minutes
    binary_operators=["+", "-", "*", "/"], # the allowed binary math operations that take two inputs
    unary_operators=[],                 # the allowed unary math operations that take one input (sin, cos, exp, log, sqrt)
    model_selection="best",   # how PySR picks the final equation. 'best': pick the one with the lowest loss/error one the training data. 'score': trade-off between accuracy and complexity
    elementwise_loss="(x, y) -> (x - y)^2", # the loss function that evaluates equation accuracy - here is Mean Squared Error
    maxsize=20,   # the maximum allowed complexity (the number of nodes in the equation's expression tree) of an equation
    procs=0,                            # number of CPU processes to use
    progress=True,
    verbosity=1,  # how much extra text output you get (0,1,2)
    random_state=0,  # seed for the random number generator - if different seeds give the same equation, the result is more reliable
    output_directory="pysr_win_only",
    warm_start=True
)

model.fit(X, y)
print(model)

best_eq = model.get_best()
print("Best equation:", best_eq)

# === Predict at blocks 1–3 (remember we centered) ===
blocks = np.array([1, 2, 3])
blocks_c = blocks - block_mean
pred = model.predict(blocks_c.reshape(-1, 1))
print("Blocks:", blocks, "Predicted gambling rate:", pred)

# === Save equations (hall of fame) ===
model.equations_.to_csv(f"{win_label}_equations.csv", index=False)
print(f"Saved to {win_label}_equations.csv")
