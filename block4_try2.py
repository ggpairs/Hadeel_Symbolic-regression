# STEP 1 - Load and check data
import pandas as pd
import numpy as np

path = "formatted_choice_data_with_block4.csv"
df = pd.read_csv(path)

cols_needed = ["user_id","Choice","x1","y1","y2","Trial","block"]
df = df[cols_needed].dropna().copy()  # keep only the required columns, remove rows with missing values in those columns, and make a clean independent copy to work with

for c in ["Choice","x1","y1","y2","Trial","block"]:
    df[c] = pd.to_numeric(df[c]) # ensure those columns are clean, numeric, and ready for modelling

# Standardize time/repetition for later modeling
df["Trial_z"] = (df["Trial"] - df["Trial"].mean()) / df["Trial"].std()
df["block_z"] = (df["block"] - df["block"].mean()) / df["block"].std()

n_rows, n_cols = df.shape
n_users = df["user_id"].nunique()
desc = df[["Choice","x1","y1","y2","Trial","block","Trial_z","block_z"]].describe()

print("Rows, Cols:", (n_rows, n_cols))
print("Unique participants (user_id):", n_users)
print("\nColumn names:", list(df.columns))
print("\nPreview (first 8 rows):")
print(df.head(8).to_string(index=False))

print("\nSummary stats (key columns):")
print(desc.to_string())

# STEP 2 - Fit a Prospect Theory (PT) baseline
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score

def softplus(z):  # keep α, β, λ > 0
    return np.log1p(np.exp(z))

def sigmoid(z): # turn a score into probability
    return 1.0 / (1.0 + np.exp(-z))

def pt_value(x, alpha, beta, lam): # prospect theory value function
    x = np.asarray(x)
    return np.where(x >= 0, np.power(x, alpha), -lam * np.power(np.abs(x), beta))

def pt_deltaV_rows(rows, alpha, beta, lam, p=0.5): # Prospect theory value difference between risky and sure options; our task is 50/50, so p = 0.5
    Vrisky = p * pt_value(rows["y1"], alpha, beta, lam) + (1-p) * pt_value(rows["y2"], alpha, beta, lam) # subjective value of risky choices
    Vsure  = pt_value(rows["x1"], alpha, beta, lam) # PT value of the sure amount
    return Vrisky - Vsure

def fit_pt_baseline_pooled(dfin):
    y = dfin["Choice"].astype(int).values
    def nll(theta):
        alpha = softplus(theta[0]) # 0 means the 1st element in the parameter vector theta
        beta  = softplus(theta[1])
        lam   = softplus(theta[2])
        b     = theta[3]
        bias  = theta[4]
        dV = pt_deltaV_rows(dfin, alpha, beta, lam, p=0.5)
        logits = b * dV + bias
        p = sigmoid(logits)
        eps = 1e-9
        ll = y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
        reg = 1e-4 * (theta @ theta)  # tiny L2 for numerical stability
        return -(ll.sum()) + reg
    init = np.array([np.log(np.expm1(0.8)),  # α≈0.8 give the optimizer starting guesses with the common numbers
                     np.log(np.expm1(0.8)),  # β≈0.8
                     np.log(np.expm1(1.8)),  # λ≈1.8
                     1.0, 0.0])              # β_choice, bias
    res = minimize(nll, init, method="L-BFGS-B", options={"maxiter":3000})
    th = res.x
    return {
        "alpha": float(softplus(th[0])),
        "beta": float(softplus(th[1])),
        "lambda": float(softplus(th[2])),
        "beta_choice": float(th[3]),
        "bias": float(th[4]),
        "success": res.success
    }
# Fit pooled PT baseline
pt_params = fit_pt_baseline_pooled(df)

# Cross-validated performance (GroupKFold by user_id)
gkf = GroupKFold(n_splits=5) # keep all trials from the same participant in the same fold
groups = df["user_id"].astype("category").cat.codes.values

ll_list, auc_list = [], []
for tr, te in gkf.split(df, df["Choice"], groups):
    dtr, dte = df.iloc[tr].copy(), df.iloc[te].copy()
    ppars = fit_pt_baseline_pooled(dtr) # fit the prospect theory baseline only on the training data to get parameters
    dte["dV_PT"] = pt_deltaV_rows(dte, ppars["alpha"], ppars["beta"], ppars["lambda"], p=0.5)
    logits = ppars["beta_choice"] * dte["dV_PT"].values + ppars["bias"]
    p = sigmoid(logits)
    ll_list.append(log_loss(dte["Choice"].astype(int).values, p, labels=[0,1])) # compute log loss
    try: # compute AUC
        auc_list.append(roc_auc_score(dte["Choice"].astype(int).values, p))
    except ValueError:
        auc_list.append(np.nan)

print("PT baseline parameters (pooled MLE):")
for k in ["alpha","beta","lambda","beta_choice","bias"]:
    print(f"  {k:>12}: {pt_params[k]:.4f}")
print("\nPT baseline cross-validated performance (5-fold GroupKFold):")
print(f"  Mean log loss (lower is better): {np.mean(ll_list):.4f}")
print(f"  Mean AUC      (higher is better): {np.nanmean(auc_list):.4f}")

# STEP 3 — Symbolic feature library + Sparse Logistic Selection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Add ΔV_PT using pooled parameters from Step 2
df["dV_PT"] = pt_deltaV_rows(df, pt_params["alpha"], pt_params["beta"], pt_params["lambda"], p=0.5)

# Build the library
lib_cols = [
    "dV_PT",
    "Trial_z",
    "block_z",
    "dV_PT_x_Trial",
    "dV_PT_x_block",
    "Trial_z2",
    "block_z2",
]
df["dV_PT_x_Trial"] = df["dV_PT"] * df["Trial_z"] # Interaction between Prospect theory and Trial (time flow)
df["dV_PT_x_block"] = df["dV_PT"] * df["block_z"] # Interaction between Prospect theory and Block (Repetition)
df["Trial_z2"] = df["Trial_z"] ** 2 # Square of Trial
df["block_z2"] = df["block_z"] ** 2 # Square of Block

X_base = df[["dV_PT"]].values
X_full = df[lib_cols].values
y = df["Choice"].astype(int).values
groups = df["user_id"].astype("category").cat.codes.values

# Standardize (important for L1)
sc_base = StandardScaler(); Xb = sc_base.fit_transform(X_base)
sc_full = StandardScaler(); Xf = sc_full.fit_transform(X_full)

def group_cv_sparse(X, y, groups, C_grid=(0.01, 0.1, 1.0, 10.0)): # "C" is the candidate regularisation strength: the weight of the simplicity penalty we impose during training
    gkf = GroupKFold(n_splits=5)
    rows, best = [], None
    for C in C_grid:
        ll_list, auc_list = [], []
        sel = np.zeros(X.shape[1], dtype=float)
        for tr, te in gkf.split(X, y, groups):
            clf = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=5000)
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:,1]
            ll_list.append(log_loss(y[te], p, labels=[0,1])) # compute log loss on the held-out fold and store it
            try: auc_list.append(roc_auc_score(y[te], p)) # compute AUC on the held-out fold and store it
            except ValueError: auc_list.append(np.nan)
            sel += (np.abs(clf.coef_[0]) > 1e-8).astype(float)
        rec = {"C": C, "CV_logloss": float(np.mean(ll_list)), "CV_AUC": float(np.nanmean(auc_list)), "sel_freq": sel / 5.0}
        rows.append(rec)
        if (best is None) or (rec["CV_logloss"] < best["CV_logloss"]):
            best = rec
    return rows, best

# Evaluate
base_grid, base_best = group_cv_sparse(Xb, y, groups) # evaluate the baseline model that uses only the standardised dV_PT feature (Xb)
full_grid, full_best = group_cv_sparse(Xf, y, groups) # evaluate the full feature library (Xf: dV_PT, Trial_z, block_z, interactions, squares)

def print_grid(grid, names):
    print("C, CV_logloss, CV_AUC, selection_frequencies")
    for g in grid:
        s = ", ".join([f"{names[i]}:{g['sel_freq'][i]:.1f}" for i in range(len(names))])
        print(f"{g['C']:>4}:  {g['CV_logloss']:.4f}  |  {g['CV_AUC']:.4f}  |  {s}")

print("Baseline (dV_PT only):")
print_grid(base_grid, ["dV_PT"])
print("\nFull library (dV_PT + time/repetition terms):")
print_grid(full_grid, lib_cols)

print("\nBest C by CV log loss (baseline):", base_best["C"])
print("Best C by CV log loss (full):", full_best["C"])

# Fit final model at the best C for the full library
C_star = full_best["C"]
final = LogisticRegression(penalty="l1", solver="saga", C=C_star, max_iter=5000)
final.fit(Xf, y)
coefs = final.coef_[0]
intercept = float(final.intercept_[0])

print("\nSelected non-zero features at best C (full library):")
for name, w in zip(lib_cols, coefs):
    if abs(w) > 1e-8:
        print(f"  {name:>15}: {w:+.4f}")
print(f"  {'intercept':>15}: {intercept:+.4f}")