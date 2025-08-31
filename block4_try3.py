import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

# initial setup
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "formatted_choice_data_with_block4.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV not found at: {DATA_PATH}\n"
                            f"Tip: Is the filename correct? Are you running from the right folder?")

df = pd.read_csv(DATA_PATH)


P_COL = None      # set to column name if risky probability varies, else None
P_DEFAULT = 0.5   # default risky prob if P_COL is None
N_SPLITS = 5

# Load & basic prep

df = pd.read_csv(DATA_PATH)
keep = ["user_id","Choice","x1","y1","y2","Trial","block"]
df = df[keep].dropna().copy()
for c in ["Choice","x1","y1","y2","Trial","block"]:
    df[c] = pd.to_numeric(df[c])
df["Choice"] = df["Choice"].astype(int)

def get_p(frame):
    if P_COL and P_COL in frame.columns:
        return frame[P_COL].astype(float).values
    return np.full(len(frame), P_DEFAULT, dtype=float)

# Core helpers
def softplus(z):
    return np.where(z > 30, z, np.log1p(np.exp(z)))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loglik(y, p, eps=1e-12):
    p = np.clip(p, eps, 1-eps)
    return np.sum(y*np.log(p) + (1-y)*np.log(1-p))

def pt_value(x, alpha, beta, lam):
    x = np.asarray(x)
    return np.where(x >= 0, np.power(x, alpha), -lam*np.power(np.abs(x), beta))

def compute_utilities(frame, alpha, beta, lam):
    p_vec = get_p(frame)
    Vr = p_vec*pt_value(frame["y1"].values, alpha, beta, lam) + (1-p_vec)*pt_value(frame["y2"].values, alpha, beta, lam)
    Vs = pt_value(frame["x1"].values, alpha, beta, lam)
    dV = Vr - Vs
    return Vr, Vs, dV

# Fit pooled Prospect Theory (for α,β,λ)

def fit_pt_pooled(rows):
    y = rows["Choice"].values
    def nll(theta):
        alpha = softplus(theta[0]); beta = softplus(theta[1]); lam = softplus(theta[2])
        b = theta[3]; bias = theta[4]   # internal link only for likelihood
        _, _, dV = compute_utilities(rows, alpha, beta, lam)
        p = sigmoid(b*dV + bias)
        return -(loglik(y, p)) + 1e-6*np.sum(theta**2)
    init = np.array([np.log(np.expm1(0.8)), np.log(np.expm1(0.8)), np.log(np.expm1(1.8)), 1.0, 0.0])
    res = minimize(nll, init, method="L-BFGS-B", options={"maxiter":3000})
    th = res.x
    return {"alpha": float(softplus(th[0])), "beta": float(softplus(th[1])),
            "lambda": float(softplus(th[2])), "beta_choice": float(th[3]), "bias": float(th[4])}

pt = fit_pt_pooled(df)

# Preview utilities
Vr_PT, Vs_PT, dV_PT = compute_utilities(df, pt["alpha"], pt["beta"], pt["lambda"])
print("Example PT utilities (first 6):")
print(pd.DataFrame({
    "user_id": df["user_id"][:6], "Trial": df["Trial"][:6], "block": df["block"][:6],
    "x1": df["x1"][:6], "y1": df["y1"][:6], "y2": df["y2"][:6],
    "Vr_PT": Vr_PT[:6], "Vs_PT": Vs_PT[:6], "dV_PT": dV_PT[:6], "Choice": df["Choice"][:6]
}).to_string(index=False))

# Symbolic library (time/repetition) - raw values

LIBRARY = [
    ("Trial",   lambda T,B: T),
    ("block",   lambda T,B: B),
    ("Trial2",  lambda T,B: T**2),
    ("block2",  lambda T,B: B**2),
    ("T_x_B",   lambda T,B: T*B),
]
CANDIDATES = [n for n,_ in LIBRARY]

def fold_raw(train, test):
    """Return raw Trial/block arrays for train and test (no scaling)."""
    tr_T = train["Trial"].values.astype(float)
    tr_B = train["block"].values.astype(float)
    te_T = test["Trial"].values.astype(float)
    te_B = test["block"].values.astype(float)
    return (tr_T, tr_B), (te_T, te_B)

def build_phi(T, B, terms):
    if not terms:
        return np.zeros((len(T), 0))
    cols = []
    for name, fn in LIBRARY:
        if name in terms:
            cols.append(fn(T, B))
    return np.column_stack(cols)

# (A) Additive on ΔV: ΔV_new = ΔV_PT + Φ c
def fit_additive(train, terms, pt):
    y = train["Choice"].values
    _, _, dV0 = compute_utilities(train, pt["alpha"], pt["beta"], pt["lambda"])
    (T,B), _ = fold_raw(train, train)
    Phi = build_phi(T,B,terms)
    def nll(theta):
        k = Phi.shape[1]; c = theta[:k] if k>0 else np.array([])
        b = theta[k]; bias = theta[k+1]
        dV_new = dV0 + (Phi@c if k>0 else 0.0)
        p = sigmoid(b*dV_new + bias)
        return -(loglik(y,p)) + 1e-3*np.sum(c**2)
    init = np.zeros(Phi.shape[1]+2); init[Phi.shape[1]] = 1.0
    return minimize(nll, init, method="L-BFGS-B", options={"maxiter":2000})

def predict_additive(train, test, terms, theta, pt):
    k = len(terms); c = theta[:k] if k>0 else np.array([])
    b = theta[k]; bias = theta[k+1]
    _, _, dV0_te = compute_utilities(test, pt["alpha"], pt["beta"], pt["lambda"])
    (T_tr,B_tr),(T_te,B_te) = fold_raw(train, test)
    Phi_te = build_phi(T_te,B_te,terms)
    dV_new_te = dV0_te + (Phi_te@c if k>0 else 0.0)
    p_te = sigmoid(b*dV_new_te + bias)
    return dV_new_te, p_te

# (B) Multiplicative on ΔV: ΔV_new = ΔV_PT * factor; factor = 1 + softplus(s) - ln2, s = Φ c
def fit_multiplicative(train, terms, pt):
    y = train["Choice"].values
    _, _, dV0 = compute_utilities(train, pt["alpha"], pt["beta"], pt["lambda"])
    (T,B), _ = fold_raw(train, train)
    Phi = build_phi(T,B,terms)
    ln2 = np.log(2.0)
    def nll(theta):
        k = Phi.shape[1]; c = theta[:k] if k>0 else np.array([])
        b = theta[k]; bias = theta[k+1]
        s = Phi@c if k>0 else 0.0
        factor = 1.0 + (np.log1p(np.exp(s)) - ln2)   # =1 at s=0; keeps factor>0
        dV_new = dV0 * factor
        p = sigmoid(b*dV_new + bias)
        return -(loglik(y,p)) + 1e-3*np.sum(c**2)
    init = np.zeros(Phi.shape[1]+2); init[Phi.shape[1]] = 1.0
    return minimize(nll, init, method="L-BFGS-B", options={"maxiter":2000})

def predict_multiplicative(train, test, terms, theta, pt):
    k = len(terms); c = theta[:k] if k>0 else np.array([])
    b = theta[k]; bias = theta[k+1]
    _, _, dV0_te = compute_utilities(test, pt["alpha"], pt["beta"], pt["lambda"])
    (T_tr,B_tr),(T_te,B_te) = fold_raw(train, test)
    Phi_te = build_phi(T_te,B_te,terms)
    ln2 = np.log(2.0)
    s = Phi_te@c if k>0 else 0.0
    factor = 1.0 + (np.log1p(np.exp(s)) - ln2)
    dV_new_te = dV0_te * factor
    p_te = sigmoid(b*dV_new_te + bias)
    return dV_new_te, p_te

# (C) Split additive: Vr_new = Vr_PT + Φ a ; Vs_new = Vs_PT + Φ b  => ΔV_new = ΔV_PT + Φ(a-b)
def fit_split(train, terms, pt):
    y = train["Choice"].values
    Vr0, Vs0, dV0 = compute_utilities(train, pt["alpha"], pt["beta"], pt["lambda"])
    (T,B), _ = fold_raw(train, train)
    Phi = build_phi(T,B,terms)
    def nll(theta):
        k = Phi.shape[1]
        a = theta[:k] if k>0 else np.array([])
        bcoef = theta[k:2*k] if k>0 else np.array([])
        beta_choice = theta[2*k]; bias = theta[2*k+1]
        Vr_new = Vr0 + (Phi@a if k>0 else 0.0)
        Vs_new = Vs0 + (Phi@bcoef if k>0 else 0.0)
        dV_new = Vr_new - Vs_new
        p = sigmoid(beta_choice*dV_new + bias)
        return -(loglik(y,p)) + 1e-3*(np.sum(a**2)+np.sum(bcoef**2))
    init = np.zeros(2*Phi.shape[1] + 2); init[-2] = 1.0
    return minimize(nll, init, method="L-BFGS-B", options={"maxiter":2000})

def predict_split(train, test, terms, theta, pt):
    _, _, dV0_te = compute_utilities(test, pt["alpha"], pt["beta"], pt["lambda"])
    (T_tr,B_tr),(T_te,B_te) = fold_raw(train, test)
    Phi_te = build_phi(T_te,B_te,terms)
    k = len(terms)
    a = theta[:k] if k>0 else np.array([])
    bcoef = theta[k:2*k] if k>0 else np.array([])
    beta_choice = theta[2*k]; bias = theta[2*k+1]
    Vr_adj = Phi_te@a if k>0 else 0.0
    Vs_adj = Phi_te@bcoef if k>0 else 0.0
    dV_new_te = dV0_te + (Vr_adj - Vs_adj)
    p_te = sigmoid(beta_choice*dV_new_te + bias)
    return dV_new_te, p_te

def get_TB_all(frame):
    return frame["Trial"].values.astype(float), frame["block"].values.astype(float)

def final_fit_and_equation(df_all, kind, terms, pt):
    if kind == "A":
        res = fit_additive(df_all, terms, pt); theta = res.x
        k = len(terms); c = theta[:k] if k>0 else np.array([])
        Vr0, Vs0, dV0 = compute_utilities(df_all, pt["alpha"], pt["beta"], pt["lambda"])
        T,B = get_TB_all(df_all); Phi = build_phi(T,B,terms)
        dV_new = dV0 + (Phi@c if k>0 else 0.0)
        eq = "ΔV_new = ΔV_PT" + ("" if k==0 else " + " + " + ".join([f"{w:+.6f}*{t}" for w, t in zip(c, terms)]))
        return dV0, dV_new, eq

    if kind == "B":
        res = fit_multiplicative(df_all, terms, pt); theta = res.x
        k = len(terms); c = theta[:k] if k>0 else np.array([])
        Vr0, Vs0, dV0 = compute_utilities(df_all, pt["alpha"], pt["beta"], pt["lambda"])
        T,B = get_TB_all(df_all); Phi = build_phi(T,B,terms)
        ln2 = np.log(2.0); s = Phi@c if k>0 else 0.0
        factor = 1.0 + (np.log1p(np.exp(s)) - ln2)
        dV_new = dV0 * factor
        fac_str = " * [ 1 + softplus(" + (" + ".join([f"{w:+.6f}*{t}" for w, t in zip(c, terms)]) if k>0 else "0") + ") - ln2 ]"
        eq = "ΔV_new = ΔV_PT" + fac_str
        return dV0, dV_new, eq

    if kind == "C":
        res = fit_split(df_all, terms, pt); theta = res.x
        k = len(terms)
        a = theta[:k] if k>0 else np.array([]); bcoef = theta[k:2*k] if k>0 else np.array([])
        Vr0, Vs0, dV0 = compute_utilities(df_all, pt["alpha"], pt["beta"], pt["lambda"])
        T,B = get_TB_all(df_all); Phi = build_phi(T,B,terms)
        Vr_new = Vr0 + (Phi@a if k>0 else 0.0)
        Vs_new = Vs0 + (Phi@bcoef if k>0 else 0.0)
        dV_new = Vr_new - Vs_new
        eq = ("V_r_new = V_r_PT" + ("" if k==0 else " + " + " + ".join([f"{w:+.6f}*{t}" for w, t in zip(a, terms)])) +
            "\nV_s_new = V_s_PT" + ("" if k==0 else " + " + " + ".join([f"{w:+.6f}*{t}" for w, t in zip(bcoef, terms)])) +
            "\nΔV_new = V_r_new - V_s_new")
        return dV0, dV_new, eq


def terms_str(weights, names):
    return " + ".join(f"{w:+.6f}*{n}" for w, n in zip(weights, names))

def baseline_pseudoR2(df_all, pt, n_splits=5):
    """Use PT ΔV only; in each training fold fit (b, bias), compute McFadden pseudo-R² on the corresponding test fold, then average."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df_all["user_id"].astype("category").cat.codes.values
    R2s = []
    for tr, te in gkf.split(df_all, df_all["Choice"].values, groups):
        dtr = df_all.iloc[tr].copy(); dte = df_all.iloc[te].copy()
        # Fit (b, bias) on the training fold
        _, _, dV_tr = compute_utilities(dtr, pt["alpha"], pt["beta"], pt["lambda"])
        y_tr = dtr["Choice"].values
        def nll(theta):
            b, bias = theta
            p = sigmoid(b*dV_tr + bias)
            return -(loglik(y_tr, p)) + 1e-6*np.sum(theta**2)
        res = minimize(nll, np.array([1.0, 0.0]), method="L-BFGS-B")
        b_hat, bias_hat = res.x
        # Evaluate on the test fold
        _, _, dV_te = compute_utilities(dte, pt["alpha"], pt["beta"], pt["lambda"])
        p_te = sigmoid(b_hat*dV_te + bias_hat)
        ll_model = loglik(dte["Choice"].values, p_te)
        pbar = dtr["Choice"].mean()  # Use the training fold's mean choice rate as the null
        ll_null = loglik(dte["Choice"].values, np.full(len(dte), pbar))
        R2s.append(1.0 - ll_model/ll_null)
    return float(np.mean(R2s))

def mcfadden_cv(df_all, terms, fit_fun, pred_fun, pt, n_splits=5):
    """Run GroupKFold for the model with candidate terms and return the average pseudo-R²."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df_all["user_id"].astype("category").cat.codes.values
    R2s = []
    for tr, te in gkf.split(df_all, df_all["Choice"].values, groups):
        dtr = df_all.iloc[tr].copy(); dte = df_all.iloc[te].copy()
        res = fit_fun(dtr, terms, pt); theta = res.x
        dV_new_te, p_te = pred_fun(dtr, dte, terms, theta, pt)
        ll_model = loglik(dte["Choice"].values, p_te)
        pbar = dtr["Choice"].mean()
        ll_null = loglik(dte["Choice"].values, np.full(len(dte), pbar))
        R2s.append(1.0 - ll_model/ll_null)
    return float(np.mean(R2s))

def forward_select(df_all, fit_fun, pred_fun, pt, max_steps=5, tol=1e-4):
    """Simple forward stepwise selection: try adding one new term each step; keep it if pseudo-R² improves by more than tol."""
    selected = []
    best = mcfadden_cv(df_all, selected, fit_fun, pred_fun, pt, n_splits=N_SPLITS)
    trace = [("NONE", best, [])]
    steps = 0
    while steps < max_steps:
        steps += 1
        best_gain = 0.0; best_term = None; best_R2 = best
        for t in CANDIDATES:
            if t in selected: continue
            R2_t = mcfadden_cv(df_all, selected+[t], fit_fun, pred_fun, pt, n_splits=N_SPLITS)
            gain = R2_t - best
            if gain > best_gain + tol:
                best_gain = gain; best_term = t; best_R2 = R2_t
        if best_term is None:
            break
        selected.append(best_term); best = best_R2
        trace.append((best_term, best, selected.copy()))
    return selected, best, trace

# 1) Baseline PT pseudo-R^2
R2_base = baseline_pseudoR2(df, pt, n_splits=N_SPLITS)
print(f"\nBaseline PT pseudo-R^2: {R2_base:.4f}")

# 2) Forward selection with additive ΔV (consistent with fit_additive / predict_additive defined above)
sel_A, R2_A, trace_A = forward_select(df, fit_additive, predict_additive, pt, max_steps=5)
print("\n(Additive) Forward selection result:")
print("  Selected terms:", sel_A if sel_A else "NONE")
print(f"  CV pseudo-R^2: {R2_A:.4f}")
for name, r2, cum in trace_A:
    print(f"    add {name:>7}: R^2={r2:.4f} | terms={cum}")

# 3) Fit on all data and print the final utility equation (still additive)
dV0_all, dV_new_all, eq_str = final_fit_and_equation(df, "A", sel_A, pt)
print("\n=== FINAL UTILITY EQUATION (Additive on dV) ===")
print(eq_str)

# 4) Preview old vs new ΔV
preview = pd.DataFrame({
    "user_id": df["user_id"].values[:8],
    "Trial":   df["Trial"].values[:8],
    "block":   df["block"].values[:8],
    "dV_PT":   dV_PT[:8],
    "dV_new":  dV_new_all[:8],
    "Choice":  df["Choice"].values[:8],
})
print("\nPreview (old vs new dV) — first 8 rows:")
print(preview.to_string(index=False))
# ================== End of final block ==================
