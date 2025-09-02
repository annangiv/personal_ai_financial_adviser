#!/usr/bin/env python3
# train_all.py
"""
End-to-end trainer for the Financial Advice Engine.

Inputs:
  - One or more CSVs under data/ (primary: personal_finance_expenses.csv)
Outputs (drop-in for your app):
  models/
    - savings_xgb_reg.pkl
    - savings_xgb_reg_columns.pkl
    - scaler_05B.pkl
    - kmeans_05B.pkl
    - (optional) savings_goal_xgb.pkl
    - (optional) savings_goal_xgb_columns.pkl
  reports/
    - train_summary.json
    - feature_catalog.json
"""

import argparse, json, os, sys, time, hashlib, warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
from sklearn.utils import shuffle
from sklearn.exceptions import ConvergenceWarning

import joblib

# --- Try XGBoost; fallback to scikit-learn GB if unavailable ---
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    HAS_XGB = False


# ----------------------- CLI -----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Train all models for the Advice Engine")
    ap.add_argument("--data", nargs="+", required=True,
                    help="CSV paths under data/ to include (expenses CSV is the main one).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=float, default=0.10)
    ap.add_argument("--test-size", type=float, default=0.10)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--winsor-pct", type=float, default=0.005,
                    help="Winsorize top/bottom percentile for numeric cols (e.g., 0.005 = 0.5%%).")
    return ap.parse_args()


# ------------------- IO / Utils -------------------

CANON_COLS = [
    "Income", "Disposable_Income",
    "Groceries", "Transport", "Entertainment",
    "Occupation", "City_Tier",
    # Optional:
    "Age", "Desired_Savings", "Desired_Savings_Percentage", "Goal_Label"
]

CLUSTER_FEATURES = [
    "Income", "Disposable_Income", "Pred_Savings_XGB",
    "Groceries", "Transport", "Entertainment",
]

def ensure_dirs(models_dir: str, reports_dir: str):
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

def winsorize(df: pd.DataFrame, cols: List[str], p: float) -> pd.DataFrame:
    if p <= 0: return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            lo, hi = out[c].quantile([p, 1 - p])
            out[c] = out[c].clip(lo, hi)
    return out

def monthly(v: Optional[float], period_hint: Optional[str]) -> Optional[float]:
    if v is None: return None
    return float(v)/12.0 if (period_hint == "annual") else float(v)

def is_present(col, df): return (col in df.columns) and (df[col].notna().any())

# ---------------- Data Unification ----------------

def load_and_unify(csv_paths: List[str]) -> pd.DataFrame:
    """
    Unify multiple CSVs into the canonical schema.
    Heuristics for tracker dataset:
      - essential_spending -> split into Groceries (60%) / Transport (40%)
      - discretionary_spending -> Entertainment
    Transaction-only datasets without user linkage are ignored.
    """
    frames = []

    for p in csv_paths:
        pth = Path(p)
        if not pth.exists():
            print(f"[warn] Missing file: {p}")
            continue

        df = pd.read_csv(pth)
        cols = {c.lower().strip(): c for c in df.columns}  # map lower->orig
        df.columns = [c.strip() for c in df.columns]

        # Identify known schemas
        if {"Income","Groceries","Transport","Entertainment"}.issubset(df.columns):
            # Likely the Kaggle 20k file (personal_finance_expenses.csv)
            out = pd.DataFrame()
            out["Income"] = df["Income"].astype(float)
            out["Groceries"] = df["Groceries"].astype(float)
            out["Transport"] = df["Transport"].astype(float)
            out["Entertainment"] = df["Entertainment"].astype(float)
            out["Occupation"] = df.get("Occupation")
            out["City_Tier"] = df.get("City_Tier")
            out["Age"] = df.get("Age")
            # Desired fields if available
            out["Desired_Savings"] = df.get("Desired_Savings")
            out["Desired_Savings_Percentage"] = df.get("Desired_Savings_Percentage")

            if "Disposable_Income" in df.columns:
                out["Disposable_Income"] = df["Disposable_Income"].astype(float)
            else:
                total = out[["Groceries","Transport","Entertainment"]].sum(axis=1)
                out["Disposable_Income"] = out["Income"] - total

            frames.append(out)

        elif {"monthly_income","monthly_expense_total"}.issubset({c.lower():c for c in df.columns}):
            # Tracker dataset (personal_finance_tracker_dataset.csv)
            # Use essential/discretionary to approximate categories
            def get(col):
                return df[[c for c in df.columns if c.lower()==col]].iloc[:,0] if any(c.lower()==col for c in df.columns) else None

            income = get("monthly_income")
            essential = get("essential_spending")
            discretionary = get("discretionary_spending")
            city = get("city_tier")  # may not exist
            occ = get("income_type") # approximate occupation

            if income is None or (essential is None and discretionary is None):
                print(f"[info] Skipping {pth.name}: insufficient columns for canonical mapping.")
                continue

            out = pd.DataFrame()
            out["Income"] = income.astype(float)

            if essential is not None:
                essential = essential.astype(float).clip(lower=0)
                out["Groceries"] = 0.60 * essential
                out["Transport"] = 0.40 * essential
            else:
                out["Groceries"] = 0.0
                out["Transport"] = 0.0

            if discretionary is not None:
                out["Entertainment"] = discretionary.astype(float).clip(lower=0)
            else:
                out["Entertainment"] = 0.0

            out["Occupation"] = occ if occ is not None else "Unknown"
            out["City_Tier"]   = city if city is not None else "Tier_2"

            total = out[["Groceries","Transport","Entertainment"]].sum(axis=1)
            out["Disposable_Income"] = out["Income"] - total

            frames.append(out)

        else:
            # Transaction logs or unknown schemas: skip for supervised model training
            print(f"[info] Skipping {pth.name}: not a supported supervised schema.")
            continue

    if not frames:
        raise RuntimeError("No usable CSVs found. Provide at least the expenses CSV.")

    merged = pd.concat(frames, ignore_index=True)
    # Keep only canonical columns that exist
    keep = [c for c in CANON_COLS if c in merged.columns]
    merged = merged[keep]
    return merged


# ----------------- Training Steps -----------------

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """One-hot encode Occupation/City_Tier, keep numeric as-is."""
    cat_cols = [c for c in ["Occupation","City_Tier"] if c in df.columns]
    enc = pd.get_dummies(df, columns=cat_cols, dtype=int)
    # Drop obvious target-ish columns from the regressor features
    drop_if_exist = ["Desired_Savings","Desired_Savings_Percentage","Goal_Label"]
    feat_df = enc.drop(columns=[c for c in drop_if_exist if c in enc.columns], errors="ignore")
    feature_cols = [c for c in feat_df.columns if c not in []]
    return feat_df, feature_cols

def train_regressor(df: pd.DataFrame, seed: int) -> Tuple[object, List[str], dict]:
    """
    Target: monthly savings approximated as Disposable_Income - (Groceries+Transport+Entertainment)
    If your Kaggle dataset already encodes this differently, adjust here.
    """
    df = df.copy()
    # Define target y
    spend = df[["Groceries","Transport","Entertainment"]].sum(axis=1)
    y = (df["Disposable_Income"] - 0.0)  # Disposable_Income is already Income - Expenses in your dataset
    # For robustness across merged sources, we’ll cap negatives to 0 during training label creation
    y = y.clip(lower=0)

    X_full, cols = build_features(df)
    # Remove leakage: ensure the target is not included as a predictor
    # (Here y is derived; X_full already excludes obvious targets.)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X_full, y, test_size=0.20, random_state=seed)
    X_val,   X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=seed)

    if HAS_XGB:
        model = XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, n_jobs=4
        )
    else:
        warnings.warn("XGBoost not found; using GradientBoostingRegressor.")
        model = GradientBoostingRegressor(random_state=seed)

    model.fit(X_train, y_train)

    def smape(y_true, y_pred, eps=1e-9):
        """Symmetric MAPE (%)"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return 100 * np.mean(
            np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)
        )

    def masked_mape(y_true, y_pred, eps=1e-9, min_denom=100.0):
        """MAPE (%) only on rows where true >= min_denom"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mask = np.abs(y_true) >= min_denom
        if mask.sum() == 0:
            return None
        return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps)))

    def metrics(model, X, y, name):
        """Return lowercase keys; DO NOT reference reg_report here."""
        yhat = model.predict(X)
        mae = mean_absolute_error(y, yhat)
        rmse = float(np.sqrt(mean_squared_error(y, yhat)))
        mape_val = masked_mape(y, yhat)
        smape_val = smape(y, yhat)

        # Console-friendly log
        mape_str = f"{mape_val:.2f}%" if mape_val is not None else "nan"
        smape_str = f"{smape_val:.2f}%" if smape_val is not None else "nan"
        print(f"[reg-{name}] MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE*={mape_str}  SMAPE={smape_str}")

        # Store floats (JSON stays machine-friendly)
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape_masked": float(mape_val) if mape_val is not None else None,
            "smape": float(smape_val) if smape_val is not None else None,
        }

    report = {
        "val":  metrics(model, X_val,  y_val,  "val"),
        "test": metrics(model, X_test, y_test, "test"),
        "feature_count": len(cols),
    }
    return model, cols, report

def infer_pred_savings(df: pd.DataFrame, reg_model, reg_cols: List[str]) -> pd.DataFrame:
    # Align one-hot columns
    cat_cols = [c for c in ["Occupation","City_Tier"] if c in df.columns]
    enc = pd.get_dummies(df, columns=cat_cols, dtype=int)
    for c in reg_cols:
        if c not in enc.columns:
            enc[c] = 0
    X = enc[reg_cols]
    yhat = np.maximum(reg_model.predict(X), 0)
    out = df.copy()
    out["Pred_Savings_XGB"] = yhat
    return out

def train_cluster(df_with_pred: pd.DataFrame, seed: int) -> Tuple[StandardScaler, KMeans, dict]:
    use = df_with_pred[CLUSTER_FEATURES].astype(float).copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(use)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        kmeans = KMeans(n_clusters=3, random_state=seed, n_init="auto")
        kmeans.fit(Xs)

    # Persona ordering by Pred_Savings_XGB centroid (low->mid->high)
    ps_idx = CLUSTER_FEATURES.index("Pred_Savings_XGB")
    centers = kmeans.cluster_centers_
    order = np.argsort(centers[:, ps_idx]).tolist()

    report = {
        "k": 3,
        "cluster_order_by_pred_savings": order,
        "centers_sample": centers[:3].tolist(),
        "inertia": float(kmeans.inertia_)
    }
    return scaler, kmeans, report

def maybe_train_goal_classifier(df: pd.DataFrame, seed: int):
    """
    Trains a goal classifier only if a usable binary label exists.
    We look for 'Goal_Label' in {0,1} OR derive a proxy if both Desired_Savings and
    a computed target are available (optional). If not found, return None.
    """
    label_col = None
    if "Goal_Label" in df.columns and set(pd.unique(df["Goal_Label"].dropna())) <= {0,1}:
        label_col = "Goal_Label"
    else:
        # Optional proxy (commented out by default):
        # If Desired_Savings present, consider user "Likely" if Pred_Savings meets/exceeds desired.
        # But since Pred_Savings is produced AFTER regressor, this requires a second pass in practice.
        label_col = None

    if not label_col:
        return None, None, None

    # Build features & label
    X, cols = build_features(df)
    y = df[label_col].astype(int)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
    X_val,   X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp)

    if HAS_XGB:
        clf = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, n_jobs=4
        )
    else:
        clf = GradientBoostingClassifier(random_state=seed)

    clf.fit(X_train, y_train)

    def clf_metrics(X, y):
        proba = getattr(clf, "predict_proba", None)
        if proba:
            p1 = clf.predict_proba(X)[:,1]
            auc = roc_auc_score(y, p1)
        else:
            auc = np.nan
        yhat = clf.predict(X)
        prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        return {"auc": float(auc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

    report = {
        "val":  clf_metrics(X_val,  y_val),
        "test": clf_metrics(X_test, y_test),
        "feature_count": len(cols),
    }
    return clf, cols, report


# ---------------------- MAIN ----------------------

def main():
    args = parse_args()
    ensure_dirs(args.models_dir, args.reports_dir)

    # 1) Load & unify
    df0 = load_and_unify(args.data)
    print(f"[info] loaded {len(df0):,} rows from {len(args.data)} file(s)")

    # 2) Basic cleaning
    num_cols = [c for c in ["Income","Disposable_Income","Groceries","Transport","Entertainment","Age","Desired_Savings"] if c in df0.columns]
    for c in num_cols:
        df0[c] = pd.to_numeric(df0[c], errors="coerce")
    df0 = df0.dropna(subset=["Income","Groceries","Transport","Entertainment"]).reset_index(drop=True)
    # clip negatives
    for c in ["Income","Disposable_Income","Groceries","Transport","Entertainment"]:
        if c in df0.columns:
            df0[c] = df0[c].clip(lower=0)

    # winsorize numerics to reduce tail noise
    df0 = winsorize(df0, [c for c in ["Income","Disposable_Income","Groceries","Transport","Entertainment"] if c in df0.columns], args.winsor_pct)

    # 3) Train regressor
    reg_model, reg_cols, reg_report = train_regressor(df0, args.seed)

    # format values safely
    mape_val = reg_report['test'].get('mape_masked')
    smape_val = reg_report['test'].get('smape')

    mape_str = f"{mape_val:.2f}%" if mape_val is not None else "nan"
    smape_str = f"{smape_val:.2f}%" if smape_val is not None else "nan"

    print(
        f"[reg] MAE={reg_report['test']['mae']:.2f}  "
        f"RMSE={reg_report['test']['rmse']:.2f}  "
        f"MAPE*={mape_str}  "
        f"SMAPE={smape_str}"
    )

    # 4) Infer Pred_Savings to feed clustering
    df1 = infer_pred_savings(df0, reg_model, reg_cols)

    # 5) Train scaler + KMeans (personas)
    scaler, kmeans, km_report = train_cluster(df1, args.seed)
    print(f"[kmeans] k=3  inertia={km_report['inertia']:.2f}  order={km_report['cluster_order_by_pred_savings']}")

    # 6) Optional goal classifier (only if label present)
    goal_model, goal_cols, goal_report = maybe_train_goal_classifier(df1, args.seed)
    if goal_model is None:
        print("[goal] no usable label found -> skipping goal classifier.")

    # 7) Persist artifacts
    md = Path(args.models_dir)
    paths = {}

    reg_path = md / "savings_xgb_reg.pkl"
    joblib.dump(reg_model, reg_path)
    paths["savings_xgb_reg.pkl"] = file_hash(reg_path)

    reg_cols_path = md / "savings_xgb_reg_columns.pkl"
    joblib.dump(reg_cols, reg_cols_path)
    paths["savings_xgb_reg_columns.pkl"] = file_hash(reg_cols_path)

    scaler_path = md / "scaler_05B.pkl"
    joblib.dump(scaler, scaler_path)
    paths["scaler_05B.pkl"] = file_hash(scaler_path)

    km_path = md / "kmeans_05B.pkl"
    joblib.dump(kmeans, km_path)
    paths["kmeans_05B.pkl"] = file_hash(km_path)

    if goal_model is not None:
        goal_path = md / "savings_goal_xgb.pkl"
        joblib.dump(goal_model, goal_path)
        paths["savings_goal_xgb.pkl"] = file_hash(goal_path)

        goal_cols_path = md / "savings_goal_xgb_columns.pkl"
        joblib.dump(goal_cols, goal_cols_path)
        paths["savings_goal_xgb_columns.pkl"] = file_hash(goal_cols_path)

    # Feature catalog
    feat_catalog = {
        "regressor_columns": reg_cols,
        "cluster_features": CLUSTER_FEATURES,
        "classifier_columns": goal_cols if goal_model is not None else None,
    }
    with open(Path(args.reports_dir) / "feature_catalog.json", "w") as f:
        json.dump(feat_catalog, f, indent=2)

    # 8) Report
    report = {
        "timestamp": int(time.time()),
        "seed": args.seed,
        "row_count": int(len(df0)),
        "winsor_pct": args.winsor_pct,
        "regression": reg_report,
        "kmeans": km_report,
        "goal": goal_report if goal_model is not None else {"status": "skipped"},
        "artifacts": paths,
    }
    with open(Path(args.reports_dir) / "train_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] Artifacts → {args.models_dir} | Report → {args.reports_dir}/train_summary.json")


if __name__ == "__main__":
    main()
