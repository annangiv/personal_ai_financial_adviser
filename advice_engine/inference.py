import numpy as np
import pandas as pd
from .features import one_hot_align
from .config import PERSONA_NAMES, CLUSTER_FEATURES

def predict_savings_xgb(df_raw: pd.DataFrame, *, xgb_reg, xgb_features):
    X = one_hot_align(df_raw, xgb_features)
    yhat = xgb_reg.predict(X)
    out = df_raw.copy()
    out["Pred_Savings_XGB"] = np.maximum(yhat, 0).round(2)
    return out

def assign_cluster_and_persona(df_in, *, scaler, kmeans):
    X = df_in[CLUSTER_FEATURES].astype(float)
    labels = kmeans.predict(scaler.transform(X)).astype(int)

    out = df_in.copy()
    out["Cluster"] = labels

    # --- stable persona mapping from centroids ---
    ps_idx = CLUSTER_FEATURES.index("Pred_Savings_XGB")
    centers = kmeans.cluster_centers_  # in scaled space, but OK for ordering
    order = np.argsort(centers[:, ps_idx])  # low→mid→high
    rank_map = {int(cluster_id): int(rank) for rank, cluster_id in enumerate(order)}
    out["Persona"] = out["Cluster"].map(rank_map).map(PERSONA_NAMES)

    return out

def classify_goal(df_in: pd.DataFrame, *, goal_clf=None, goal_features=None, threshold=0.5):
    if goal_clf is None or goal_features is None:
        return df_in.copy()

    Xg = one_hot_align(df_in, goal_features)

    # coverage = fraction of non-zero features; tweak cutoff as you like
    coverage = (Xg.astype(bool).sum(axis=1) / Xg.shape[1]).to_numpy()
    if float(coverage.mean()) < 0.6:
        # Not enough info → abstain
        return df_in.copy()

    if hasattr(goal_clf, "predict_proba"):
        probs = goal_clf.predict_proba(Xg)[:, 1]
    elif hasattr(goal_clf, "decision_function"):
        scores = goal_clf.decision_function(Xg)
        mn, mx = float(scores.min()), float(scores.max())
        probs = (scores - mn) / (mx - mn + 1e-9)
    else:
        probs = goal_clf.predict(Xg).astype(float)

    out = df_in.copy()
    out["Goal_Prob"] = np.round(probs, 3)
    out["Goal_Label"] = np.where(out["Goal_Prob"] >= threshold, "Likely", "Unlikely")
    return out

