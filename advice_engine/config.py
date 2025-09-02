from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

XGB_PATH      = MODEL_DIR / "savings_xgb_reg.pkl"
XGB_COLS_PATH = MODEL_DIR / "savings_xgb_reg_columns.pkl"

SCALER_PATH   = MODEL_DIR / "scaler_05B.pkl"
KMEANS_PATH   = MODEL_DIR / "kmeans_05B.pkl"

GOAL_PATH     = MODEL_DIR / "savings_goal_xgb.pkl"
GOAL_COLS     = MODEL_DIR / "savings_goal_xgb_columns.pkl"

CLUSTER_FEATURES = [
    "Income", "Disposable_Income", "Pred_Savings_XGB",
    "Groceries", "Transport", "Entertainment",
]

PERSONA_NAMES = {
    0: "Budget-conscious majority",
    1: "Comfortable middle",
    2: "Affluent elite",
}
