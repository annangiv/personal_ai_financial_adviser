import joblib
from .config import XGB_PATH, XGB_COLS_PATH, SCALER_PATH, KMEANS_PATH, GOAL_PATH, GOAL_COLS

def load_artifacts():
    xgb_reg = joblib.load(XGB_PATH)
    xgb_features = joblib.load(XGB_COLS_PATH)
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(KMEANS_PATH)

    try:
        goal_clf = joblib.load(GOAL_PATH)
        goal_features = joblib.load(GOAL_COLS)
    except Exception:
        goal_clf, goal_features = None, None

    return dict(
        xgb_reg=xgb_reg, xgb_features=xgb_features,
        scaler=scaler, kmeans=kmeans,
        goal_clf=goal_clf, goal_features=goal_features,
    )
