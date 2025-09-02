import pandas as pd
from .artifacts import load_artifacts
from .inference import predict_savings_xgb, assign_cluster_and_persona, classify_goal

# load once at import time (simple)
ART = load_artifacts()

def run_advice_engine(df_input: pd.DataFrame) -> pd.DataFrame:
    df1 = predict_savings_xgb(df_input, xgb_reg=ART["xgb_reg"], xgb_features=ART["xgb_features"])
    df2 = assign_cluster_and_persona(df1, scaler=ART["scaler"], kmeans=ART["kmeans"])
    df3 = classify_goal(df2, goal_clf=ART["goal_clf"], goal_features=ART["goal_features"])
    return df3
