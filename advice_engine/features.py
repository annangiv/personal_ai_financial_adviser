import pandas as pd

def one_hot_align(df_in: pd.DataFrame, required_cols: list[str], cat_cols=("Occupation","City_Tier")) -> pd.DataFrame:
    use_cats = [c for c in cat_cols if c in df_in.columns]
    enc = pd.get_dummies(df_in.copy(), columns=use_cats, dtype=int)
    for c in required_cols:
        if c not in enc.columns:
            enc[c] = 0
    return enc[required_cols]
