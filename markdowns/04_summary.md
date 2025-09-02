04 — Classification Summary
Task: Predict whether a user meets their savings goal (savings_goal_met).
Data: personal_finance_tracker_dataset.csv, 15 numeric features (income, expenses, credit score, etc.).
Models compared: Decision Tree vs XGBoost.
Results:
Decision Tree — Accuracy ≈ 0.94; weaker on the minority “Met” class (Recall ≈ 0.64).
XGBoost — Accuracy ≈ 0.99; strong minority performance (Precision = 1.00, Recall ≈ 0.89); higher ROC–AUC.
Takeaway: XGBoost is the preferred classifier for production; Decision Tree is simpler and interpretable.
Artifacts saved:
../models/savings_goal_dt.pkl + _columns.pkl
../models/savings_goal_xgb.pkl + _columns.pkl
Next: optional threshold tuning for business trade-offs; proceed to 05 (KMeans segmentation).