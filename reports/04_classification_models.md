# 04 — Classification Models: Wrap-Up Summary

## Datasets
- **finance_tracker** (`../data/profile/personal_finance_tracker_dataset.csv`)
  - Used for **classification** targets:
    - `savings_goal_met` (binary)
    - `financial_stress_level` (multiclass: Low/Medium/High)

## Features
- Numeric: `monthly_income`, `monthly_expense_total`, `savings_rate`, `budget_goal`, `credit_score`, `debt_to_income_ratio`, `loan_payment`, `investment_amount`, `subscription_services`, `emergency_fund`, `transaction_count`, `discretionary_spending`, `essential_spending`, `rent_or_mortgage`, `financial_advice_score`
- Categorical (one-hot): `financial_scenario`, `income_type`, `category`, `cash_flow_status`, `financial_stress_level` (as feature when target is not stress level)

## Models & Performance

### A) Binary: `savings_goal_met`
- **Logistic Regression (baseline)**  
  - Confusion Matrix (test): near-perfect (only ~2 errors)  
  - Metrics (test): Accuracy ≈ **0.997**, ROC-AUC ≈ **1.000**
- **RandomForest (comparison)**  
  - Accuracy ≈ **0.943**; high precision but lower recall for the positive class  
  - Used to extract **feature importances**

**Saved artifacts**
- `../models/savings_goal_met_clf.pkl`
- `../models/savings_goal_met_columns.pkl`
- `../models/savings_goal_met_feature_importances.csv`

### B) Multiclass: `financial_stress_level` (Low/Medium/High)
- **Logistic Regression (with numeric scaling via ColumnTransformer)**  
  - Confusion Matrix: perfect separation across all 3 classes  
  - Metrics (test): Accuracy = **1.000**
- (Optional) Tree model comparison omitted since LR was perfect

**Saved artifacts**
- `../models/financial_stress_level_clf.pkl`
- `../models/financial_stress_level_columns.pkl`

## Notes
- Scaling numeric features resolved LR convergence warning for the multiclass task.
- RandomForest provided interpretability via feature importances, though LR performed best overall.

## How to Use (Inference)
1. Load model and columns:
   - Binary: `joblib.load("../models/savings_goal_met_clf.pkl")`
   - Multiclass: `joblib.load("../models/financial_stress_level_clf.pkl")`
2. Prepare an input row as a DataFrame, `pd.get_dummies`, then `reindex(columns=saved_columns, fill_value=0)`.
3. Call `.predict(...)` (and `.predict_proba(...)` if probabilities are needed).

## Next Steps
- (Optional) Calibrated probabilities for decision thresholds (e.g., CalibratedClassifierCV).
- Integrate classifiers into the **advice engine** to tailor recommendations by goal attainment & stress level.
- Proceed to **05_clustering.ipynb** to refresh user segmentation with the current data/features.
