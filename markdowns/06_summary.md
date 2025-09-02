# 06 — Financial Advice Engine

## Objective
Generate personalized, actionable financial advice for each individual based on income, savings goals, discretionary spending, and potential savings opportunities.

## Method
- Loaded clustered dataset from **05** (`05_clustered_data.csv`).
- Engineered key signals:
  - **savings_gap** = Desired_Savings – Disposable_Income  
  - **can_meet_goal** = True/False  
  - **discretionary_total** = sum of flexible expenses  
  - **top1/2/3 savings levers** = categories with largest potential savings
- Built rule-based advice engine:
  - Check if savings goals are achievable.  
  - Comment on discretionary spending level (low, balanced, elevated, very high).  
  - Highlight top 3 potential savings levers.  
  - Provide cluster-specific next-step guidance.

## Results
- Advice generated for all 20,000 individuals.
- Advice format:
  - Persona tag (cluster segment)  
  - Bullet-pointed personalized guidance with rupee formatting (₹)  
  - Specific next-step recommendation

### Example Output
