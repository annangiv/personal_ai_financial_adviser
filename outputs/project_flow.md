         ┌─────────────────────┐
         │ 01–04: Supervised   │
         │ Models (Linear,     │
         │ XGBoost)            │
         │                     │
         │ Learn patterns:     │
         │ e.g. what factors   │
         │ drive savings?      │
         └─────────┬───────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │ Engineered Features │
         │ (Disposable_Income, │
         │ Desired_Savings,    │
         │ etc.)               │
         └─────────┬───────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │ 05: Clustering      │
         │ (KMeans)            │
         │                     │
         │ Group people by     │
         │ behavior into 3     │
         │ segments            │
         └─────────┬───────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │ 06: Rule-based      │
         │ Advice Engine       │
         │                     │
         │ Combine clusters +  │
         │ savings gap +       │
         │ discretionary spend │
         │ → Personalized text │
         └─────────┬───────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │ 07: Interface / UI  │
         │ (Notebook widgets,  │
         │ Streamlit option)   │
         │ Explore, filter,    │
         │ visualize advice    │
         └─────────────────────┘
