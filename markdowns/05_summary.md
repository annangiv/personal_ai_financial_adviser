# 05 — Clustering & Segmentation

## Objective
Segment individuals based on income, savings, and spending behaviors to identify meaningful financial personas.

## Method
- Selected 6 numeric features:
  - Income, Disposable_Income, Desired_Savings, Groceries, Transport, Entertainment
- Standardized all features (mean=0, std=1).
- Explored cluster counts with **Elbow** and **Silhouette** methods.
- Chose **KMeans with k=3** (balance of interpretability and separation).
- Visualized clusters with PCA (2D scatter).
- Profiled clusters via mean values and bar charts.

## Results
- **Silhouette score (k=3):** 0.587 (good separation).
- **Cluster sizes:** 
  - Cluster 0 = 14,571 (73%)  
  - Cluster 1 = 718 (4%)  
  - Cluster 2 = 4,711 (23%)

### Cluster Profiles
- **Cluster 0 — Budget-conscious majority**  
  Income ~24k, modest savings (~2k), modest spending.  
- **Cluster 2 — Comfortable middle/upper-middle**  
  Income ~74k, savings ~9.6k, higher discretionary spend.  
- **Cluster 1 — Affluent elite**  
  Income ~183k, savings ~35k, very high lifestyle spend.  

## Outputs
- `05_clustered_data.csv` → full dataset with cluster labels (`cluster_k3`).  
- `05_cluster_profiles.csv` → summary of cluster means.  
- `05_clusters_pca.png` → PCA scatterplot of clusters.  
- `05_cluster_profiles.png` → bar chart of average feature values per cluster.  

---
**Conclusion:** clear 3-segment structure:  
- Budget-conscious majority  
- Comfortable middle  
- Affluent elite  