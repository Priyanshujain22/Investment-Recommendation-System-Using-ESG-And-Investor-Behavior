import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

SAVE_PATH = r'C:\Users\mp2hl\Documents\Investment-Recommendation-System-Using-ESG-And-Investor-Behavior\Unsupervised_GMM'

# Load both versions of the data
X_scaled = pd.read_csv(os.path.join(SAVE_PATH, 'X_scaled_gmm.csv'))
X_unscaled = pd.read_csv(os.path.join(SAVE_PATH, 'X_unscaled_gmm.csv'))

# --- BIC Analysis ---
components = [2,3,4,5,6,7,8,9,10]
bic_scores = []
for k in components:
    gmm_test = GaussianMixture(n_components=k, random_state=42)
    gmm_test.fit(X_scaled)
    bic_scores.append((k, gmm_test.bic(X_scaled)))

best_k_val = min(bic_scores, key=lambda x: x[1])[0]

# --- Final GMM Fit ---
gmm_final = GaussianMixture(n_components=best_k_val, covariance_type='full', random_state=42)
gmm_final.fit(X_scaled)
clusters = gmm_final.predict(X_scaled)
probabilities = gmm_final.predict_proba(X_scaled)

# --- Cluster Summary (Counts) ---
counts = pd.Series(clusters).value_counts().sort_index()
counts_df = pd.DataFrame({'Cluster': counts.index, 'Company_Count': counts.values})

# --- Cluster Profiles (RAW AVERAGES) ---
X_unscaled_with_clusters = X_unscaled.copy()
X_unscaled_with_clusters['cluster'] = clusters
cluster_profile_raw = X_unscaled_with_clusters.groupby('cluster').mean()

# --- Prepare Excel Outputs ---
prob_df = pd.DataFrame(
    probabilities, 
    columns=[f'Cluster_{i}_Prob' for i in range(best_k_val)]
)

with pd.ExcelWriter(os.path.join(SAVE_PATH, 'GMM_Cluster_Details.xlsx')) as writer:
    counts_df.to_excel(writer, sheet_name='Cluster_Summary', index=False)
    prob_df.to_excel(writer, sheet_name='Soft_Assignments', index=False)
    cluster_profile_raw.T.to_excel(writer, sheet_name='Cluster_Profiles_Raw')

# Save Labeled Dataset for Script 3
X_labeled = X_scaled.copy()
X_labeled['gmm_cluster'] = clusters
X_labeled.to_csv(os.path.join(SAVE_PATH, 'GMM_Labeled_Data.csv'), index=False)

# --- Heatmap (Using RAW values as requested) ---
plt.figure(figsize=(12, 8))
# Note: We use annot=True to show the real numbers in the boxes
sns.heatmap(cluster_profile_raw, annot=True, fmt=".4f", cmap='YlGnBu')
plt.title("Cluster Feature Averages (Raw Values)")
plt.savefig(os.path.join(SAVE_PATH, 'cluster_heatmap.png'))
plt.close()

# --- PCA Plot ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title("GMM Clusters in 2D Space")
plt.colorbar(label='Cluster')
plt.savefig(os.path.join(SAVE_PATH, 'pca_clusters.png'))
plt.close()

print(f"GMM applied. Cluster counts and Raw Profiles saved.")