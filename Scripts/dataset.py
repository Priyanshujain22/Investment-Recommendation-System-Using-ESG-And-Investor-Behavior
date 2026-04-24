print("T1a")
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Load datasets
esg = pd.read_csv("sp500_esg_data.csv")
price = pd.read_csv("sp500_price_data.csv")

# Standardize Date column
price['Date'] = pd.to_datetime(price['Date'], utc=True)
price = price.sort_values('Date').reset_index(drop=True)

#print(f"✅ Data Ingested. ESG: {esg.shape}, Price: {price.shape}")
# Set Date as index for math operations
price.set_index('Date', inplace=True)

# Calculate daily returns for all columns (companies)
daily_returns = price.pct_change()

# Extract Market Features
market_features = pd.DataFrame({
    'Symbol'      : price.columns,
    'avg_return'  : daily_returns.mean().values,
    'volatility'  : daily_returns.std().values,
    'momentum_6m' : ((price.iloc[-1] - price.iloc[0]) / price.iloc[0]).values
})

#print("✅ Market signals extracted (Volatility, Momentum, Avg Return).")
# Merge on Ticker Symbol
df = pd.merge(esg, market_features, on='Symbol', how='inner')

#print(f"✅ Merge Complete. Unified Dataset Shape: {df.shape}")
scaler_mm = MinMaxScaler()

# Normalize Volatility and ESG for fair comparison
df[['volatility_norm', 'totalEsg_norm']] = scaler_mm.fit_transform(
    df[['volatility', 'totalEsg']]
)

# Apply weighted risk formula
df['risk_score'] = (
    0.6 * df['volatility_norm'] + 
    0.4 * (1 - df['totalEsg_norm'])
)

# Assign Risk Classes
df['risk_class'] = pd.cut(
    df['risk_score'],
    bins=[0.0, 0.30, 0.50, 1.0],
    labels=['Low', 'Medium', 'High']
)

#print("✅ Risk scoring and labeling finalized.")
print(df['risk_class'].value_counts())
# --- 5.1: PRUNING & LEAKAGE PREVENTION ---
# We remove:
# 1. Identifiers (Full Name, etc.)
# 2. Components of the Formula (Volatility, totalEsg) -> To prevent "Cheating"
# 3. Intermediate Math (risk_score, norms)
cols_to_drop = [
    'Full Name', 'GICS Sub-Industry', 'percentile', 'ratingYear', 
    'ratingMonth', 'overallRisk', 'Symbol', 'totalEsg', 'volatility', 
    'risk_score', 'volatility_norm', 'totalEsg_norm', 'GICS Sector'
]

X = df.drop(columns=cols_to_drop + ['risk_class'], errors='ignore')

# --- 5.2: TARGET ENCODING ---
le = LabelEncoder()
y = le.fit_transform(df['risk_class'])

# --- 5.3: ONE-HOT ENCODING (Categorical to Numerical) ---
#X = pd.get_dummies(X, columns=['GICS Sector'])

# --- 5.4: FEATURE SCALING (Z-Score) ---
scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)

#print("✅ Metadata pruned and ML translation complete.")
#print(f"Final Feature Set (X) Shape: {X_scaled.shape}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA # For visualizing high-dimensional data

# ... [KEEP STEPS 1-5 FROM YOUR ORIGINAL CODE EXACTLY THE SAME] ...
# (Data Loading, Market Features, Merging, Scoring, and Scaling)

# ============================================================
# STEP 7: Gaussian Mixture Model (Unsupervised)
# ============================================================

# We initialize GMM to look for 3 clusters (to match Low, Med, High)
# covariance_type='full' allows clusters to be any elliptical shape (very flexible)
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)

# Fit the model and predict clusters
# Note: We are NOT using 'y' here. The model doesn't see your labels!
gmm_clusters = gmm.fit_predict(X_scaled)

# Get the "Soft" probabilities (how sure is GMM about each company?)
gmm_probs = gmm.predict_proba(X_scaled)

# Add results back to our main dataframe for analysis
df['gmm_cluster'] = gmm_clusters
df['cluster_confidence'] = gmm_probs.max(axis=1)

print("✅ GMM Clustering Complete.")

# ============================================================
# STEP 8: Mapping Clusters to Risk (Comparison)
# ============================================================

# Since GMM labels are random (0, 1, 2), we find which cluster 
# has the highest average 'risk_score' to name them logically.
cluster_map = df.groupby('gmm_cluster')['risk_score'].mean().sort_values().index
mapping = {cluster_map[0]: 'Cluster_Low', cluster_map[1]: 'Cluster_Medium', cluster_map[2]: 'Cluster_High'}
df['gmm_label'] = df['gmm_cluster'].map(mapping)

print("\n📊 Cluster Mapping (Based on your Risk Score):")
for cluster, label in mapping.items():
    avg_score = df[df['gmm_cluster'] == cluster]['risk_score'].mean()
    print(f"   {label}: Center Score = {avg_score:.3f}")

# ============================================================
# STEP 9: The "Overlap" Audit (Crosstab)
# ============================================================
# This table shows how many 'Medium' companies the GMM thinks 
# are actually 'Low' or 'High'.
comparison = pd.crosstab(df['risk_class'], df['gmm_label'])
print("\n" + "="*50)
print("COMPARISON: FORMULA LABELS VS. GMM CLUSTERS")
print("="*50)
print(comparison)

# ============================================================
# STEP 10: Visualizing the Natural Clusters (PCA)
# ============================================================
# Since we have many features, we use PCA to squash them into 2D for a plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1], 
    hue=df['gmm_label'], 
    size=df['cluster_confidence'],
    sizes=(20, 200),
    palette='viridis', 
    alpha=0.7
)
plt.title('GMM Natural Risk Clusters (PCA Projection)', fontsize=14, fontweight='bold')
plt.xlabel('Principal Component 1 (General Market Factors)')
plt.ylabel('Principal Component 2 (ESG/Social Factors)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ============================================================
# STEP 11: Identify the "Confused" Companies
# ============================================================
# Find companies where GMM confidence is low (< 60%)
low_conf = df[df['cluster_confidence'] < 0.60]
print(f"\n⚠️ Identified {len(low_conf)} companies sitting in the Overlap (Low Confidence).")
print(low_conf[['Symbol', 'risk_class', 'gmm_label', 'cluster_confidence']].head(10))

# Create a temporary dataframe to compare the 126 companies
# These are companies you called 'Medium' but GMM called 'High'
audit_df = df[(df['risk_class'] == 'Medium') & (df['gmm_label'] == 'Cluster_High')]

# Companies that 'Stayed' in Medium
stayed_df = df[(df['risk_class'] == 'Medium') & (df['gmm_label'] == 'Cluster_Medium')]

print("🔎 AUDIT REPORT: Why did 126 Mediums move to High?")
print("-" * 50)

# Compare the averages of key features
features_to_check = ['beta', 'socialScore', 'environmentScore', 'marketCap', 'momentum_6m']

comparison_stats = pd.DataFrame({
    'Stayed_Medium': stayed_df[features_to_check].mean(),
    'Moved_to_High': audit_df[features_to_check].mean()
})

comparison_stats['Difference_%'] = ((comparison_stats['Moved_to_High'] - comparison_stats['Stayed_Medium']) / comparison_stats['Stayed_Medium']) * 100
print(comparison_stats)

# Check if a specific sector dominated the move
print("\n🏢 Sectors that moved the most to High:")
print(audit_df['GICS Sector'].value_counts().head(5))

# ============================================================
# STEP 12: Preparing the New Target (GMM Labels)
# ============================================================

# We use the labels created by the GMM in the previous step
le_gmm = LabelEncoder()
y_gmm = le_gmm.fit_transform(df['gmm_label'])

# Split the data using the NEW labels
# We still use stratify to keep the cluster proportions balanced
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_scaled, y_gmm, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_gmm
)

print(f"✅ Data prepared for GMM Validation. Training on {X_train_g.shape[0]} samples.")
print(f"Original Training Balance: {pd.Series(y_train_g).value_counts().to_dict()}")

# 7.2: Apply SMOTE (Only to Training Data)
# This creates synthetic examples for Low and High classes to match Medium.
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_g, y_train_g)

print(f"Balanced Training Balance (SMOTE): {pd.Series(y_train_res).value_counts().to_dict()}")
print(f"Testing set remains untouched: ({X_test_g.shape[0]} samples)")

# ============================================================
# STEP 13: The Supervised Battle Royale (GMM Edition)
# ============================================================

models_gmm = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'SVM': SVC(kernel='rbf', class_weight='balanced', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='mlogloss')
}

results_gmm = []

for name, model in models_gmm.items():
    # Train on GMM labels
    model.fit(X_train_res, y_train_res)
    
    # Predict
    y_pred = model.predict(X_test_g)
    
    # Calculate Metrics
    acc = accuracy_score(y_test_g, y_pred)
    prec = precision_score(y_test_g, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test_g, y_pred, average='macro')
    f1 = f1_score(y_test_g, y_pred, average='macro')
    
    
    results_gmm.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4)
    })

# Convert to DataFrame and sort
results_gmm_df = pd.DataFrame(results_gmm).sort_values(by='F1 Score', ascending=False)

print("\n" + "="*60)
print("RESULTS: PREDICTING GMM NATURAL CLUSTERS")
print("="*60)
print(results_gmm_df.to_string(index=False))

# ============================================================
# STEP 14: Final Winner Confusion Matrix
# ============================================================
best_gmm_model_name = results_gmm_df.iloc[0]['Model']
best_gmm_model = models_gmm[best_gmm_model_name]

y_pred_final = best_gmm_model.predict(X_test_g)
cm_final = confusion_matrix(y_test_g, y_pred_final)

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=le_gmm.classes_)
disp.plot(cmap='Greens', values_format='d') # Using Green to distinguish from the first run
plt.title(f'Final Validation: {best_gmm_model_name} on GMM Labels', fontsize=13, fontweight='bold')
plt.grid(False)
plt.show()
# 1. Get the Cluster Counts
print("📊 FINAL CLUSTER DISTRIBUTION (GMM):")
print(df['gmm_label'].value_counts())
print("-" * 30)

# 2. Identify the "Jumpers"
# We look for companies where your Manual Label != Machine Label
jumpers = df[df['risk_class'] != df['gmm_label']].copy()

# 3. Calculate the "Why" for each company
# We compare each jumper's features to the average of their ORIGINAL risk_class
# to see what made them "stand out" so much that the GMM moved them.

# Calculate means for the original manual classes
manual_means = df.groupby('risk_class', observed=False)[features_to_check].mean()

def identify_jump_reason(row):
    original_class = row['risk_class']
    # Get the average values for that original class
    avg_vals = manual_means.loc[original_class]
    
    # Calculate how many standard deviations the company is away from its original class average
    # (This is a simplified 'Z-score' to find the biggest outlier factor)
    differences = {}
    for feat in features_to_check:
        diff = abs(row[feat] - avg_vals[feat]) / (df[feat].std() + 1e-6)
        differences[feat] = diff
    
    # The feature with the biggest difference is the "Trigger"
    trigger_feature = max(differences, key=differences.get)
    return trigger_feature

# Apply the logic
jumpers['jump_trigger'] = jumpers.apply(identify_jump_reason, axis=1)

# 4. Display the results for the first 20 jumpers
#print(f"🚀 Identified {len(jumpers)} companies that 'Jumped' to a new risk category.")
#print("\nSample Audit Table (First 20 Jumpers):")
#print(jumpers[['Symbol', 'risk_class', 'gmm_label', 'jump_trigger']].head(20))
# Get the 'Means' (Centers) of each cluster
# This is essentially the 'Formula' the GMM used
cluster_centers = pd.DataFrame(
    gmm.means_, 
    columns=X.columns, 
    index=['Cluster 0', 'Cluster 1', 'Cluster 2']
)

print("🧬 THE CLUSTER DNA (The machine's internal logic):")
print("-" * 50)
print(cluster_centers.T) # Transposed for easier reading

# Get the 'Weights' (How big is each bell curve?)
for i, weight in enumerate(gmm.weights_):
    print(f"\nCluster {i} Weight: {weight:.2%}")