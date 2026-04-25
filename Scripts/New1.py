print("T1a")
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

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

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# 1. Define your features
features_to_check = ['avg_return', 'volatility', 'momentum_6m', 'totalEsg', 
                     'marketCap', 'environmentScore', 'socialScore', 
                     'governanceScore', 'beta']

# 2. Calculate dynamic grid size
num_features = len([f for f in features_to_check if f in df.columns])
cols = 3  # We'll stick to 3 columns
rows = int(np.ceil(num_features / cols)) # Calculate rows needed

# 3. Create the figure
plt.figure(figsize=(15, 5 * rows)) # Adjust height based on number of rows

for i, col in enumerate(features_to_check):
    if col in df.columns:
        # Calculate Skewness
        skew_val = skew(df[col].dropna())
        
        # 4. Use the dynamic rows/cols for the subplot
        plt.subplot(rows, cols, i + 1)
        sns.histplot(df[col], kde=True, color='skyblue')
        plt.title(f"{col}\nSkewness: {skew_val:.2f}", fontsize=10)
        plt.xlabel("") # Clean up x-axis labels

plt.tight_layout()
#plt.show()

# 1. Critical Transformations (Extremely high skew)
df['marketCap'] = np.log1p(df['marketCap'])

# For momentum, we use a 'Signed Log' because momentum can be negative. 
# This preserves the direction (gain vs loss) while squishing the outliers.
df['momentum_6m'] = np.sign(df['momentum_6m']) * np.log1p(np.abs(df['momentum_6m']))

# 2. High Priority Transformations (Skew > 1.0)
df['volatility'] = np.log1p(df['volatility'])
df['governanceScore'] = np.log1p(df['governanceScore'])
df['environmentScore'] = np.log1p(df['environmentScore'])

# 3. Optional: Beta (Skew 0.81 is borderline)
# If you want to be extra precise, log it too. If not, it's okay to leave it.
# df['beta'] = np.log1p(df['beta']) 

#print("Log transformations applied to: marketCap, momentum_6m, volatility, governanceScore, environmentScore")

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

# --- 5.2: PRESERVE FOR EVALUATION (New) ---
# We save these before dropping them from the feature set X
risk_score_final = df['risk_score'].copy()
risk_class_final = df['risk_class'].copy()

# --- 5.1: PRUNING & LEAKAGE PREVENTION ---
# We remove:
# 1. Identifiers (Full Name, etc.)
# 2. Components of the Formula (Volatility, totalEsg) -> To prevent "Cheating"
# 3. Intermediate Math (risk_score, norms)
cols_to_drop = [
    'Full Name', 'GICS Sub-Industry', 'percentile', 'ratingYear', 
    'ratingMonth', 'overallRisk', 'Symbol', 'totalEsg', 'volatility', 
    'risk_score', 'volatility_norm', 'totalEsg_norm'
]

X = df.drop(columns=cols_to_drop + ['risk_class'], errors='ignore')

# --- 5.3: ONE-HOT ENCODING (Categorical to Numerical) ---
X = pd.get_dummies(X, columns=['GICS Sector'])

# --- 5.4: FEATURE SCALING (Z-Score) ---
scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)

#print("✅ Metadata pruned and ML translation complete.")
#print(f"Final Feature Set (X) Shape: {X_scaled.shape}")

from sklearn.mixture import GaussianMixture

# --- STEP 2 & 3: FIND THE OPTIMAL NUMBER OF CLUSTERS (BIC TEST) ---
# We use BIC (Bayesian Information Criterion) to find the "Sweet Spot". 
# Lower BIC = Better balance between model fit and simplicity.
components = [3]
bic_scores = []

for k in components:
    gmm_test = GaussianMixture(n_components=k, random_state=42)
    gmm_test.fit(X_scaled)
    bic_scores.append((k, gmm_test.bic(X_scaled)))

# Select K with the lowest BIC
print(components)
print(bic_scores)
best_k_val = min(bic_scores, key=lambda x: x[1])[0]
print(f"Optimal number of clusters according to BIC: {best_k_val}")

# --- STEP 4: FIT THE FINAL GMM MODEL ---
gmm_final = GaussianMixture(n_components=best_k_val, covariance_type='full', random_state=42)
gmm_final.fit(X_scaled)

# Clusters (Hard assignment: 0, 1, or 2)
clusters = gmm_final.predict(X_scaled)

# Probabilities (Soft assignment: e.g., [0.1, 0.8, 0.1])
probabilities = gmm_final.predict_proba(X_scaled)

# --- STEP 5: EVALUATE USING THE RISK SCORE ---
# Re-attach the results to the original dataframe for interpretation
df['gmm_cluster'] = clusters
df['risk_score_eval'] = risk_score_final

# Analyze the average risk within each cluster
cluster_analysis = df.groupby('gmm_cluster')['risk_score_eval'].mean().sort_values()
print("\nAverage Risk Score per Cluster:")
print(cluster_analysis)

# --- STEP 6: EXAMINE BOUNDARY CASES ---
# This shows how "certain" the model is about each company
prob_df = pd.DataFrame(
    probabilities, 
    columns=[f'Cluster_{i}_Prob' for i in range(best_k_val)],
    index=df.index
)
print("\nSample Probabilities (Soft Assignments):")
print(prob_df.head())
l1 = []
for i in range(0, 3):
    l1.append((prob_df[f"Cluster_{i}_Prob"] == 1.0).sum())
print(l1)
'''
# --- STEP 7: MAP CLUSTERS TO HUMAN LABELS ---
# We map the cluster numbers to 'Low', 'Medium', 'High' based on their average risk
risk_mapping = {
    cluster_analysis.index[0]: 'Low Risk',
    cluster_analysis.index[1]: 'Medium Risk' if best_k_val > 2 else 'High Risk',
}
if best_k_val >= 3:
    risk_mapping[cluster_analysis.index[-1]] = 'High Risk'

df['gmm_label'] = df['gmm_cluster'].map(risk_mapping)

print("\nFinal GMM Class Distribution:")
print(df['gmm_label'].value_counts())'''