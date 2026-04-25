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
    'risk_score', 'volatility_norm', 'totalEsg_norm', 'GICS Sector'
]

X = df.drop(columns=cols_to_drop + ['risk_class'], errors='ignore')

# --- 5.3: ONE-HOT ENCODING (Categorical to Numerical) ---
#X = pd.get_dummies(X, columns=['GICS Sector'])

# --- 5.4: FEATURE SCALING (Z-Score) ---
scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)

#print("✅ Metadata pruned and ML translation complete.")
#print(f"Final Feature Set (X) Shape: {X_scaled.shape}")

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pandas as pd

# 1. Initialize the Models
# We fix n_clusters=3 for most to match your 'Low, Medium, High' goal
n_clusters = 3

models = {
    "K-Means": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
    "GMM": GaussianMixture(n_components=n_clusters, random_state=42),
    "Hierarchical": AgglomerativeClustering(n_clusters=n_clusters),
    "Spectral": SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors'),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5) # Note: DBSCAN finds its own number of clusters
}

results = []

print("--- Running Clustering Tournament ---")

for name, model in models.items():
    # Fit and Predict
    if name == "GMM":
        model.fit(X_scaled)
        labels = model.predict(X_scaled)
    else:
        labels = model.fit_predict(X_scaled)
    
    # Calculate Silhouette Score (ignoring noise -1 for DBSCAN)
    mask = labels != -1
    if len(np.unique(labels[mask])) > 1:
        s_score = silhouette_score(X_scaled[mask], labels[mask])
    else:
        s_score = np.nan
        
    # Add to main dataframe temporarily to calculate risk alignment
    df[f'labels_{name}'] = labels
    
    # Calculate how distinct the risk scores are between clusters
    # We use the standard deviation of the means; higher means better risk separation
    cluster_means = df.groupby(f'labels_{name}')['risk_score'].mean()
    risk_sep = cluster_means.std()

    results.append({
        "Model": name,
        "Clusters Found": len(np.unique(labels[mask])),
        "Silhouette Score": round(s_score, 3),
        "Risk Separation": round(risk_sep, 3)
    })

# 2. Display the Leaderboard
leaderboard = pd.DataFrame(results).sort_values(by="Risk Separation", ascending=False)
print("\n🏆 Clustering Leaderboard:")
print(leaderboard)

# 3. Final Decision Logic
best_model_name = leaderboard.iloc[0]['Model']
print(f"\nRecommended Model: {best_model_name}")