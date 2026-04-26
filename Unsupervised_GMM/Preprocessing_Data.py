import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

# 1. Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
esg_path = os.path.join(BASE_DIR, "sp500_esg_data.csv")
price_path = os.path.join(BASE_DIR, "sp500_price_data.csv")

esg = pd.read_csv(esg_path)
price = pd.read_csv(price_path)

SAVE_PATH = r'C:\Users\mp2hl\Documents\Investment-Recommendation-System-Using-ESG-And-Investor-Behavior\Unsupervised_GMM'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# 2. Date fixing and feature engineering
price['Date'] = pd.to_datetime(price['Date'], utc=True)
price = price.sort_values('Date').reset_index(drop=True)
price.set_index('Date', inplace=True)

daily_returns = price.pct_change()
market_features = pd.DataFrame({
    'Symbol'      : price.columns,
    'avg_return'  : daily_returns.mean().values,
    'volatility'  : daily_returns.std().values,
    'momentum_6m' : ((price.iloc[-1] - price.iloc[0]) / price.iloc[0]).values
})
df = pd.merge(esg, market_features, on='Symbol', how='inner')

# 3. Skewness Correction (Log Transformations)
df['marketCap'] = np.log1p(df['marketCap'])
df['momentum_6m'] = np.sign(df['momentum_6m']) * np.log1p(np.abs(df['momentum_6m']))
df['volatility'] = np.log1p(df['volatility'])
df['governanceScore'] = np.log1p(df['governanceScore'])
df['environmentScore'] = np.log1p(df['environmentScore'])

# 4. Prepare Features
cols_to_drop = [
    'Full Name', 'GICS Sub-Industry', 'percentile', 'ratingYear', 
    'ratingMonth', 'overallRisk', 'Symbol', 'GICS Sector',
    'risk_score', 'volatility_norm', 'totalEsg_norm'
]
X = df.drop(columns=cols_to_drop + ['risk_class'], errors='ignore')

# Save the unscaled (but log-transformed) data for readable profiles
X.to_csv(os.path.join(SAVE_PATH, 'X_unscaled_gmm.csv'), index=False)

# 5. Scaling
scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df.to_csv(os.path.join(SAVE_PATH, 'X_scaled_gmm.csv'), index=False)

print("Preprocessing complete. Scaled and Unscaled CSVs saved.")