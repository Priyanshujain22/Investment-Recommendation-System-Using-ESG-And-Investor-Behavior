import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# 1. Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
esg_path = os.path.join(BASE_DIR, "sp500_esg_data.csv")
price_path = os.path.join(BASE_DIR, "sp500_price_data.csv")

esg = pd.read_csv(esg_path)
price = pd.read_csv(price_path)

SAVE_PATH = r'C:\Users\mp2hl\Documents\Investment-Recommendation-System-Using-ESG-And-Investor-Behavior\Supervised_Weighted'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# 2. Feature Engineering
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

# 3. Log Transformations for Skewness
df['momentum_6m'] = np.sign(df['momentum_6m']) * np.log1p(np.abs(df['momentum_6m']))
df['volatility'] = np.log1p(df['volatility'])
df['governanceScore'] = np.log1p(df['governanceScore'])
df['environmentScore'] = np.log1p(df['environmentScore'])

# 4. Complex Weighted Risk Scoring
selected_features = [
    'volatility', 'beta', 'avg_return', 'momentum_6m',
    'environmentScore', 'socialScore', 'governanceScore', 'highestControversy'
]
scaler_mm = MinMaxScaler()
df_scaled = pd.DataFrame(scaler_mm.fit_transform(df[selected_features]), columns=selected_features)

# Inverting scores so 1.0 = High Risk
df_scaled['momentum_6m'] = np.abs(df_scaled['momentum_6m'])
df_scaled['avg_return'] = 1 - df_scaled['avg_return']
for col in ['environmentScore','socialScore','governanceScore']:
    df_scaled[col] = 1 - df_scaled[col]

market_risk = (0.3 * df_scaled['volatility'] + 0.3 * df_scaled['beta'] + 
               0.2 * df_scaled['momentum_6m'] + 0.2 * df_scaled['avg_return'])

esg_risk = (0.3 * df_scaled['environmentScore'] + 0.3 * df_scaled['socialScore'] + 
            0.3 * df_scaled['governanceScore'] + 0.1 * df_scaled['highestControversy'])

df['risk_score'] = 0.5 * market_risk + 0.5 * esg_risk

# Use qcut for 3 equal percentile bins (Balanced classes)
df['risk_class'] = pd.qcut(df['risk_score'], q=3, labels=['Low', 'Medium', 'High'])

# 5. ML Preprocessing
X = df.drop(columns=['Full Name', 'GICS Sub-Industry', 'percentile', 'ratingYear', 
                     'ratingMonth', 'overallRisk', 'Symbol', 'totalEsg', 'volatility',
                     'risk_score', 'GICS Sector', 'marketCap', 'risk_class'], errors='ignore')

# Save unscaled features and Y labels
X.to_csv(os.path.join(SAVE_PATH, 'X_unscaled_weighted.csv'), index=False)
pd.Series(df['risk_class']).to_csv(os.path.join(SAVE_PATH, 'y_labels_weighted.csv'), index=False)

# Standardize features for modeling
scaler_std = StandardScaler()
X_scaled = pd.DataFrame(scaler_std.fit_transform(X), columns=X.columns)
X_scaled.to_csv(os.path.join(SAVE_PATH, 'X_scaled_weighted.csv'), index=False)

print(f"Weighted preprocessing complete. Classes:\n{df['risk_class'].value_counts()}")