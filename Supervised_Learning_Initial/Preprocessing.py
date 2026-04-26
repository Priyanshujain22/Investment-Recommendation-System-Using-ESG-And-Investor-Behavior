# ============================================================
# SCRIPT 1: Data Preprocessing & Risk Categorization
# ============================================================

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Get the directory where this script is actually located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create full paths to your CSV files
esg_path = os.path.join(BASE_DIR, "sp500_esg_data.csv")
price_path = os.path.join(BASE_DIR, "sp500_price_data.csv")

# Now load them using the full paths
# 2. Load ESG dataset
esg = pd.read_csv(esg_path)
# 3. Load stock price dataset
price = pd.read_csv(price_path)

# Set the save path based on your folder structure
SAVE_PATH = r'C:\Users\mp2hl\Documents\Investment-Recommendation-System-Using-ESG-And-Investor-Behavior\Supervised_Learning_Initial'

# Ensure the folder exists
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Fix date column
price['Date'] = pd.to_datetime(price['Date'], utc=True)
price = price.sort_values('Date').reset_index(drop=True)

# ============================================================
# STEP 2: Clean ESG Dataset
# ============================================================

esg.drop(columns=[
    'Full Name', 'GICS Sub-Industry', 'percentile', 
    'ratingYear', 'ratingMonth', 'overallRisk'
], inplace=True, errors='ignore')

# ============================================================
# STEP 3: Feature Engineering from Price Data
# ============================================================

price.set_index('Date', inplace=True)
daily_returns = price.pct_change()

market_features = pd.DataFrame({
    'Symbol'      : price.columns,
    'avg_return'  : daily_returns.mean().values,
    'volatility'  : daily_returns.std().values,
    'momentum_6m' : ((price.iloc[-1] - price.iloc[0]) / price.iloc[0]).values
})

# ============================================================
# STEP 4: Merge ESG + Market Features
# ============================================================

df = pd.merge(esg, market_features, on='Symbol', how='inner')

# ============================================================
# STEP 5: Create Target Variable (risk_class)
# ============================================================

scaler_mm = MinMaxScaler()

df[['volatility_norm', 'totalEsg_norm']] = scaler_mm.fit_transform(
    df[['volatility', 'totalEsg']]
)

# Composite risk score logic
df['risk_score'] = (
    0.6 * df['volatility_norm'] +
    0.4 * (1 - df['totalEsg_norm'])
)

df['risk_class'] = pd.cut(
    df['risk_score'],
    bins=[0.0, 0.30, 0.50, 1.0],
    labels=['Low', 'Medium', 'High']
)

# Save the primary dataset (Final_Dataset) as CSV for the pipeline
FINAL_DATASET_PATH = os.path.join(SAVE_PATH, 'Final_Dataset.csv')
df.to_csv(FINAL_DATASET_PATH, index=False)
print(f"Final_Dataset saved to: {FINAL_DATASET_PATH}")

# ============================================================
# VISUALIZATION - Dataset Analysis
# ============================================================

sns.set_theme(style="whitegrid")

# 1. Risk Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['risk_score'], bins=20, kde=True, color='skyblue')
plt.axvline(0.30, color='red', linestyle='--', label='Low-Med Boundary')
plt.axvline(0.50, color='red', linestyle='--', label='Med-High Boundary')
plt.title('Distribution of Composite Risk Scores', fontsize=14, fontweight='bold')
plt.xlabel('Risk Score (0 to 1)')
plt.ylabel('Number of Companies')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'risk_distribution.png'))
plt.close()

# 2. ESG vs. Volatility Scatter
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df, x='totalEsg', y='volatility', style='risk_class',
    palette='viridis', hue='GICS Sector', s=100, alpha=0.7
)
plt.title('ESG Performance vs. Stock Volatility by Sector', fontsize=14, fontweight='bold')
plt.xlabel('Total ESG Score')
plt.ylabel('Daily Volatility')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'esg_vs_volatility_scatter.png'))
plt.close()

print(f"Graphs saved to: {SAVE_PATH}")

# ============================================================
# STEP 6: ML Preprocessing (Encoding & Scaling)
# ============================================================

X = df.drop(columns=[
    'Symbol', 'totalEsg', 'volatility', 'risk_score',
    'volatility_norm', 'totalEsg_norm', 'risk_class'
])
le = LabelEncoder()
y = le.fit_transform(df['risk_class'])

X = pd.get_dummies(X, columns=['GICS Sector'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save preprocessed data for model training
PREPROCESSED_DATA_PATH = os.path.join(SAVE_PATH, 'Preprocessed_Data.xlsx')
with pd.ExcelWriter(PREPROCESSED_DATA_PATH, engine='openpyxl') as writer:
    X.to_excel(writer, sheet_name='X_Unscaled', index=False)
    pd.DataFrame(X_scaled, columns=X.columns).to_excel(writer, sheet_name='X_Scaled', index=False)
    pd.DataFrame(y, columns=['target']).to_excel(writer, sheet_name='Y_Label', index=False)

print(f"Preprocessed data (Excel) saved to: {PREPROCESSED_DATA_PATH}")