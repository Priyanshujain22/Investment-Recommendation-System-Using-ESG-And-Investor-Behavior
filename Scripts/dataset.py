# ============================================================
# MODEL 1 - Company Risk Prediction (ESG + Market Data)
# ============================================================

# STEP 1: Import Libraries
import os
import pandas as pd
import numpy as np

# Set path relative to current script (because of VS code error using OS)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'Dataset')

# Load datasets
esg = pd.read_csv(os.path.join(DATASET_DIR, 'Model1_sp500_esg_data.csv'))
price = pd.read_csv(os.path.join(DATASET_DIR, 'model1_sp500_price_data.csv'))

# Fix date column
price['Date'] = pd.to_datetime(price['Date'], utc=True)
price=price.sort_values('Date').reset_index(drop=True)

# Quick check
print("ESG shape:", esg.shape)
print("Price shape:", price.shape)
print("\nESG columns:", esg.columns.tolist())
print("\nPrice (first 3 rows, first 4 cols):")
print(price.iloc[:3, :4])


# ============================================================
# STEP 2: Clean ESG Dataset
# ============================================================

esg.drop(columns=[
    'Full Name',
    'GICS Sub-Industry',
    'percentile',
    'ratingYear',
    'ratingMonth',
    'overallRisk'
], inplace=True, errors='ignore')
# errors='ignore' → won't crash if any column is already missing

print("ESG cleaned shape:", esg.shape)
print("ESG columns after cleaning:", esg.columns.tolist())
print("\nESG sample:")
print(esg.head(3))

# ============================================================
# STEP 3: Feature Engineering from Price Data
# ============================================================

# The price dataset has Date as a column right now
# We make Date the "index" (like row label) so we can do math easily
price.set_index('Date', inplace=True)

# ---------------------------------------------------------------
# DAILY RETURN = how much did the stock change each day?
# Formula: (today's price - yesterday's price) / yesterday's price
# pct_change() does this automatically for all 426 companies at once
# ---------------------------------------------------------------
daily_returns = price.pct_change()

# ---------------------------------------------------------------
# Now we compute 3 features for each company using daily_returns
#
# avg_return   → what is the average daily growth of this stock?
#                positive = generally growing, negative = declining
#
# volatility   → how much does the price jump around day to day?
#                high std = very jumpy = risky
#                low std  = stable = safe
#
# momentum_6m  → did the stock go UP or DOWN overall from start to end?
#                (last price - first price) / first price
#                positive = upward trend, negative = downward trend
# ---------------------------------------------------------------
market_features = pd.DataFrame({
    'Symbol'      : price.columns,                                              # company ticker names
    'avg_return'  : daily_returns.mean().values,                                # average daily return
    'volatility'  : daily_returns.std().values,                                 # how risky/unstable
    'momentum_6m' : ((price.iloc[-1] - price.iloc[0]) / price.iloc[0]).values  # overall price trend
})

# Quick checks
print("Market features shape:", market_features.shape)  # should be (426, 4)
print("\nSample:")
print(market_features.head(5))
print("\nAny missing values?")
print(market_features.isnull().sum())


# ============================================================
# STEP 4: Merge ESG + Market Features
# ============================================================

# Both datasets have a 'Symbol' column → use it to join them
# how='inner' means → only keep companies that exist in BOTH datasets
# (if a company is in ESG but not in price data, it gets dropped)
df = pd.merge(esg, market_features, on='Symbol', how='inner')

print("Merged dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSample:")
print(df.head(3))
print("\nAny missing values?")
print(df.isnull().sum())

# ============================================================
# STEP 5: Create Target Variable (risk_class)
# ============================================================
# ============================================================
# STEP 5: Create Target Variable (risk_class) using ESG + Volatility
# ============================================================

from sklearn.preprocessing import MinMaxScaler

# --- Normalize volatility and ESG ---
scaler_mm = MinMaxScaler()

df[['volatility_norm', 'totalEsg_norm']] = scaler_mm.fit_transform(
    df[['volatility', 'totalEsg']]
)

# --- Create composite risk score ---
# Higher volatility → higher risk
# Higher ESG → lower risk
df['risk_score'] = (
    0.7 * df['volatility_norm'] +
    0.3 * (1 - df['totalEsg_norm'])
)

# --- Convert into risk classes ---
df['risk_class'] = pd.qcut(
    df['risk_score'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# --- Debug prints ---
print("Risk class distribution:")
print(df['risk_class'].value_counts())

print("\nSample (risk_score vs risk_class):")
print(df[['Symbol', 'risk_score', 'risk_class']].head(10))


# ============================================================
# STEP 6: Preprocessing
# ============================================================

from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 6.1 Drop columns not needed for ML ---
# Symbol → just a name, not a feature
# volatility → we already used it to CREATE risk_class, 
#              keeping it as input would be cheating (data leakage!)
X = df.drop(columns=[
    'Symbol',
    'volatility',
    'risk_score',
    'volatility_norm',
    'totalEsg_norm',
    'risk_class'
])
le = LabelEncoder()
y = le.fit_transform(df['risk_class'])

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)
print("\nFeature columns:", X.columns.tolist())

# --- 6.2 Encode GICS Sector (text → numbers) ---
# One-Hot Encoding splits 'GICS Sector' into separate 0/1 columns
# e.g. 'Health Care' → Health Care=1, all others=0
X = pd.get_dummies(X, columns=['GICS Sector'])

print("\nAfter encoding shape:", X.shape)

# --- 6.3 Scale features (important for SVM and KNN) ---
# StandardScaler makes all numbers on same scale
# so big numbers (marketCap) don't dominate small ones (beta)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaling done ✅")
print("Sample scaled values (first row):")
print(X_scaled[0])

# ============================================================
# STEP 7: Train Test Split
# ============================================================

from sklearn.model_selection import train_test_split

# Split data into 80% training and 20% testing
indices = np.arange(len(df))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train shape:", X_train.shape)  # 80% of 426
print("X_test shape:", X_test.shape)    # 20% of 426
print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts())
print("\nTesting class distribution:")
print(pd.Series(y_test).value_counts())


# ============================================================
# STEP 8: Train All Models
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Define all models in a dictionary ---
# This way we can loop through all of them instead of repeating code
models = {
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
    'SVM'                 : SVC(random_state=42),
    'KNN'                 : KNeighborsClassifier(),
    'Decision Tree'       : DecisionTreeClassifier(random_state=42),
    'Random Forest'       : RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    ),
    'XGBoost' : XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
    )
}

# --- Train and evaluate each model ---
results = []  # we'll store each model's scores here

for model_name, model in models.items():

    # Train the model on training data
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate scores
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    f1        = f1_score(y_test, y_pred, average='weighted')

    # Save results
    results.append({
        'Model'     : model_name,
        'Accuracy'  : round(accuracy, 4),
        'Precision' : round(precision, 4),
        'Recall'    : round(recall, 4),
        'F1 Score'  : round(f1, 4)
    })

    print(f"✅ {model_name} done")

# --- Show comparison table ---
results_df = pd.DataFrame(results)
print("\n============================================================")
print("MODEL COMPARISON TABLE")
print("============================================================")
print(results_df.to_string(index=False))


# ============================================================
# STEP 9: Confusion Matrix (Best Model → Logistic Regression)
# ============================================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Get predictions from best model
best_model_name = results_df.sort_values(by='F1 Score', ascending=False).iloc[0]['Model']
best_model = models[best_model_name]

print("Best model is: ",best_model_name)
y_pred_best = best_model.predict(X_test)

# --- Create Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_best, labels = best_model.classes_)

# --- Display it nicely ---
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_model.classes_
)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')

ax.set_title('Confusion Matrix - Logistic Regression', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()

print("Confusion Matrix saved ✅")

# --- Print it as numbers too ---
print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
print(pd.DataFrame(
    cm,
    index=[f'Actual {c}' for c in le.classes_],
    columns=[f'Pred {c}' for c in le.classes_]
))


importances = models['Decision Tree'].feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop Important Features:")
print(feature_importance_df.head(10))


# ============================================================
# STEP 11: Save All Outputs to Excel
# ============================================================

# Path to save excel
EXCEL_SAVE_PATH = os.path.join(BASE_DIR, 'model1_results.xlsx')

# --- Prepare final dataset with predictions ---

# Get predictions for ALL data (not just test)
# This way we can see risk class for every company
X_all_scaled = scaler.transform(X)
df['predicted_risk'] = le.inverse_transform(best_model.predict(X_all_scaled))

# --- Prepare model comparison table ---
results_df = pd.DataFrame(results)

# --- Prepare confusion matrix as dataframe ---
cm_df = pd.DataFrame(
    cm,
    index  =[f'Actual {c}' for c in le.classes_],
    columns=[f'Pred {c}' for c in le.classes_]
)

# --- Write all sheets to one Excel file ---
with pd.ExcelWriter(EXCEL_SAVE_PATH, engine='openpyxl') as writer:

    # Sheet 1 → Full cleaned + merged dataset with risk class
    df.to_excel(writer, sheet_name='Full Dataset', index=False)

    # Sheet 2 → Market features computed from price data
    market_features.to_excel(writer, sheet_name='Market Features', index=False)

    # Sheet 3 → Model comparison table
    results_df.to_excel(writer, sheet_name='Model Comparison', index=False)

    # Sheet 4 → Confusion matrix of best model
    cm_df.to_excel(writer, sheet_name='Confusion Matrix')

    # Sheet 5 → Test predictions (actual vs predicted)
    test_results = pd.DataFrame({
        'Symbol'          : df.iloc[idx_test]['Symbol'].values,
        'Actual Risk'     : le.inverse_transform(y_test),
        'Predicted Risk'  : le.inverse_transform(y_pred_best),
        'Correct'         : ['✅' if a == p else '❌'
                             for a, p in zip(y_test, y_pred_best)]
    })
    test_results.to_excel(writer, sheet_name='Test Predictions', index=False)

    # Sheet 6 → Feature Importance (from Decision Tree)
    feature_importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)

print("✅ Excel saved →", EXCEL_SAVE_PATH)
print("\nSheets inside:")
print("  📄 Full Dataset      → all 426 companies with risk class")
print("  📄 Market Features   → avg_return, volatility, momentum_6m")
print("  📄 Model Comparison  → accuracy, F1 of all 4 models")
print("  📄 Confusion Matrix  → actual vs predicted breakdown")
print("  📄 Test Predictions  → company wise correct/wrong prediction")
print("  📄 Feature Importance → contribution of each feature to the model")