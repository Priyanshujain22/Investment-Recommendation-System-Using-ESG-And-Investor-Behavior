# ============================================================
# MODEL 1 - Company Risk Prediction (ESG + Market Data)
# ============================================================

# STEP 1: Import Libraries
import pandas as pd
import numpy as np

# Set path relative to current script (because of VS code error using OS)
# 2. Load ESG dataset
esg = pd.read_csv("../Dataset/Model1_sp500_esg_data.csv")

# 3. Load stock price dataset
price = pd.read_csv("../Dataset/Model1_sp500_price_data.csv")

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
    0.6 * df['volatility_norm'] +
    0.4 * (1 - df['totalEsg_norm'])
)


# Create percentile groups (10 groups = deciles)
df['percentile_group'] = pd.qcut(df['risk_score'], q=10)

# Calculate min and max for each percentile
percentile_summary = df.groupby('percentile_group', observed=False)['risk_score'].agg(['min', 'max', 'count'])

# Reset index so it saves nicely in Excel with the 'percentile_group' column
risk_summary_df = percentile_summary.reset_index()

print("📊 Percentile-wise Risk Score Ranges:")
print(percentile_summary)

# Convert to percentage scale (0–100)
df['risk_score_100'] = df['risk_score'] * 100

# Define bins
bins = list(range(0, 110, 10))  # 0,10,20,...,100

# Create range labels
labels = [f"{i}-{i+10}" for i in bins[:-1]]

# Categorize into bins
df['risk_range'] = pd.cut(df['risk_score_100'], bins=bins, labels=labels, include_lowest=True)

# Count how many companies in each range
range_counts = df['risk_range'].value_counts().sort_index()

print("\n📊 Company Count in Risk Score Ranges (0–100):")
print(range_counts)

df.drop(columns=['percentile_group', 'risk_range', 'risk_score_100'], inplace=True, errors='ignore')


df['risk_class'] = pd.cut(
    df['risk_score'],
    bins=[0.0, 0.30, 0.50, 1.0],
    labels=['Low', 'Medium', 'High']
)

# --- Debug prints ---
print("Risk class distribution:")
print(df['risk_class'].value_counts())


print("\nSample (risk_score vs risk_class):")
print(df[['Symbol', 'risk_score', 'risk_class']].head(10))

# ============================================================
# VISUALIZATION - Dataset Analysis (Place after Step 5)
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns

# Set the visual style
sns.set_theme(style="whitegrid")

# 1. Risk Score Distribution (Histogram)
# This shows how your 0.6*Vol + 0.4*ESG formula distributed the companies
plt.figure(figsize=(10, 6))
sns.histplot(df['risk_score'], bins=20, kde=True, color='skyblue')
plt.axvline(0.30, color='red', linestyle='--', label='Low-Med Boundary')
plt.axvline(0.50, color='red', linestyle='--', label='Med-High Boundary')
plt.title('Distribution of Composite Risk Scores', fontsize=14, fontweight='bold')
plt.xlabel('Risk Score (0 to 1)')
plt.ylabel('Number of Companies')
plt.legend()
plt.tight_layout()
plt.show()

# 2. ESG vs. Volatility (Scatter Plot)
# This shows the "Non-Linear" relationship you want to prove to your instructor
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    data=df, 
    x='totalEsg', 
    y='volatility', 
    hue='GICS Sector', 
    style='risk_class',
    palette='viridis',
    s=100, 
    alpha=0.7
)
plt.title('ESG Performance vs. Stock Volatility by Sector', fontsize=14, fontweight='bold')
plt.xlabel('Total ESG Score')
plt.ylabel('Daily Volatility (Std Dev of Returns)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.show()


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
    'totalEsg',
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

#print("\nAfter encoding shape:", X.shape)

# --- 6.3 Scale features (important for SVM and KNN) ---
# StandardScaler makes all numbers on same scale
# so big numbers (marketCap) don't dominate small ones (beta)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaling done ✅")
print("Sample scaled values (first row):")
print(X_scaled[0])


# --- STEP 6.4: Save Preprocessed Data to a single Excel file ---

# 1. Define the path for the preprocessed data Excel file
PREPROCESSED_DATA_PATH = r'F:\ML_PROJECT\results\Preprocessed_Data.xlsx'

# 2. Use ExcelWriter to save multiple sheets
with pd.ExcelWriter(PREPROCESSED_DATA_PATH, engine='openpyxl') as writer:
    
    # Sheet 1: Unscaled Features (X after encoding)
    X.to_excel(writer, sheet_name='X_Unscaled', index=False)
    
    # Sheet 2: Scaled Features (X_scaled converted back to DataFrame)
    pd.DataFrame(X_scaled, columns=X.columns).to_excel(writer, sheet_name='X_Scaled', index=False)
    
    # Sheet 3: Labels (y)
    pd.DataFrame(y, columns=['target']).to_excel(writer, sheet_name='Y_Label', index=False)

print(f"✅ Preprocessed data saved to one Excel file with 3 sheets: {PREPROCESSED_DATA_PATH}")

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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Define all models in a dictionary ---
# This way we can loop through all of them instead of repeating code
models = {
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'SVM'                 : SVC(random_state=42, class_weight='balanced'),
    'KNN'                 : KNeighborsClassifier(),
    'Decision Tree'       : DecisionTreeClassifier(random_state=42, class_weight='balanced'),
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
    eval_metric='mlogloss'
    )
}

# --- Train and evaluate each model ---
results = []  # we'll store each model's scores here
all_confusion_matrices = {}  # NEW: Dictionary to store CMs for each model

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
    
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm, 
        index=[f'Actual {c}' for c in le.classes_], 
        columns=[f'Pred {c}' for c in le.classes_]
    )
    all_confusion_matrices[model_name] = cm_df # Store it for later

    print(f"✅ {model_name} done")

# --- Show comparison table ---
results_df = pd.DataFrame(results)
print("\n============================================================")
print("MODEL COMPARISON TABLE")
print("============================================================")
print(results_df.to_string(index=False))


# ============================================================
# STEP 9: Confusion Matrix 
# ============================================================

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

ax.set_title('Confusion Matrix - Random Forest', fontsize=13, fontweight='bold')
plt.tight_layout()
#plt.show()

print("Confusion Matrix saved ✅")

# --- Print it as numbers too ---
print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
print(pd.DataFrame(
    cm,
    index=[f'Actual {c}' for c in le.classes_],
    columns=[f'Pred {c}' for c in le.classes_]
))


importances = models['Random Forest'].feature_importances_

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
EXCEL_SAVE_PATH = r'F:\ML_PROJECT\results\Final_Results.xlsx'

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

    # Sheet 1: Full Dataset
    df.to_excel(writer, sheet_name='Full Dataset', index=False)

    # Sheet 2: Risk Score Summary (NEW)
    risk_summary_df.to_excel(writer, sheet_name='Risk Score Analysis', index=False)

    # Sheet 3: Market Features
    market_features.to_excel(writer, sheet_name='Market Features', index=False)

    # Sheet 4: Model Comparison
    results_df.to_excel(writer, sheet_name='Model Comparison', index=False)

    # NEW: Sheets 5 onwards -> One sheet for each Model's Confusion Matrix
    for model_name, cm_df in all_confusion_matrices.items():
        # Sheet names have a 31 character limit, so we clean the name slightly
        sheet_name = f"CM_{model_name}"[:31] 
        cm_df.to_excel(writer, sheet_name=sheet_name)

    # Sheet: Test Predictions (Actual vs Predicted for Best Model)
    test_results = pd.DataFrame({
        'Symbol': df.iloc[idx_test]['Symbol'].values,
        'Actual Risk': le.inverse_transform(y_test),
        'Predicted Risk': le.inverse_transform(y_pred_best),
        'Correct': ['✅' if a == p else '❌' for a, p in zip(y_test, y_pred_best)]
    })
    test_results.to_excel(writer, sheet_name='Test Predictions', index=False)

    # Sheet: Feature Importance
    feature_importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)
print("✅ Excel saved →", EXCEL_SAVE_PATH)
print("\nSheets inside:")
print("  📄 Full Dataset      → all 426 companies with risk class")
print("  📄 Market Features   → avg_return, volatility, momentum_6m")
print("  📄 Model Comparison  → accuracy, F1 of all 6 models")
print("  📄Excel saved with all Confusion Matrices and Risk Analysis")
print("  📄 Test Predictions  → company wise correct/wrong prediction")
print("  📄 Feature Importance → contribution of each feature to the model")

# ============================================================
# VISUALIZATION - Model Results (Place after Step 11)
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 3. Feature Importance Bar Chart (Random Forest)
# We use Random Forest because it is your "Winner" for Scenario B
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for plotting
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10) # Top 10

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette='magma')
plt.title('Top 10 Drivers of Corporate Risk (Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()

# 4. Multi-Model Confusion Matrices
# This loop generates a plot for EVERY model in your dictionary
for model_name, model in models.items():
    # Get predictions for the test set
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    
    plt.title(f'Confusion Matrix: {model_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

print("📊 All result visualizations generated successfully!")