# ============================================================
# SCRIPT 3: Supervised Validation on GMM Clusters
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix)

# 1. Setup paths
SAVE_PATH = r'C:\Users\mp2hl\Documents\Investment-Recommendation-System-Using-ESG-And-Investor-Behavior\Unsupervised_GMM'
DATA_FILE = os.path.join(SAVE_PATH, 'GMM_Labeled_Data.csv')

# 2. Load the GMM Labeled data
df = pd.read_csv(DATA_FILE)
X = df.drop(columns=['gmm_cluster'])
y = df['gmm_cluster']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\nData Split Complete: Training ({X_train.shape[0]}), Testing ({X_test.shape[0]})")

# 4. Model Definitions (Parameters kept exactly as provided)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'SVM': SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='mlogloss')
}

# 5. Training Loop
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4)
    })
    print(f"{name} trained.")

# 6. Comparison Table
results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n🏆 Winner: {best_model_name}")

# 7. Best Model Confusion Matrix
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
cm_df = pd.DataFrame(cm, index=[f'Actual_{i}' for i in range(len(np.unique(y)))], 
                         columns=[f'Pred_{i}' for i in range(len(np.unique(y)))])

# 8. Feature Importance for Best Model
importance_df = pd.DataFrame()
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
elif hasattr(best_model, 'coef_'):
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(best_model.coef_[0])
    }).sort_values(by='Importance', ascending=False)

# 9. Save All Results to Excel
RESULTS_PATH = os.path.join(SAVE_PATH, 'Supervised_GMM_Results.xlsx')
with pd.ExcelWriter(RESULTS_PATH, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
    cm_df.to_excel(writer, sheet_name='Best_Model_CM')
    if not importance_df.empty:
        importance_df.to_excel(writer, sheet_name='Best_Model_Importance', index=False)

# 10. Visualization (CM Plot)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title(f"Confusion Matrix: {best_model_name}\n(Predicting GMM Clusters)")
plt.ylabel('Actual Cluster')
plt.xlabel('Predicted Cluster')
plt.savefig(os.path.join(SAVE_PATH, 'best_model_gmm_cm.png'))
plt.close()

print(f"Final results and graphs saved to: {SAVE_PATH}")