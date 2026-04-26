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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

SAVE_PATH = r'C:\Users\mp2hl\Documents\Investment-Recommendation-System-Using-ESG-And-Investor-Behavior\Supervised_Weighted'

# Load Data
X = pd.read_csv(os.path.join(SAVE_PATH, 'X_scaled_weighted.csv'))
y_raw = pd.read_csv(os.path.join(SAVE_PATH, 'y_labels_weighted.csv')).values.ravel()

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'SVM': SVC(kernel='rbf', class_weight='balanced', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='mlogloss')
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
        'F1 Score': round(f1_score(y_test, y_pred, average='weighted'), 4)
    })

results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

# Prepare Feature Importance for Best Model
importance_df = pd.DataFrame()
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': best_model.feature_importances_}).sort_values(by='Importance', ascending=False)
elif hasattr(best_model, 'coef_'):
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(best_model.coef_[0])}).sort_values(by='Importance', ascending=False)

# Confusion Matrix for Best Model
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
cm_df = pd.DataFrame(cm, index=[f'Actual_{c}' for c in le.classes_], columns=[f'Pred_{c}' for c in le.classes_])

# Save to Excel
with pd.ExcelWriter(os.path.join(SAVE_PATH, 'Weighted_Model_Results.xlsx')) as writer:
    results_df.to_excel(writer, sheet_name='Comparison', index=False)
    cm_df.to_excel(writer, sheet_name='Best_Model_CM')
    if not importance_df.empty:
        importance_df.to_excel(writer, sheet_name='Best_Importance', index=False)

# Graphing Best CM
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"Confusion Matrix: {best_model_name} (Weighted System)")
plt.savefig(os.path.join(SAVE_PATH, 'best_weighted_cm.png'))
plt.close()

print(f"Analysis complete. Best model: {best_model_name}")