# ============================================================
# SCRIPT 2: Logistic Regression & SVM Training
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.preprocessing import label_binarize

# Set the save path
SAVE_PATH = r'C:\Users\priya\Documents\Investment-Recommendation-System-Using-ESG-And-Investor-Behavior\Supervised_Learning_Initial'
DATA_FILE = os.path.join(SAVE_PATH, 'Preprocessed_Data.xlsx')

# --- Step 1: Load Scaled Data ---
X_scaled = pd.read_excel(DATA_FILE, sheet_name='X_Scaled')
y = pd.read_excel(DATA_FILE, sheet_name='Y_Label').values.ravel()

# --- Step 2: Train Test Split ---
# Keeping random_state=42 and stratify=y to match Script 1 exactly
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- Step 3: Define Models (Parameters kept as requested) ---
models = {
    'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
}

# --- Step 4: Train and Evaluate ---
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Calculate Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    }

    # Save Metrics & Predictions to Excel
    results_path = os.path.join(SAVE_PATH, f'Results_{name}.xlsx')
    with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
        pd.DataFrame([metrics]).to_excel(writer, sheet_name='Metrics', index=False)
        pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_excel(writer, sheet_name='Predictions', index=False)
        
        # Feature Importance (Coefficient for LR)
        if name == 'Logistic_Regression':
            importance = pd.DataFrame({
                'Feature': X_scaled.columns,
                'Importance': np.abs(model.coef_[0])
            }).sort_values(by='Importance', ascending=False)
            importance.to_excel(writer, sheet_name='Feature_Importance', index=False)

    # --- Visualizations ---
    
    # 1. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f'Confusion Matrix: {name}')
    plt.savefig(os.path.join(SAVE_PATH, f'CM_{name}.png'))
    plt.close()

    # 2. ROC Curve (Multi-class)
    plt.figure(figsize=(8, 6))
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc(fpr, tpr):.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve: {name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, f'ROC_{name}.png'))
    plt.close()

    print(f"{name} results and graphs saved.")