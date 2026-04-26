# ============================================================
# SCRIPT 5: Final Model Comparison & Best Model Analysis
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

# Set the save path
SAVE_PATH = r'C:\Users\mp2hl\Documents\Investment-Recommendation-System-Using-ESG-And-Investor-Behavior\Supervised_Learning_Initial'

# List of models we trained
model_names = ['Logistic_Regression', 'SVM', 'KNN', 'Decision_Tree', 'Random_Forest', 'XGBoost']

# --- Step 1: Compile Metrics ---
all_metrics = []

for name in model_names:
    file_path = os.path.join(SAVE_PATH, f'Results_{name}.xlsx')
    
    if os.path.exists(file_path):
        # Load the metrics sheet
        m_df = pd.read_excel(file_path, sheet_name='Metrics')
        m_df.insert(0, 'Model', name) # Add name column
        all_metrics.append(m_df)
    else:
        print(f"Warning: Results for {name} not found.")

# Combine all into one table
comparison_df = pd.concat(all_metrics, ignore_index=True)
comparison_df = comparison_df.sort_values(by='F1_Score', ascending=False)

# --- Step 2: Identify the Best Model ---
best_model_name = comparison_df.iloc[0]['Model']
print(f"🏆 The Best Performing Model is: {best_model_name}")

# --- Step 3: Extract Best Model Details ---
best_file_path = os.path.join(SAVE_PATH, f'Results_{best_model_name}.xlsx')
preds_df = pd.read_excel(best_file_path, sheet_name='Predictions')

# Re-calculate Confusion Matrix for the Excel sheet
cm = confusion_matrix(preds_df['Actual'], preds_df['Predicted'])
cm_df = pd.DataFrame(
    cm, 
    index=['Actual_Low', 'Actual_Med', 'Actual_High'], 
    columns=['Pred_Low', 'Pred_Med', 'Pred_High']
)

# --- Step 4: Save to Final Excel ---
FINAL_OUT = os.path.join(SAVE_PATH, 'Final_Model_Comparison.xlsx')

with pd.ExcelWriter(FINAL_OUT, engine='openpyxl') as writer:
    comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
    cm_df.to_excel(writer, sheet_name='Best_Model_CM')
    
    # Try to grab Feature Importance if it exists for the best model
    try:
        importance_df = pd.read_excel(best_file_path, sheet_name='Feature_Importance')
        importance_df.to_excel(writer, sheet_name='Best_Model_Importance', index=False)
        
        # --- Step 5: Graph Feature Importance of Best Model ---
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', palette='viridis')
        plt.title(f'Top 10 Drivers of Risk: {best_model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH, 'best_model_feature_importance.png'))
        plt.close()
        print("Best model feature importance graph saved.")
        
    except ValueError:
        print(f"Note: {best_model_name} does not have a Feature Importance sheet. Skipping graph.")

print(f"Final comparison saved to: {FINAL_OUT}")