# Corporate Risk Prediction: ESG & Market Behavior Pipeline

This repository contains a modular supervised learning pipeline designed to classify S&P 500 companies into **Low**, **Medium**, and **High** risk categories. The project moves beyond simple linear formulas by integrating Environmental, Social, and Governance (ESG) metrics with historical market volatility.

## Project Architecture
To maintain a professional workflow, the project is broken into five sequential scripts. This modularity ensures that data preparation is isolated from model training and final evaluation.

1.  **`Preprocessing.py`**: Handles data cleaning, feature engineering (Volatility, Momentum, Returns), and the initial composite risk scoring.
2.  **`Logistic_and_SVM.py`**: Baseline testing using Logistic Regression and Support Vector Machines with balanced class weights.
3.  **`KNN_and_DecisionTree.py`**: Evaluates instance-based (KNN) and rule-based (Decision Tree) logic.
4.  **`RandomForest_and_XGBoost.py`**: Implementation of ensemble methods (Bagging and Boosting) for higher predictive power.
5.  **`Final_Result_Compilation.py`**: Aggregates all model results, ranks performance by F1-Score, and extracts the drivers of the winning model.

---

## Model Performance & Results
We evaluated six different algorithms. **Random Forest** emerged as the top performer, providing the best balance between precision and recall across the three risk classes.

### Final Comparison Table
| Model | Accuracy | Precision | Recall | F1_Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random_Forest** | **0.7326** | **0.7440** | **0.7326** | **0.7298** |
| XGBoost | 0.7209 | 0.7218 | 0.7209 | 0.7172 |
| KNN | 0.6977 | 0.6962 | 0.6977 | 0.6922 |
| SVM | 0.6628 | 0.6948 | 0.6628 | 0.6671 |
| Logistic_Regression | 0.6628 | 0.6918 | 0.6628 | 0.6657 |
| Decision_Tree | 0.6628 | 0.6652 | 0.6628 | 0.6639 |

### Best Model (Random Forest) Confusion Matrix
| | Pred_Low | Pred_Med | Pred_High |
| :--- | :---: | :---: | :---: |
| **Actual_Low** | 11 | 0 | 8 |
| **Actual_Med** | 0 | 11 | 7 |
| **Actual_High** | 7 | 1 | 41 |

---

## Feature Importance & Logic Validation
A key objective was to determine if risk could be predicted by balancing internal ESG performance with external market behavior. Our initial formula used a **60/40 split** (Market Volatility vs. ESG).

The feature importance from the Random Forest model validates this approach. The top drivers show that **Social and Environmental scores** contribute almost equally to **Market Beta and Momentum**. This confirms that corporate risk is a hybrid of how a company operates and how the market perceives its stability.

**Top Risk Drivers:**
1. Social Score (19.1%)
2. Environment Score (18.5%)
3. Beta (12.3%)
4. Market Cap (9.3%)
5. 6-Month Momentum (9.2%)
6. Governance Score	(8.0%)
7. Average Return	(7.8%)
8. Highest Controversy	(3.7%)

---

## Key Limitations
While the pipeline achieved a ~73% F1-score, the performance hit a ceiling due to the inherent constraints of the initial risk definition:

* **Oversimplified Risk Formula**: The target labels were generated using only two features (volatility + ESG), ignoring critical drivers like Beta and Governance quality that the model later identified as important.
* **Hard Threshold-Based Labeling**: Using fixed cutoffs (0.3, 0.5) created rigid boundaries. Real-world risk is a continuous spectrum; forcing it into 3 "buckets" caused information loss and overlap between Medium and High risk groups.
* **Label Noise**: Companies sitting near the 0.3 or 0.5 boundaries essentially "confused" the model during training, leading to misclassification in the test set.

## Conclusion
This initial supervised approach provided a strong baseline but proved that risk cannot be fully captured by a simple weighted formula. 

---

### How to Run
1. Ensure `sp500_esg_data.csv` and `sp500_price_data.csv` are in the folder.
2. Run scripts `01` through `05` in order.
3. Outputs (Excel results and PNG graphs) will be generated in the `Supervised_Learning_Initial` directory.