# Supervised Learning: Multi-Factor Weighted Risk Classification

## 1. Objective
The goal of this phase was to develop a realistic and comprehensive risk definition by combining multiple ESG and market features. Unlike the initial model, this approach uses supervised learning to classify companies into categories defined by a complex, multi-dimensional risk engine.

## 2. Motivation
The initial model's performance was capped at ~73% due to several factors we identified:
* **Oversimplified risk definition**: Relying on too few variables.
* **Rigid class boundaries**: Using fixed thresholds that didn't account for data distribution.
* **Information loss**: Failing to capture the interplay between different ESG pillars and market metrics.

This motivated a move toward a more structured, feature-rich formulation that balances short-term behavior with long-term stability.

## 3. Feature Selection & Preprocessing
We grouped the data into two primary components to ensure a balanced view of corporate risk. **Note:** `totalEsg` was excluded to avoid redundancy with the individual pillar scores.

### A. Market Risk (Short-term behavior)
* **Volatility**, **Beta**, **Avg_Return**, **Momentum_6M**.

### B. ESG Risk (Long-term stability)
* **EnvironmentScore**, **SocialScore**, **GovernanceScore**, **HighestControversy**.

### Data Transformation
To ensure "higher" values consistently represent "higher risk" across all inputs, we applied the following transformations:
* **Log/Signed-Log**: Applied to skewed features to normalize distribution.
* **Inversion (1 - Score)**: Applied to ESG scores and Returns so that poor performance/low returns result in a higher risk value.
* **Absolute Value**: Applied to Momentum to capture extreme price swings in either direction.

---

## 4. Risk Score Formulation & Classification
We defined the continuous risk score using a 50/50 balance:

$$\text{Total Risk} = 0.5 \times \text{Market Risk} + 0.5 \times \text{ESG Risk}$$

### Percentile-Based Binning
Instead of rigid thresholds, we used **Quantile Binning** to create three perfectly balanced classes:
* **Low Risk**: Bottom 33%
* **Medium Risk**: Middle 33%
* **High Risk**: Top 33%

This approach reduced training bias and ensured the models learned to differentiate relative risk within the S&P 500.

---

## 5. Results & Model Comparison
We tested two scenarios. While Case 1 (including Volatility as a feature) showed ~98% accuracy, we identified this as **Data Leakage** since Volatility was a primary component of the target label itself. 

The results below reflect **Case 2**, where Volatility was removed from the input features to create a more realistic and robust predictive scenario.

### Performance Table (Case 2)
| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.8488** | **0.8550** | **0.8488** | **0.8508** |
| SVM | 0.8140 | 0.8304 | 0.8140 | 0.8182 |
| Random Forest | 0.8023 | 0.8103 | 0.8023 | 0.8045 |
| XGBoost | 0.7907 | 0.8159 | 0.7907 | 0.7945 |
| KNN | 0.7558 | 0.7558 | 0.7558 | 0.7558 |
| Decision Tree | 0.6744 | 0.6973 | 0.6744 | 0.6798 |

### Top Drivers of Risk (Logistic Regression)
| Feature | Importance (Abs Coef) |
| :--- | :--- |
| **Environment Score** | 2.5428 |
| **Beta** | 2.0175 |
| **Social Score** | 1.6748 |
| **Governance Score** | 1.3197 |
| Momentum (6M) | 0.2645 |
| Highest Controversy | 0.1949 |
| Avg Return | 0.1431 |

---

## 6. Key Insights & Conclusion
* **Balanced Risk Representation**: The importance of `environmentScore` and `beta` as the top two drivers confirms that our objective was met: risk is indeed a combination of ESG pillars and systematic market behavior.
* **Leakage Awareness**: The drop from 98% (Case 1) to 84% (Case 2) represents a more honest and generalizable model. It shows that even without the dominant "Volatility" feature, the model can infer risk through other variables.
* **Stability**: Percentile-based labeling provided a much more stable training environment compared to our initial experiment.

**Conclusion:** The weighted risk model provides the most comprehensive and interpretable approach to risk classification in this project. By avoiding rigid thresholds and incorporating multiple dimensions of stability, we have created a robust framework for identifying corporate risk profiles.