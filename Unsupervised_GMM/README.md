# Unsupervised Learning: Gaussian Mixture Models (GMM)

## 1. Objective
The goal of this phase was to explore whether companies naturally form risk-based groups using unsupervised learning, instead of relying on predefined labels. This allows us to see if the internal patterns of ESG performance and market behavior create their own distinct "risk profiles" without human intervention.

## 2. Dataset Description
We used the same merged dataset containing:
* **Market Features:** Volatility, Beta, Avg_Return, Momentum_6M.
* **ESG Features:** EnvironmentScore, SocialScore, GovernanceScore, HighestControversy.

All features were standardized using **StandardScaler** before clustering to ensure that features with larger scales (like Market Cap) did not dominate the model.

## 3. Methodology
We applied **Gaussian Mixture Model (GMM)** for clustering. GMM is a probabilistic clustering technique that assumes data is generated from a mixture of multiple Gaussian distributions.

**Key steps:**
1.  Input features were scaled.
2.  GMM was applied to identify underlying patterns.
3.  The model grouped companies into clusters based on mathematical similarity.



## 4. Number of Clusters
Using model selection criteria (**BIC - Bayesian Information Criterion**), the optimal number of clusters was found to be **4 clusters**. This suggests that the data naturally forms four distinct groups based on combined ESG and market characteristics, rather than the three groups we manually defined in the first phase.

## 5. Cluster Interpretation
Each cluster represents a group of companies with similar market behavior and ESG characteristics. These clusters act as data-driven profiles of company risk and stability.

### Cluster Summary
| Cluster | Company Count |
| :--- | :--- |
| 0 | 44 |
| 1 | 184 |
| 2 | 67 |
| 3 | 131 |

### Cluster Feature Averages (Raw/Log Values)
| Feature | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
| :--- | :--- | :--- | :--- | :--- |
| Environment Score | 1.6821 | 1.0995 | 1.3130 | 2.4686 |
| Social Score | 10.3282 | 8.6214 | 8.1436 | 9.7546 |
| Governance Score | 2.0237 | 2.0587 | 2.0286 | 1.9155 |
| Total ESG | 23.8734 | 18.2457 | 18.5228 | 26.9791 |
| Highest Controversy | 2.1364 | 1.7174 | 1.7612 | 2.0916 |
| Market Cap | 23.9493 | 24.7855 | 24.5520 | 24.4311 |
| Beta | 1.0070 | 0.9844 | 1.3014 | 0.9823 |
| Avg Return | -0.0007 | 0.0008 | 0.0011 | 0.0006 |
| Volatility | 0.0198 | 0.0147 | 0.0223 | 0.0149 |
| Momentum (6M) | -0.2470 | 0.2818 | 0.3566 | 0.2203 |

---

## 6. Supervised Learning on Clusters
To further validate the clustering, we treated the cluster labels as target classes and trained supervised models to predict cluster membership. 

### Results
Supervised models achieved exceptionally high classification accuracy, confirming that the GMM clusters are well-structured and easily learnable in the feature space.

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.9767** | **0.9797** | **0.9767** | **0.9761** |
| XGBoost | 0.9651 | 0.9670 | 0.9651 | 0.9643 |
| Decision Tree | 0.9302 | 0.9384 | 0.9302 | 0.9325 |
| Logistic Regression | 0.9070 | 0.9172 | 0.9070 | 0.9093 |
| SVM | 0.8721 | 0.8986 | 0.8721 | 0.8792 |
| KNN | 0.8140 | 0.8079 | 0.8140 | 0.8027 |

## 7. Key Insights
Instead of forcing predefined risk categories, this approach allows the data to define its own structure. It provides an alternative perspective where:
* **Risk is not manually imposed**: The groupings emerge naturally from the data points.
* **Strong Separability**: The high accuracy of the supervised models (especially Random Forest) indicates that the GMM successfully identified distinct boundaries between clusters.

## 8. Conclusion
The GMM-based approach demonstrates that companies can be effectively grouped into 4 distinct clusters based on ESG and market features. These clusters are highly predictable and structured, offering a robust data-driven alternative to manual risk classification. This method serves as a critical complement to our initial supervised approach by uncovering the underlying distribution of the S&P 500 risk landscape.