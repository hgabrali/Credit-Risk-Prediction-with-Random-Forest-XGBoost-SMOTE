# 🏦 Credit Risk Prediction: End-to-End Classification Analysis

---

## 1. 🔍 Data Inspection & Exploratory Data Analysis (EDA)
Before modeling, a thorough investigation of the `loans_modified` dataset was conducted to understand data integrity and distributions.

### Step 1.1: Data Integrity Checks
* **Missing Values:** Checked for null values in critical columns (e.g., income, loan amount). Strategies such as mean imputation or row dropping were considered based on the ratio of missing data.
* **Data Types:** Ensured that categorical variables (e.g., `term`, `grade`) were strictly separated from numerical features.
* **Duplicates:** Verified and removed duplicate entries to prevent data leakage.

### Step 1.2: Univariate & Bivariate Analysis
* **Target Distribution:** Visualized the `loan_status` column.
    * *Observation:* The dataset is highly imbalanced. The ratio of Safe Loans (Class 1) to Risky Loans (Class 0) was significantly high, indicating that standard accuracy metrics would be misleading.
* **Correlation Matrix:** Analyzed the heatmap to identify multicollinearity between features.

---

## 2. ⚙️ Data Preprocessing & Feature Engineering
To prepare the data for tree-based algorithms (Random Forest & XGBoost), the following steps were implemented:

### Step 2.1: Encoding & Scaling
* **Categorical Encoding:** Applied One-Hot Encoding (or Label Encoding) to transform string variables into machine-readable numeric formats.
* **Scaling:** While tree-based models are robust to unscaled data, scaling was reviewed for potential future model comparisons (e.g., Logistic Regression).

### Step 2.2: Handling Class Imbalance (SMOTE)
Given the scarcity of the minority class (Class 0 - Risky Loans), I utilized **SMOTE (Synthetic Minority Over-sampling Technique)**.
* **Strategy:** SMOTE was applied **only to the training set** to prevent data leakage.
* **Result:** The training data distribution was balanced to a 50/50 ratio, forcing the model to learn the patterns of risky borrowers rather than biasing towards the majority class.

```python
# Snippet: Correct application of SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

## 3. 🤖 Model Selection & Optimization

I experimented with three distinct modeling phases to find the optimal configuration for the problem.

### 🔹 Phase 1: Baseline Models
* **Random Forest:** Used as a robust baseline algorithm to establish initial performance benchmarks.
* **XGBoost (Standard):** Implemented with default parameters. Note that `scale_pos_weight` was adjusted implicitly via the balanced training data from **SMOTE**.

### 🔹 Phase 2: Hyperparameter Tuning
* **Technique:** `GridSearchCV`
* **Parameters Tuned:** `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight`.
* **Outcome:** The tuning process resulted in a model that was **highly precise but too conservative**. It drastically reduced the **Recall rate**, meaning the model missed too many actual defaults (Class 0).

### 🔹 Phase 3: Threshold Moving
* **Technique:** Adjusted the decision threshold from the standard `0.5` to experimental values of `0.3` and `0.7`.
* **Goal:** To aggressively capture **Class 0** instances by lowering the probability confidence required for the model to classify a loan as "Risky".


## 4. 📊 Results & Business Evaluation

The following table summarizes the performance evolution across the three distinct modeling phases.

| Model Version | Risk Detection (Recall - Class 0) | Reliability (Precision - Class 0) | False Alarm (False Positive) | Comment |
| :--- | :---: | :---: | :---: | :--- |
| **1. First Model** (Standard) | **53%** (16 Count) | **64%** | 9 Count | **Most Balanced:** Captures risks well while keeping errors low. |
| **2. Second Model** (Tuned) | **40%** (12 Count) | **67%** | 6 Count | **Too Conservative:** Made very few mistakes but missed significant risks. |
| **3. Final Model** (Threshold 0.7) | **50%** (15 Count) | **56%** | 12 Count | **More Aggressive:** Detection rate increased (40% -> 50%), but "noise" (false alarms) increased. |

---

### 🎯 Business Conclusion: Why the Standard Model?

In the banking domain, the cost of a **False Negative** (approving a bad loan → financial loss) is typically significantly higher than a **False Positive** (rejecting a good loan → opportunity cost).

* **Recall Priority:** The **Standard XGBoost Model** provided the highest Recall (**53%**), successfully identifying the most high-risk applicants compared to other iterations.
* **Stability:** The Tuned model proved to be too "safe" (overfitted to precision), while the Threshold-Moved model introduced excessive noise without a proportional gain in safety.
* **Final Decision:** The **Standard Model** was selected as the **champion model** for deployment due to its superior trade-off between risk mitigation and operational efficiency.

---

## 🚀 Future Steps

To further improve the model's performance and deployment readiness, the following steps are planned:

* **Feature Engineering:** Deriving new financial features (e.g., *Debt-to-Income ratio*, *Credit Utilization*) to improve model separability.
* **Ensemble Methods:** Experimenting with Stacking architectures, combining XGBoost with LightGBM or CatBoost.
* **Deployment:** Creating a Streamlit API for real-time inference and visualization.

---

# 🏦 Credit Risk Prediction: Handling Imbalanced Data
**Technical Case Study | Mock Interview Session**

## 📌 Project Overview
This project focuses on predicting loan defaults (Credit Risk) using a highly imbalanced dataset. The primary challenge was to accurately identify risky loans (Class 0) without significantly increasing false alarms, simulating a real-world financial risk scenario.

## 🛠 Tech Stack & Methodology
* **Data Processing:** Pandas, NumPy
* **Handling Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique)
* **Models:** * Random Forest Classifier
    * XGBoost (Extreme Gradient Boosting)
    * Gradient Boosting (Scikit-Learn)
* **Optimization:** Threshold Moving for maximizing Recall/Precision trade-off.

## 📊 Key Results & Business Decision
We compared **Random Forest** vs. **Gradient Boosting**. The results highlighted a trade-off between detecting risky loans (Recall) and model stability.

| Model | Threshold | Recall (Risky Loans) | Precision | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.50 (Default) | 47% | 70% | **0.78** |
| **Random Forest** | **0.43 (Optimized)** | **40%** | **80%** | **0.78** |
| Gradient Boosting | 0.21 (Optimized) | 33% | 91% | 0.67 |

### 🏆 Conclusion
**Random Forest** was selected as the final model due to its superior stability and higher ROC-AUC score (0.78). 

Using **Threshold Moving**, we identified that lowering the decision threshold to **0.43** allows the bank to maintain a high precision (**80%**), ensuring that when the model predicts a risk, it is highly likely to be correct, thus optimizing operational efficiency.

## 📂 Repository Structure
* `loans_modified.csv`: The dataset used for training and testing.
* `Hande_Mock interview_Loans_modified.ipynb`: The Jupyter Notebook containing the full end-to-end analysis, code, and visualizations.
