# 🏦 Credit Risk Prediction with Random Forest vs. XGBoost & SMOTE

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-XGBoost-orange)
![Technique](https://img.shields.io/badge/Technique-SMOTE-green) 
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## 📌 Project Overview
This project focuses on predicting **loan defaults (Credit Risk)** using machine learning. The primary challenge was handling an **imbalanced dataset**, where risky loans (Class 0) were significantly underrepresented compared to safe loans.

Using **SMOTE** for data balancing and comparing **Random Forest** vs. **XGBoost**, I aimed to maximize the detection of risky loans (Recall) without creating excessive false alarms (Precision).

## 🛠️ Tech Stack & Methodology
* **Data Processing:** Pandas, NumPy
* **Handling Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique)
* **Models:** Random Forest Classifier, XGBoost (Extreme Gradient Boosting)
* **Optimization:** GridSearchCV (Hyperparameter Tuning), Threshold Moving

---

## 📊 Model Benchmark: The Evolution

I tested three different iterations of the XGBoost model to find the best business fit.

| Model Version | Risk Detection (Recall - Class 0) | Reliability (Precision - Class 0) | False Alarm (False Positive) | Comment |
| :--- | :---: | :---: | :---: | :--- |
| **1. First Model** (Standard) | **53%** (16 Count) | **64%** | 9 Count | **Most Balanced:** Captures risks well while keeping errors low. |
| **2. Second Model** (Tuned) | **40%** (12 Count) | **67%** | 6 Count | **Too Conservative:** Made very few mistakes but missed significant risks. |
| **3. Final Model** (Threshold 0.7) | **50%** (15 Count) | **56%** | 12 Count | **More Aggressive:** Detection rate increased (40% -> 50%), but "noise" (false alarms) increased as well. |

---

## 🎯 Business Conclusion: Why I Selected the Standard Model?

In credit risk analysis, the cost of missing a default (**False Negative**) is significantly higher than the operational cost of checking a false alarm (**False Positive**).

1.  **Priority on Recall:** The **Standard XGBoost Model** achieved the highest Recall (**53%**), successfully identifying the most high-risk applicants compared to other iterations.
2.  **The Problem with Tuning:** While the Tuned Model improved Precision (**67%**), it became too conservative, failing to detect a larger portion of actual defaults. This "safe" behavior creates a hidden financial risk.
3.  **Final Decision:** I selected the **Standard Model** for deployment because it offers the best trade-off, maximizing risk detection while maintaining a manageable False Positive rate.

---

## 📈 Key Visualizations

### Confusion Matrix Comparison
*(You can upload your confusion matrix images here)*
> The Standard Model (Left) vs. The Tuned Model (Right) shows how we traded Recall for Precision.

### Precision-Recall Trade-off
> Threshold moving was used to explore how aggressive the model should be.

---


