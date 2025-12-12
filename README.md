# 🏦 Credit Risk Prediction with XGBoost & SMOTE

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-XGBoost-orange)
![Technique](https://img.shields.io/badge/Technique-SMOTE-green)
![Context](https://img.shields.io/badge/Context-Masters%20School-purple)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## 🎓 Context & Objective
This project was developed as a **Technical Case Study** for a **Mock Interview** session at **Masters School**. 

The primary objective was to simulate a real-world Data Science take-home assignment, demonstrating end-to-end capabilities in:
* Handling imbalanced financial datasets.
* Implementing and comparing tree-based algorithms (Random Forest vs. XGBoost).
* Translating technical metrics (Recall/Precision) into actionable business decisions.

---

## 📌 Project Overview
The core task focuses on predicting **loan defaults (Credit Risk)**. The dataset was highly imbalanced, where risky loans (Class 0) were significantly underrepresented. 

Using **SMOTE** for data balancing and applying **Threshold Moving** techniques, I aimed to maximize the detection of risky loans (Recall) while maintaining a reasonable false alarm rate (Precision).

## 🛠️ Tech Stack & Methodology
* **Data Processing:** Pandas, NumPy
* **Handling Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique)
* **Models:** Random Forest Classifier, XGBoost (Extreme Gradient Boosting)
* **Optimization:** GridSearchCV (Hyperparameter Tuning), Threshold Moving

---
