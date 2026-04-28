# 🚢 Titanic Survival Prediction using ML Pipeline

## 🎯 Objective

Build an end-to-end machine learning pipeline to predict passenger survival on the Titanic dataset using proper preprocessing, model comparison, and evaluation techniques.

---

## ⚙️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-red)
![Pandas](https://img.shields.io/badge/Pandas-latest-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-latest-013243?logo=numpy&logoColor=white)

---

## 📂 Project Structure

```
titanic-ml-pipeline/
│
├── notebook/
│   └── analysis.ipynb
│
├── src/
│   └── train.py
│
├── README.md
└── requirements.txt
```

---

## 🔄 ML Pipeline Workflow

### 1. Data Preprocessing
- Missing value imputation (median & most frequent)
- Feature scaling (`StandardScaler`)
- Categorical encoding (`OneHotEncoder`)

### 2. Model Training
- Logistic Regression
- Random Forest
- XGBoost

### 3. Evaluation Strategy
- Stratified K-Fold Cross Validation
- Separate Test Set Evaluation

---

## 📊 Model Performance

| Model | CV Score | CV Std | Test Score |
|---|---|---|---|
| Logistic Regression | 0.79 | 0.016 | 0.77 |
| Random Forest | **0.81** | 0.017 | **0.82** |
| XGBoost | 0.80 | 0.018 | 0.80 |

---

## 🏆 Best Model

**Random Forest** performed the best based on:
- Highest test accuracy (0.82)
- Stable cross-validation performance
- Better generalization compared to other models

---

## 🧠 Key Learnings

- Pipelines prevent data leakage by applying preprocessing within CV folds
- Cross-validation provides robust performance estimation
- Test set ensures unbiased final evaluation
- Ensemble models (Random Forest, XGBoost) handle non-linear relationships effectively

---

## ⚠️ Challenges Faced

- Avoiding data leakage during preprocessing
- Correctly interpreting CV vs Test performance
- Handling categorical and numerical features together in a single pipeline

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/train.py
```

---

## 🚀 Future Improvements

- [ ] Feature engineering (interaction features, binning)
- [ ] Hyperparameter tuning with `RandomizedSearchCV`
- [ ] Model explainability (feature importance, SHAP values)
- [ ] Deploy model using Flask or FastAPI
