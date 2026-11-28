# Model Development

## 1. Objective

The goal of this phase is to train and compare machine learning models that predict employee attrition (`Yes` or `No`).

## 2. Selected Models

Three classification models were selected:

- **Logistic Regression:** A simple, interpretable baseline model.
- **Decision Tree:** A tree-based model that learns decision rules from employee features.
- **Random Forest:** An ensemble model that combines multiple decision trees.

## 3. Training Data

- Training samples: 1,176 employees.
- Target: `Attrition` (`No = 0`, `Yes = 1`).
- Features: The selected employee features prepared during Phase 04.

The separate test dataset is reserved for final evaluation in Phase 06.

## 4. Training Method

Each model was trained and compared using **5-Fold Stratified Cross-Validation**.

A Scikit-learn Pipeline applies preprocessing within each cross-validation fold to prevent data leakage.

Preprocessing includes:

- StandardScaler for numerical features.
- OneHotEncoder for categorical features.

## 5. Hyperparameter Tuning

GridSearchCV was used to test different model settings.

**F1-score** was used as the tuning metric because employee attrition is an imbalanced classification problem.

Precision, Recall, and ROC-AUC were also examined.

## 6. Candidate Model

Logistic Regression achieved the highest cross-validation F1-score among the three tuned models.

It was selected as the candidate model for Phase 06, not as the final approved model.

The trained pipeline was saved locally as:

`models/attrition_pipeline.joblib`

## 7. Next Phase

Phase 06 will evaluate the selected model on the untouched test dataset, examine classification errors, and assess the trade-off between Precision and Recall.