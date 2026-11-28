# Model Development Report

## 1. Overview

Three machine learning models were trained and compared to predict employee attrition using the IBM HR Employee Attrition dataset.

**Models evaluated:**
- Logistic Regression
- Decision Tree
- Random Forest

The experiments used 1,176 training samples and 5-Fold Stratified Cross-Validation.

The separate test dataset was not used during this phase.

## 2. Initial Cross-Validation Results

| Model | F1-score | Recall | Precision | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.510 | 0.763 | 0.385 | 0.829 |
| Decision Tree | 0.364 | 0.479 | 0.301 | 0.645 |
| Random Forest | 0.476 | 0.447 | 0.527 | 0.799 |

Logistic Regression achieved the highest initial F1-score, Recall, and ROC-AUC.

## 3. Hyperparameter Tuning Results

GridSearchCV was used to optimize the three models based on F1-score.

| Model | Initial F1 | Best Tuned F1 |
|---|---:|---:|
| Logistic Regression | 0.510 | **0.526** |
| Decision Tree | 0.364 | 0.406 |
| Random Forest | 0.476 | 0.494 |

### Best Parameters

**Logistic Regression**
- C: 10
- class_weight: None

**Decision Tree**
- max_depth: 3
- min_samples_leaf: 1
- class_weight: balanced

**Random Forest**
- n_estimators: 100
- max_depth: 6
- min_samples_leaf: 5
- class_weight: balanced

## 4. Candidate Model Performance

Logistic Regression achieved the highest tuned F1-score and was selected as the candidate for further evaluation.

| Metric | Tuned Logistic Regression |
|---|---:|
| F1-score | 0.526 |
| Recall | 0.421 |
| Precision | 0.712 |
| ROC-AUC | 0.834 |

The tuned model improved F1-score and Precision compared with the initial configuration. However, Recall decreased from 0.763 to 0.421.

This trade-off is important because identifying employees who may leave is a key objective of the project.

These results come from the training-data cross-validation process. The tuned model was rechecked using the same cross-validation setup used for model selection, so the reported performance is not an independent final estimate.

## 5. Model Selection

**Candidate model: Logistic Regression**

Reasons:
- Highest tuned cross-validation F1-score among the tested models.
- Strong ROC-AUC compared with the alternatives.
- Relatively simple and interpretable.

This selection is provisional. The final operating threshold and Precision–Recall trade-off still need to be assessed.

The trained model and preprocessing pipeline were saved locally as:

`models/attrition_pipeline.joblib`

## 6. Limitations and Next Steps

- The dataset is relatively small and has an imbalanced target.
- Hyperparameter tuning results may be optimistic because model settings were selected using cross-validation.
- The model identifies statistical patterns, not proven causes of employee attrition.
- The dataset is synthetic and must not be treated as validation for real-world employment decisions.

**Phase 06 — Model Evaluation** will assess the candidate on the held-out test dataset, report the confusion matrix and classification metrics, and examine the Precision–Recall trade-off.

The test results will determine whether the candidate is suitable for the project's demonstration goals.