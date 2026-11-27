# Data Preparation & Feature Engineering Report

**Project:** Employee Attrition Analytics & Prediction  
**Phase:** 04 — Data Preparation & Feature Engineering

## 1. Objective

Prepare a consistent feature set for training and evaluating three machine learning models while keeping the original dataset unchanged.

## 2. Cleaning Results

| Check or Action | Result |
|---|---|
| Original Dataset | 1,470 rows × 35 columns |
| Missing Values | 0 |
| Duplicate Records | 0 |
| Removed Constant Columns | `EmployeeCount`, `Over18`, `StandardHours` |
| Cleaned Dataset | 1,470 rows × 32 columns |

No employee records were removed. Potential outliers were kept because they were not confirmed as data errors.

## 3. Feature Preparation

- Selected **18 original features** covering working conditions, satisfaction, income, job characteristics, and experience.
- Created **`IsEarlyTenure`**, indicating employees with two years or less at the company.
- Used `Attrition` as the target (`No = 0`, `Yes = 1`).
- Excluded `EmployeeNumber` from model inputs.
- Applied OneHotEncoder to categorical features.
- Applied StandardScaler to numerical features.
- Fitted preprocessing on training data only.

**Final result:** 19 selected and engineered features became **32 numerical columns** after encoding.

## 4. Prepared Datasets

An 80/20 stratified train/test split was used.

| Dataset | Shape |
|---|---|
| `X_train.csv` | 1,176 × 32 |
| `X_test.csv` | 294 × 32 |
| `y_train.csv` | 1,176 × 1 |
| `y_test.csv` | 294 × 1 |

The fitted preprocessor was saved as `models/preprocessor.joblib`.

## 5. Validation Results

The prepared datasets passed the basic validation checks:

- Training and testing feature columns match.
- No missing values were found.
- Feature and label row counts match.
- Target values are binary.
- `Attrition` and `EmployeeNumber` are excluded from model features.

## 6. Decisions and Limitations

- The selected features are a starting point, not a proven optimal selection.
- Potential outliers were not removed automatically.
- The same prepared features will be used for all three models.
- Model performance has not yet been evaluated.
- The dataset is synthetic and does not establish reliable predictions for real employees.

## 7. Next Phase

Train and compare three machine learning models using the same prepared data.

Evaluate their performance using Accuracy, Precision, Recall, F1-score, and ROC-AUC before selecting the final model.

## Related Documentation

- [Data Preparation Documentation](../docs/data_preparation.md)
- [EDA Report](eda_report.md)