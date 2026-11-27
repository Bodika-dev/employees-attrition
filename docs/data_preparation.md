# Data Preparation & Feature Engineering

## 1. Objective

Prepare the employee attrition dataset for machine learning by cleaning the data, selecting relevant features, and applying consistent preprocessing.

## 2. Data Cleaning

The dataset was checked for missing values, duplicate records, and data quality issues during EDA.

Three constant columns were removed:

- `EmployeeCount`
- `Over18`
- `StandardHours`

The original dataset was preserved in `data/raw/`.

## 3. Feature Selection

We selected 18 original features covering:

- Job characteristics and working conditions
- Employee satisfaction and involvement
- Salary and job level
- Work experience and company tenure

The target column, `Attrition`, and the employee identifier, `EmployeeNumber`, were excluded from model inputs.

The selected features are an initial modeling choice and will be evaluated during model development.

## 4. Feature Engineering

One additional feature was created:

**IsEarlyTenure:** Indicates whether an employee has worked at the company for two years or less.

This produces 19 features before encoding.

## 5. Train/Test Split

The dataset was divided into:

- Training data: 80% (1,176 employees)
- Testing data: 20% (294 employees)

Stratified splitting was used to preserve the target distribution.

## 6. Preprocessing

The same preprocessing rules are applied to training and testing data.

- **StandardScaler:** Scales numerical features.
- **OneHotEncoder:** Converts categorical features into numerical columns.
- **Target Encoding:** Converts `Attrition` into 0 (No) and 1 (Yes).

The preprocessor is fitted on training data only and then applied to testing data to prevent data leakage.

## 7. Output Files

The prepared datasets are stored in `data/processed/`.

- `X_train.csv` — Training features
- `X_test.csv` — Testing features
- `y_train.csv` — Training labels
- `y_test.csv` — Testing labels

The fitted preprocessor is saved as `models/preprocessor.joblib`.

Both feature datasets contain 32 columns after encoding.

## 8. Validation

Validation checks cover:

- Matching training and testing feature columns
- Missing values
- Feature and label row counts
- Target and employee ID exclusion
- Binary target values

## 9. Next Step

Train and compare three machine learning models using the same prepared features.

Model performance will be used to assess the effectiveness of the selected feature set.