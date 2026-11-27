import pandas as pd

# Load prepared data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Validation checks
assert X_train.columns.equals(X_test.columns)
assert not X_train.isna().any().any()
assert not X_test.isna().any().any()
assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)
assert "Attrition" not in X_train.columns
assert not any("EmployeeNumber" in col for col in X_train.columns)
assert set(y_train["Attrition"].unique()).issubset({0, 1})
assert set(y_test["Attrition"].unique()).issubset({0, 1})

print("All validation checks passed!")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
