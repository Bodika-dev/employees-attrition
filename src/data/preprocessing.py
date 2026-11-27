
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from features import build_features


# Load cleaned data
df = pd.read_csv("data/processed/employee_attrition_clean.csv")

# Split into training and testing
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Attrition"]
)

# Save the split datasets
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)


# 2. Prepare selected features
X_train = build_features(train_df)
X_test = build_features(test_df)

# Separate target
y_train = train_df["Attrition"].map({"No": 0, "Yes": 1})
y_test = test_df["Attrition"].map({"No": 0, "Yes": 1})


# 3. Identify column types
numeric_cols = X_train.select_dtypes(include="number").columns
categorical_cols = X_train.select_dtypes(include="object").columns


# 4. Define preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore",
                          sparse_output=False), categorical_cols)
])


# 5. Transform data
X_train = pd.DataFrame(
    preprocessor.fit_transform(X_train),
    columns=preprocessor.get_feature_names_out()
)

X_test = pd.DataFrame(
    preprocessor.transform(X_test),
    columns=preprocessor.get_feature_names_out()
)


# 6. Save prepared data
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)

y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

joblib.dump(preprocessor, "models/preprocessor.joblib")


# 7. Check results
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
