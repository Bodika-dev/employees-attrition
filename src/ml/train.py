
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data.features import build_features


# 1. Load training data
df = pd.read_csv("data/processed/train.csv")

X = build_features(df)
y = df["Attrition"].map({"No": 0, "Yes": 1})


# 2. Prepare preprocessing
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     categorical_cols)
])


# 3. Define the three models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )
}


# 4. Create 5 folds
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 5. Compare models
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "f1": "f1",
            "recall": "recall",
            "precision": "precision",
            "roc_auc": "roc_auc"
        }
    )

    print(f"\n{name}")
    print(f"F1-score : {scores['test_f1'].mean():.3f}")
    print(f"Recall   : {scores['test_recall'].mean():.3f}")
    print(f"Precision: {scores['test_precision'].mean():.3f}")
    print(f"ROC-AUC  : {scores['test_roc_auc'].mean():.3f}")
    
    
# 6. Tune Logistic Regression
logistic_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__class_weight": [None, "balanced"]
}

grid_search = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

grid_search.fit(X, y)

print("\nLogistic Regression Tuning Results")
print("Best Parameters:", grid_search.best_params_)
print(f"Best F1-score: {grid_search.best_score_:.3f}")


# 7. Tune Decision Tree and Random Forest
tuning_models = {
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "model__max_depth": [3, 5, 8, None],
            "model__min_samples_leaf": [1, 5],
            "model__class_weight": [None, "balanced"]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "model__n_estimators": [100],
            "model__max_depth": [6, None],
            "model__min_samples_leaf": [1, 5],
            "model__class_weight": [None, "balanced"]
        }
    )
}

for name, (model, params) in tuning_models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring="f1",
        cv=cv,
        n_jobs=-1
    )

    search.fit(X, y)

    print(f"\n{name} Tuning Results")
    print("Best Parameters:", search.best_params_)
    print(f"Best F1-score: {search.best_score_:.3f}")


# 8. Check tuned Logistic Regression
best_model = grid_search.best_estimator_

scores = cross_validate(
    best_model,
    X,
    y,
    cv=cv,
    scoring={
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "roc_auc": "roc_auc"
    }
)

print("\nTuned Logistic Regression")
for metric in ["f1", "recall", "precision", "roc_auc"]:
    print(f"{metric}: {scores[f'test_{metric}'].mean():.3f}")

# 9. Save the candidate model with its preprocessing
Path("models").mkdir(exist_ok=True)
joblib.dump(best_model, "models/attrition_pipeline.joblib")

print("\nCandidate model saved successfully!")
