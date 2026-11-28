# Employee Attrition Analytics & Prediction

## Project Overview

A data science and machine learning project designed to analyze employee attrition patterns and predict employee attrition risk.

The project aims to support HR teams through data analysis, machine learning predictions, and an interactive dashboard.

## Project Objectives

- Explore employee data and identify attrition patterns.
- Build and evaluate machine learning models.
- Identify employees with potentially higher attrition risk.
- Explain factors associated with model predictions.
- Present insights through an interactive HR dashboard.

## Dataset

**IBM HR Analytics Employee Attrition & Performance**

- **Records:** 1,470 employees
- **Columns:** 35
- **Target:** `Attrition` (Yes / No)
- **Data Type:** Synthetic HR dataset

## Project Progress

| Phase | Status |
|---|---|
| Project Planning | Completed |
| Data Understanding & EDA | Completed |
| Data Cleaning & Preprocessing | Completed |
| Model Development | Completed |
| Model Evaluation | Next |
| Model Explainability | Planned |
| Dashboard Data Preparation | Planned |
| Frontend Development | Planned |
| Testing & Delivery | Planned |

## Exploratory Data Analysis

The EDA explores data quality, employee characteristics, attrition patterns, feature relationships, and potential outliers.

**Related Files:**

- [EDA Notebook](notebooks/final_eda.ipynb)
- [EDA Report](reports/eda_report.md)
- [EDA Methodology](docs/eda_methodology.md)

## Data Preparation & Feature Engineering

The dataset has been cleaned and prepared for machine learning.

- Removed three constant columns.
- Selected 18 original features.
- Created one additional feature: `IsEarlyTenure`.
- Split the dataset into training (80%) and testing (20%).
- Applied categorical encoding and numerical scaling.
- Fitted preprocessing on training data only.
- Prepared 32 numerical feature columns for model development.
- Completed basic validation checks.

**Related Files:**

- [Data Preparation Documentation](docs/data_preparation.md)
- [Data Preparation Report](reports/data_preparation_report.md)

## Model Development

Three classification models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest

The models were evaluated using **5-fold stratified cross-validation** on the training dataset.

Hyperparameter tuning was performed using GridSearchCV, with F1-score as the optimization metric.

### Model Development Results

| Model | Initial F1-score | Tuned F1-score |
|---|---:|---:|
| Logistic Regression | 0.510 | **0.526** |
| Decision Tree | 0.364 | 0.406 |
| Random Forest | 0.476 | 0.494 |

**Logistic Regression** achieved the highest tuned cross-validation F1-score and was selected as the candidate model for further evaluation.

Its tuned cross-validation results were:

- **F1-score:** 0.526
- **Precision:** 0.712
- **Recall:** 0.421
- **ROC-AUC:** 0.834

The tuned model improved precision but reduced recall compared with the initial configuration. This trade-off will be reviewed during Model Evaluation.

The candidate model and its preprocessing pipeline were saved locally as `models/attrition_pipeline.joblib`.

**Note:** These are training cross-validation results, not final test performance.

**Related Files:**

- [Model Development Documentation](docs/model_development.md)
- [Model Development Report](reports/model_development_report.md)

## Project Documentation

- [Project Overview](docs/project_overview.md)
- [Project Roadmap](docs/project_roadmap.md)
- [Folder Structure](docs/folder_structure.md)

## Tools & Technologies

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook
- Git & GitHub

## Current Status

The Exploratory Data Analysis, Data Preparation, and Model Development phases are complete.

Three machine learning models have been trained, compared, and tuned using cross-validation.

Logistic Regression is currently the leading candidate based on the tuned F1-score.

The next phase is **Model Evaluation**, where the selected model will be tested on the held-out test dataset. Classification performance, prediction errors, and the precision–recall trade-off will be examined before finalizing the model.

The interactive HR dashboard will be developed in later phases.

## Dataset Limitations

The dataset is synthetic. Findings and predictions should not be treated as validated conclusions about real employees.

The planned system is intended to support HR decisions, not automate them.