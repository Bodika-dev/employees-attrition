# Selected features for all three models
FEATURE_COLUMNS = [
    "BusinessTravel",
    "Department",
    "DistanceFromHome",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "OverTime",
    "StockOptionLevel",
    "TotalWorkingYears",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]


def build_features(df):
    X = df[FEATURE_COLUMNS].copy()

    # New feature: employee has worked 2 years or less
    X["IsEarlyTenure"] = (X["YearsAtCompany"] <= 2).astype(int)

    return X
