import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Remove constant columns
columns_to_drop = [
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

df = df.drop(columns=columns_to_drop)

# Save cleaned dataset
df.to_csv("data/processed/employee_attrition_clean.csv", index=False)

print("Cleaned dataset shape:", df.shape)