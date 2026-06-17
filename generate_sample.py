import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 150 records
n = 150

# Core features
age = np.random.randint(22, 60, size=n)
years_exp = np.clip(age - 22 - np.random.randint(0, 5, size=n), 0, None)

# Categorical groups
genders = ["Female", "Male"]
gender = np.random.choice(genders, size=n, p=[0.48, 0.52])

departments = ["Engineering", "Sales", "Marketing", "HR"]
department = np.random.choice(departments, size=n, p=[0.4, 0.3, 0.2, 0.1])

education_levels = ["Bachelor", "Master", "PhD"]
education = np.random.choice(education_levels, size=n, p=[0.5, 0.35, 0.15])

# Design statistically correlated outputs
# 1. Salary depends on Years_Exp, Education, and Department (Engineering pays more)
base_salary = 45000
exp_bonus = years_exp * 2800
edu_bonus = np.where(education == "PhD", 25000, np.where(education == "Master", 12000, 0))
dept_bonus = np.where(department == "Engineering", 15000, np.where(department == "Sales", 5000, 0))
noise = np.random.normal(0, 4000, size=n)

salary = base_salary + exp_bonus + edu_bonus + dept_bonus + noise
salary = np.round(salary, -2).astype(int)

# 2. Performance score (1-5) correlated with Years_Exp
perf_prob_base = np.clip(0.1 + (years_exp / 40.0), 0.1, 0.9)
performance = [np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.45, 0.25, 0.15]) for _ in range(n)]

# 3. Satisfaction score (1-10) negatively correlated with age (just for fun/test)
satisfaction = np.clip(10 - (age / 10) + np.random.normal(0, 1.5, size=n), 1, 10)
satisfaction = np.round(satisfaction, 1)

# 4. Remote Worker (Yes/No) associated with Department (Engineering has high remote rate)
remote = []
for dept in department:
    if dept == "Engineering":
        remote.append(np.random.choice(["Yes", "No"], p=[0.8, 0.2]))
    elif dept == "Sales":
        remote.append(np.random.choice(["Yes", "No"], p=[0.3, 0.7]))
    elif dept == "Marketing":
        remote.append(np.random.choice(["Yes", "No"], p=[0.5, 0.5]))
    else: # HR
        remote.append(np.random.choice(["Yes", "No"], p=[0.2, 0.8]))

# Combine into DataFrame
df = pd.DataFrame({
    "Employee_ID": [f"EMP-{1000+i}" for i in range(n)],
    "Age": age,
    "Gender": gender,
    "Department": department,
    "Education": education,
    "Years_Experience": years_exp,
    "Salary": salary,
    "Satisfaction_Score": satisfaction,
    "Remote_Worker": remote
})

# Save to CSV
df.to_csv("sample_dataset.csv", index=False)
print("sample_dataset.csv created successfully with 150 rows!")
