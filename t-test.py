import pandas as pd
from scipy import stats
import numpy as np

# --- LOAD DATA ---
df = pd.read_csv("cloudcover_daily.csv")

# --- CLEAN DATA ---
df = df.dropna(subset=["cloudcover", "label"])  # Remove missing values if any

# --- SPLIT GROUPS ---
weekdays = df[df["label"] == "Weekday"]["cloudcover"]
weekends = df[df["label"] == "Weekend"]["cloudcover"]

# --- DESCRIPTIVE STATS ---
def describe(group, name):
    n = len(group)
    mean = group.mean()
    std = group.std()
    se = std / np.sqrt(n)
    moe = 1.96 * se  # 95% CI
    print(f"{name}:")
    print(f"  n = {n}")
    print(f"  Mean cloud cover = {mean:.2f}% ± {moe:.2f} (95% CI)")
    print(f"  Std Dev = {std:.2f}\n")
    return mean, std, n

print("\n=== Cloud Cover Comparison: Weekdays vs Weekends ===\n")

mean_weekdays, std_weekdays, n_weekdays = describe(weekdays, "Weekdays")
mean_weekends, std_weekends, n_weekends = describe(weekends, "Weekends")

# --- TWO-SAMPLE T-TEST (Welch’s t-test) ---
t_stat, p_val = stats.ttest_ind(weekdays, weekends, equal_var=False)

print("=== Two-Sample t-Test Results ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

if p_val < 0.05:
    print("YES. Statistically significant difference in average cloud cover between weekdays and weekends (p < 0.05)")
else:
    print("NO. No statistically significant difference in average cloud cover (p ≥ 0.05)")

