import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Load data
df = pd.read_csv("cloudcover_daily.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.dropna(subset=["cloudcover", "label"])

# Split groups
weekdays = df[df["label"] == "Weekday"]["cloudcover"]
weekends = df[df["label"] == "Weekend"]["cloudcover"]

# Calculate statistics
mean_weekdays = weekdays.mean()
mean_weekends = weekends.mean()
std_weekdays = weekdays.std()
std_weekends = weekends.std()
se_weekdays = std_weekdays / np.sqrt(len(weekdays))
se_weekends = std_weekends / np.sqrt(len(weekends))
t_stat, p_val = stats.ttest_ind(weekdays, weekends, equal_var=False)

# Create figure with multiple panels
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# ============================================================
# Panel A: Distribution comparison (violin + box plots)
# ============================================================
ax1 = fig.add_subplot(gs[0, :])
data_combined = pd.DataFrame({
    'Cloud Cover (%)': pd.concat([weekdays, weekends]),
    'Day Type': ['Weekday']*len(weekdays) + ['Weekend']*len(weekends)
})

parts = ax1.violinplot([weekdays, weekends], positions=[0, 1], 
                        showmeans=True, showmedians=True, widths=0.7)
for pc in parts['bodies']:
    pc.set_facecolor('#8dd3c7')
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(1)

bp = ax1.boxplot([weekdays, weekends], positions=[0, 1], widths=0.3,
                  patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
                  medianprops=dict(color='red', linewidth=2),
                  whiskerprops=dict(color='black', linewidth=1.5),
                  capprops=dict(color='black', linewidth=1.5))

ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Weekday', 'Weekend'])
ax1.set_ylabel('Cloud Cover (%)', fontweight='bold')
ax1.set_title('A. Distribution of Cloud Cover: Weekdays vs Weekends', 
              fontweight='bold', loc='left', pad=15)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 105)

# Add sample size annotations
ax1.text(0, 102, f'n = {len(weekdays)}', ha='center', fontsize=9, fontweight='bold')
ax1.text(1, 102, f'n = {len(weekends)}', ha='center', fontsize=9, fontweight='bold')

# ============================================================
# Panel B: Time series with monthly moving average
# ============================================================
ax2 = fig.add_subplot(gs[1, :])
df_sorted = df.sort_values('date')
df_sorted['year'] = df_sorted['date'].dt.year

# Plot raw data with transparency
for label, color in [('Weekday', '#8dd3c7'), ('Weekend', '#fb8072')]:
    mask = df_sorted['label'] == label
    ax2.scatter(df_sorted[mask]['date'], df_sorted[mask]['cloudcover'], 
               alpha=0.15, s=5, color=color, label=label)

# Add 30-day rolling mean for each group
for label, color in [('Weekday', '#117864'), ('Weekend', '#cb4335')]:
    mask = df_sorted['label'] == label
    subset = df_sorted[mask].set_index('date')['cloudcover']
    rolling = subset.rolling(window=30, min_periods=15).mean()
    ax2.plot(rolling.index, rolling.values, color=color, linewidth=2, 
            label=f'{label} (30-day avg)', alpha=0.8)

ax2.set_xlabel('Date', fontweight='bold')
ax2.set_ylabel('Cloud Cover (%)', fontweight='bold')
ax2.set_title('B. Cloud Cover Time Series (1990-2025)', 
              fontweight='bold', loc='left', pad=15)
ax2.legend(loc='upper right', framealpha=0.9)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 105)

# ============================================================
# Panel C: Bar chart with error bars
# ============================================================
ax3 = fig.add_subplot(gs[2, 0])
x_pos = [0, 1]
means = [mean_weekdays, mean_weekends]
sems = [se_weekdays, se_weekends]
colors = ['#8dd3c7', '#fb8072']

bars = ax3.bar(x_pos, means, yerr=[1.96*s for s in sems], 
               capsize=8, color=colors, edgecolor='black', linewidth=1.5,
               error_kw={'linewidth': 2, 'ecolor': 'black'})

ax3.set_xticks(x_pos)
ax3.set_xticklabels(['Weekday', 'Weekend'])
ax3.set_ylabel('Mean Cloud Cover (%) ± 95% CI', fontweight='bold')
ax3.set_title('C. Mean Cloud Cover Comparison', fontweight='bold', loc='left', pad=15)
ax3.set_ylim(0, 70)
ax3.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (m, s) in enumerate(zip(means, sems)):
    ax3.text(i, m + 1.96*s + 2, f'{m:.2f}%\n±{1.96*s:.2f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================
# Panel D: Statistical summary
# ============================================================
ax4 = fig.add_subplot(gs[2, 1])
ax4.axis('off')

summary_text = f"""
STATISTICAL SUMMARY
{'='*50}

Weekdays (n = {len(weekdays):,}):
  Mean ± SD: {mean_weekdays:.2f}% ± {std_weekdays:.2f}%
  95% CI: [{mean_weekdays - 1.96*se_weekdays:.2f}, {mean_weekdays + 1.96*se_weekdays:.2f}]
  Median: {weekdays.median():.2f}%
  
Weekends (n = {len(weekends):,}):
  Mean ± SD: {mean_weekends:.2f}% ± {std_weekends:.2f}%
  95% CI: [{mean_weekends - 1.96*se_weekends:.2f}, {mean_weekends + 1.96*se_weekends:.2f}]
  Median: {weekends.median():.2f}%

{'='*50}
WELCH'S TWO-SAMPLE T-TEST
{'='*50}

Difference in means: {mean_weekdays - mean_weekends:.2f}%
T-statistic: {t_stat:.4f}
P-value: {p_val:.4f}
Degrees of freedom: {len(weekdays) + len(weekends) - 2:,}

Result: {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)

Interpretation:
{'There IS sufficient evidence to reject the null' if p_val < 0.05 else 'There is NO sufficient evidence to reject the null'}
hypothesis of equal mean cloud cover between
weekdays and weekends.

Effect size (Cohen\'s d): {(mean_weekdays - mean_weekends) / np.sqrt((std_weekdays**2 + std_weekends**2) / 2):.4f}
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontfamily='monospace', fontsize=8.5, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ============================================================
# Overall title
# ============================================================
fig.suptitle('Cloud Cover Analysis: Weekdays vs Weekends (New York, NY)\n1990-2025',
             fontsize=16, fontweight='bold', y=0.995)

# Save figure
plt.savefig('cloud_cover_analysis.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("\n✅ Visualization saved as 'cloud_cover_analysis.png'")

plt.show()
