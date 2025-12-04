import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Your personal health data example
np.random.seed(42)
sleep_hours = np.random.normal(7.5, 1.2, 1000)
energy_levels = 2 * sleep_hours + np.random.normal(0, 2, 1000)

# Create comprehensive distribution analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Histogram with KDE
sns.histplot(sleep_hours, kde=True, ax=axes[0,0], color='skyblue', alpha=0.7)
axes[0,0].set_title('Sleep Distribution with Density Curve', fontweight='bold')
axes[0,0].set_xlabel('Hours of Sleep')

# 2. Box plot to identify outliers
sns.boxplot(y=sleep_hours, ax=axes[0,1], color='lightcoral')
axes[0,1].set_title('Sleep Quality: Outliers & Quartiles', fontweight='bold')

# 3. Violin plot: distribution shape + statistics
sns.violinplot(y=sleep_hours, ax=axes[1,0], color='lightgreen')
axes[1,0].set_title('Sleep Distribution Shape', fontweight='bold')

# Script here - no dataframes yet - just 2 NumpY arrays
# 4. Distribution comparison
data = pd.DataFrame({'Sleep': sleep_hours, 'Energy': energy_levels})
sns.scatterplot(data=data, x='Sleep', y='Energy', ax=axes[1,1], alpha=0.6)
axes[1,1].set_title('Sleep vs Energy Relationship', fontweight='bold')

plt.tight_layout()
plt.show()

# Quick insights
print(f"Your average sleep: {sleep_hours.mean():.1f} hours")
print(f"Sleep consistency: {sleep_hours.std():.1f} hour standard deviation")
print(f"Energy correlation: {np.corrcoef(sleep_hours, energy_levels)[0,1]:.2f}")

