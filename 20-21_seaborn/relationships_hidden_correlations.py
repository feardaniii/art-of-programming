# Personal finance correlation analysis
# Real example: Find which expenses correlate with life satisfaction

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sample data - track this yourself!
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
entertainment = [400, 350, 500, 280, 450, 320]
food_delivery = [180, 220, 160, 240, 190, 280]
gym_membership = [80, 80, 80, 0, 80, 80]  # Notice the gap!
life_satisfaction = [7, 6, 8, 4, 7, 5]   # Scale 1-10

df = pd.DataFrame({
    'Month': months,
    'Entertainment': entertainment,
    'Food_Delivery': food_delivery,
    'Gym': gym_membership,
    'Satisfaction': life_satisfaction
})

# Create relationship exploration
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Scatter plot with regression line
sns.regplot(data=df, x='Entertainment', y='Satisfaction', ax=axes[0,0])
axes[0,0].set_title('Entertainment Spending vs Life Satisfaction', fontweight='bold')

# 2. Multiple variables relationship
sns.scatterplot(data=df, x='Food_Delivery', y='Satisfaction', 
                size='Entertainment', sizes=(50, 200), ax=axes[0,1])
axes[0,1].set_title('Food + Entertainment vs Satisfaction', fontweight='bold')

# 3. Correlation heatmap
corr_data = df[['Entertainment', 'Food_Delivery', 'Gym', 'Satisfaction']].corr()
sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
axes[1,0].set_title('Expense Correlation Matrix', fontweight='bold')

# 4. Pairwise relationships
df_numeric = df[['Entertainment', 'Food_Delivery', 'Gym', 'Satisfaction']]
sns.heatmap(df_numeric.corr(), annot=True, cmap='viridis', ax=axes[1,1])
axes[1,1].set_title('All Relationships at Once', fontweight='bold')

plt.tight_layout()
plt.show()

# Actionable insights
print("💡 INSIGHTS:")
print(f"Entertainment-Satisfaction correlation: {df['Entertainment'].corr(df['Satisfaction']):.2f}")
print(f"Gym-Satisfaction correlation: {df['Gym'].corr(df['Satisfaction']):.2f}")
print(f"Food Delivery-Satisfaction correlation: {df['Food_Delivery'].corr(df['Satisfaction']):.2f}")



# Key relationships:
# 💡 Strong Positive Correlations

# Entertainment ↔ Satisfaction: 0.96
# People who spend more on entertainment have much higher satisfaction.

# Gym ↔ Satisfaction: 0.72
# Fitness spending is also associated with higher life satisfaction.

# Gym ↔ Entertainment: 0.61
# People who spend more on entertainment also tend to spend more on the gym.

# 💡 Strong Negative Correlations

# Food_Delivery ↔ Entertainment: −0.85
# Higher entertainment spending tends to occur with lower food-delivery spending.

# Food_Delivery ↔ Satisfaction: −0.87
# More spending on food delivery is associated with lower life satisfaction.

# Food_Delivery ↔ Gym: −0.32
# Mild negative relationship—people ordering food delivery slightly less likely to spend on gym.