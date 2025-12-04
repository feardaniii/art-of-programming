# Real example: Analyze your daily habits by weekday vs weekend
# Track yourself for 30 days and see the patterns!
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 📚 SCIPY: Scientific Python Library
# Why do we need it? To do REAL statistical testing!
# 
# Imagine you flip a coin 10 times and get 6 heads, 4 tails.
# Is the coin unfair? Or just random luck?
# 
# Same question here: Is your weekend productivity REALLY different?
# Or could it just be random variation?
# 
# scipy.stats helps us answer: "Is this difference REAL or just LUCK?"
from scipy import stats

# Sample data - replace with your actual tracking
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] * 5
days = days[:32]

day_type = (['Weekday']*5 + ['Weekend']*2) * 5
day_type = day_type[:32]


# Your actual metrics (32 data points for 4+ weeks)
productivity = [7, 8, 6, 7, 5, 4, 6, 8, 7, 8, 6, 5, 3, 7, 6, 8, 7, 8, 5, 4, 2, 5, 
                7, 8, 6, 7, 4, 3, 6, 8, 7, 6]

exercise_minutes = [45, 60, 30, 45, 0, 90, 120, 60, 45, 30, 60, 0, 180, 90, 45, 60, 
                   30, 45, 0, 150, 180, 60, 45, 30, 60, 45, 0, 120, 90, 75, 40, 100]

sleep_hours = [7, 6.5, 7, 6, 6.5, 9, 8.5, 7, 6.5, 7, 6, 7, 10, 9, 7, 6.5, 
               7, 6, 6.5, 9.5, 8, 7, 6.5, 7, 6, 7, 10.5, 9, 7, 6.5, 7.5, 8.5]

# Ensure all arrays are same length
assert len(days) == len(day_type) == len(productivity) == len(exercise_minutes) == len(sleep_hours), \
    "All arrays must have the same length!"

df_habits = pd.DataFrame({
    'Day': days,
    'Day_Type': day_type,
    'Productivity': productivity,
    'Exercise_Min': exercise_minutes,
    'Sleep_Hours': sleep_hours
})

# Statistical comparison visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('📊 Personal Habit Analysis: Weekday vs Weekend Patterns', 
             fontsize=16, fontweight='bold', y=0.995)

# 1. Box plots - shows median, quartiles, and outliers
sns.boxplot(data=df_habits, x='Day_Type', y='Productivity', ax=axes[0,0], palette='Set2')
axes[0,0].set_title('Productivity Distribution\n(Box shows middle 50% of data)', 
                     fontweight='bold', pad=10)
axes[0,0].set_ylabel('Productivity Score (1-10)')
axes[0,0].grid(axis='y', alpha=0.3)

# 2. Violin plots - shows full probability density
sns.violinplot(data=df_habits, x='Day_Type', y='Sleep_Hours', ax=axes[0,1], palette='Set3')
axes[0,1].set_title('Sleep Pattern Distributions\n(Width = frequency of that value)', 
                     fontweight='bold', pad=10)
axes[0,1].set_ylabel('Hours of Sleep')
axes[0,1].grid(axis='y', alpha=0.3)

# 3. Strip plot with swarm - every individual data point visible
sns.swarmplot(data=df_habits, x='Day_Type', y='Exercise_Min', ax=axes[1,0], 
              palette='husl', size=8, alpha=0.7)
axes[1,0].set_title('Exercise Minutes - Individual Days\n(Each dot = one day)', 
                     fontweight='bold', pad=10)
axes[1,0].set_ylabel('Minutes Exercised')
axes[1,0].grid(axis='y', alpha=0.3)

# 4. Bar plot with 95% confidence intervals
sns.barplot(data=df_habits, x='Day_Type', y='Productivity', ax=axes[1,1], 
            palette='muted', ci=95, capsize=0.1, errwidth=2)
axes[1,1].set_title('Average Productivity with Statistical Confidence\n(Error bars = 95% confidence interval)', 
                     fontweight='bold', pad=10)
axes[1,1].set_ylabel('Average Productivity Score')
axes[1,1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# DETAILED STATISTICAL INSIGHTS
# ============================================
print("\n" + "="*70)
print("📊 COMPREHENSIVE HABIT ANALYSIS")
print("="*70)

# Productivity Analysis
weekday_prod = df_habits[df_habits['Day_Type']=='Weekday']['Productivity']
weekend_prod = df_habits[df_habits['Day_Type']=='Weekend']['Productivity']

print("\n🎯 PRODUCTIVITY INSIGHTS:")
print(f"   Weekday average: {weekday_prod.mean():.2f} ± {weekday_prod.std():.2f}")
print(f"   Weekend average: {weekend_prod.mean():.2f} ± {weekend_prod.std():.2f}")
print(f"   📈 Difference: {weekday_prod.mean() - weekend_prod.mean():.2f} points")

# ============================================
# 🔬 STATISTICAL TESTING WITH SCIPY
# ============================================
print("\n" + "="*70)
print("🔬 STATISTICAL TEST: Is this difference REAL or just RANDOM?")
print("="*70)

# T-TEST: Compares two groups to see if they're truly different
# 
# Example: You and your friend both take a test 10 times
# You average 85, friend averages 80
# Question: Are you ACTUALLY smarter, or just got lucky?
#
# The t-test answers this by considering:
# 1. How different are the averages? (bigger difference = more likely real)
# 2. How much do scores vary? (less variation = more reliable)
# 3. How many tests taken? (more data = more reliable)

t_stat, p_value = stats.ttest_ind(weekday_prod, weekend_prod)

print(f"\n   📊 T-statistic: {t_stat:.3f}")
print(f"      (Measures HOW DIFFERENT the groups are)")
print(f"      Bigger absolute value = bigger difference")

print(f"\n   🎲 P-value: {p_value:.4f}")
print(f"      (Probability this happened by RANDOM CHANCE)")
print(f"      Lower = more confident it's a REAL difference")

print("\n   📖 WHAT DOES P-VALUE MEAN?")
if p_value < 0.001:
    print(f"      P < 0.001 → Less than 0.1% chance this is random")
    print(f"      ✅✅✅ EXTREMELY confident this is a real difference!")
elif p_value < 0.01:
    print(f"      P < 0.01 → Less than 1% chance this is random")
    print(f"      ✅✅ Very confident this is a real difference!")
elif p_value < 0.05:
    print(f"      P < 0.05 → Less than 5% chance this is random")
    print(f"      ✅ Confident this is a real difference! (Standard threshold)")
elif p_value < 0.10:
    print(f"      P < 0.10 → Less than 10% chance this is random")
    print(f"      ⚠️  Suggestive, but not quite convincing")
else:
    print(f"      P > 0.10 → More than 10% chance this is random")
    print(f"      ❌ Could easily be just luck/random variation")

print("\n   💡 ANALOGY:")
print("      Imagine flipping a coin 100 times and getting 75 heads.")
print("      P-value tells you: 'What are odds of getting 75+ heads")
print("      if the coin was actually FAIR?'")
print("      If p < 0.05 → coin is probably UNFAIR!")
print("      If p > 0.05 → could just be bad luck, coin might be fair")

# Sleep Analysis
weekday_sleep = df_habits[df_habits['Day_Type']=='Weekday']['Sleep_Hours']
weekend_sleep = df_habits[df_habits['Day_Type']=='Weekend']['Sleep_Hours']

print("\n" + "="*70)
print("😴 SLEEP PATTERN INSIGHTS:")
print("="*70)
print(f"   Weekday average: {weekday_sleep.mean():.2f} hours ± {weekday_sleep.std():.2f}")
print(f"   Weekend average: {weekend_sleep.mean():.2f} hours ± {weekend_sleep.std():.2f}")
sleep_diff = weekend_sleep.mean() - weekday_sleep.mean()
print(f"   🛌 Weekend sleep advantage: {sleep_diff:.2f} hours ({sleep_diff*60:.0f} minutes)")
print(f"   📊 You get {(sleep_diff/weekday_sleep.mean())*100:.1f}% more sleep on weekends!")

# Statistical test for sleep
t_stat_sleep, p_value_sleep = stats.ttest_ind(weekday_sleep, weekend_sleep)
print(f"\n   🔬 Sleep difference p-value: {p_value_sleep:.4f}")
if p_value_sleep < 0.05:
    print(f"   ✅ Statistically significant! You REALLY do sleep more on weekends.")
else:
    print(f"   ⚠️  Not statistically significant (could be random variation)")

# Exercise Analysis
weekday_exercise = df_habits[df_habits['Day_Type']=='Weekday']['Exercise_Min']
weekend_exercise = df_habits[df_habits['Day_Type']=='Weekend']['Exercise_Min']

print("\n" + "="*70)
print("💪 EXERCISE PATTERN INSIGHTS:")
print("="*70)
print(f"   Weekday average: {weekday_exercise.mean():.1f} minutes ± {weekday_exercise.std():.1f}")
print(f"   Weekend average: {weekend_exercise.mean():.1f} minutes ± {weekend_exercise.std():.1f}")
print(f"   🏃 Difference: {weekend_exercise.mean() - weekday_exercise.mean():.1f} minutes")

# Days with zero exercise
weekday_zeros = (weekday_exercise == 0).sum()
weekend_zeros = (weekend_exercise == 0).sum()
print(f"   ⚠️  Skipped exercise: {weekday_zeros} weekdays, {weekend_zeros} weekend days")

# Correlation Analysis
print("\n" + "="*70)
print("🔗 CORRELATION INSIGHTS:")
print("="*70)
print("   📖 Correlation measures if two things move TOGETHER")
print("      +1.0 = Perfect positive (one goes up → other goes up)")
print("      -1.0 = Perfect negative (one goes up → other goes down)")
print("       0.0 = No relationship at all")
print()

corr_sleep_prod = df_habits['Sleep_Hours'].corr(df_habits['Productivity'])
corr_exercise_prod = df_habits['Exercise_Min'].corr(df_habits['Productivity'])

print(f"   Sleep ↔ Productivity: {corr_sleep_prod:.3f}")
if abs(corr_sleep_prod) > 0.7:
    print(f"   {'✅' if corr_sleep_prod > 0 else '⚠️ '} STRONG relationship!")
elif abs(corr_sleep_prod) > 0.5:
    print(f"   {'✅' if corr_sleep_prod > 0 else '⚠️ '} MODERATE relationship")
elif abs(corr_sleep_prod) > 0.3:
    print(f"   ℹ️  WEAK relationship")
else:
    print(f"   ❌ No clear relationship")
    
print(f"\n   Exercise ↔ Productivity: {corr_exercise_prod:.3f}")
if abs(corr_exercise_prod) > 0.7:
    print(f"   {'✅' if corr_exercise_prod > 0 else '⚠️ '} STRONG relationship!")
elif abs(corr_exercise_prod) > 0.5:
    print(f"   {'✅' if corr_exercise_prod > 0 else '⚠️ '} MODERATE relationship")
elif abs(corr_exercise_prod) > 0.3:
    print(f"   ℹ️  WEAK relationship")
else:
    print(f"   ❌ No clear relationship")

# Actionable Recommendations
print("\n" + "="*70)
print("💡 ACTIONABLE RECOMMENDATIONS:")
print("="*70)

if weekday_sleep.mean() < 7:
    print("   1. 🛌 Try to get at least 7 hours on weekdays (current: {:.1f}h)".format(weekday_sleep.mean()))
    
if weekend_prod.mean() < weekday_prod.mean() - 1:
    print("   2. 🎯 Weekend productivity drops significantly - consider if this is intentional rest")
    
if weekday_zeros > 2:
    print("   3. 💪 You skipped exercise {} weekdays - aim for consistency".format(weekday_zeros))

if corr_sleep_prod > 0.5:
    print("   4. 😴 Sleep strongly affects your productivity - prioritize rest!")

if corr_exercise_prod > 0.5:
    print("   5. 🏃 Exercise boosts your productivity - make it a priority!")

print("\n" + "="*70)
print("📈 Keep tracking to see trends over time!")
print("="*70 + "\n")

# ============================================
# 📚 SCIPY SUMMARY
# ============================================
print("="*70)
print("📚 WHY WE USE SCIPY:")
print("="*70)
print("""
scipy.stats gives us SCIENTIFIC TOOLS to answer:
   
   ❓ "Is this difference REAL or just LUCK?"
   ❓ "Can I trust this pattern?"
   ❓ "Should I change my behavior based on this data?"

WITHOUT scipy → Just guessing based on averages
WITH scipy    → Mathematical confidence in your conclusions

Key functions we used:
   • stats.ttest_ind() - Compare two groups
   • Pandas .corr()    - Find relationships between variables
   
Other useful scipy.stats functions:
   • stats.pearsonr()  - Correlation with p-value
   • stats.spearmanr() - Correlation for non-linear relationships
   • stats.chi2_contingency() - For categorical data
   • stats.mannwhitneyu() - Non-parametric alternative to t-test
""")
print("="*70 + "\n")