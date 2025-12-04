# 🎯 MULTI-PANEL STATISTICAL NARRATIVES
# Learn how to tell a complete data story across multiple coordinated visualizations
# This is how professionals analyze complex datasets!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta

# ============================================
# 📚 WHAT IS A STATISTICAL NARRATIVE?
# ============================================
print("="*70)
print("📖 MULTI-PANEL STATISTICAL NARRATIVES")
print("="*70)
print("""
A statistical narrative is like telling a story with data using multiple
coordinated visualizations. Each panel answers a different question:

Panel 1 (Big Picture):  "What's the main relationship?"
Panel 2 (Correlations): "What's connected to what?"
Panel 3 (Distributions): "What's normal vs unusual?"
Panel 4 (Time/Patterns): "Are there patterns over time?"
Panel 5-8 (Deep Dives): "What specific insights matter?"

Think of it like a movie:
- Single chart = One photo
- Multi-panel dashboard = Complete film with scenes, angles, close-ups

Let's build a complete narrative about life optimization!
""")
print("="*70 + "\n")

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

# ============================================
# 📊 GENERATE REALISTIC LIFE DATA
# ============================================
print("🔧 Generating 90 days of realistic personal data...")

np.random.seed(42)  # Reproducible results
n_days = 90

# Create realistic date range
start_date = datetime(2024, 8, 1)
dates = [start_date + timedelta(days=i) for i in range(n_days)]
day_names = [d.strftime('%A') for d in dates]
is_weekend = [d.weekday() >= 5 for d in dates]  # Sat=5, Sun=6

# ============================================
# 💤 SLEEP: Realistic patterns
# - Better on weekends
# - Natural variation
# - Occasional bad nights
# ============================================
base_sleep = np.where(is_weekend, 8.5, 7.0)
sleep_hours = base_sleep + np.random.normal(0, 0.8, n_days)
# Add occasional insomnia nights (5% chance)
insomnia_nights = np.random.random(n_days) < 0.05
sleep_hours = np.where(insomnia_nights, np.random.uniform(4, 5.5, n_days), sleep_hours)
sleep_hours = np.clip(sleep_hours, 4, 11)

# ============================================
# 💪 EXERCISE: Realistic patterns
# - More motivation on weekends
# - Many skip days (realistic!)
# - Occasional intense workouts
# ============================================
weekend_exercise = np.random.gamma(3, 20, n_days)
weekday_exercise = np.random.gamma(2, 15, n_days)
exercise_minutes = np.where(is_weekend, weekend_exercise, weekday_exercise)
# 35% chance of skipping (very realistic!)
skip_exercise = np.random.random(n_days) < 0.35
exercise_minutes = np.where(skip_exercise, 0, exercise_minutes)
exercise_minutes = np.clip(exercise_minutes, 0, 180)

# ============================================
# 😰 STRESS: Work patterns
# - Higher during weekdays
# - Occasional crisis days
# - Weekend relief
# ============================================
base_stress = np.where(is_weekend, 3.5, 6.5)
work_stress = base_stress + np.random.normal(0, 1.5, n_days)
# Add occasional high-stress events (10% of weekdays)
crisis_days = np.random.random(n_days) < 0.10
work_stress = np.where(crisis_days & ~np.array(is_weekend), 
                       np.random.uniform(8, 10, n_days), 
                       work_stress)
work_stress = np.clip(work_stress, 1, 10)

# ============================================
# 👥 SOCIAL TIME: Natural patterns
# - More on weekends
# - Occasional social weeknight
# - Some lonely days
# ============================================
base_social = np.where(is_weekend, 4, 1.5)
social_hours = base_social + np.random.exponential(1, n_days)
social_hours = np.clip(social_hours, 0, 12)

# ============================================
# 📱 SCREEN TIME: Modern reality
# - Consistent across days
# - Slightly more on weekends
# - Correlation with low activity
# ============================================
base_screen = np.where(is_weekend, 5, 4)
phone_hours = base_screen + np.random.normal(0, 1.2, n_days)
# More screen time on no-exercise days
phone_hours = np.where(exercise_minutes == 0, 
                       phone_hours + np.random.uniform(0.5, 2, n_days),
                       phone_hours)
phone_hours = np.clip(phone_hours, 1, 10)

# ============================================
# ☕ COFFEE: Energy management
# - More on tired days
# - Habit-forming patterns
# ============================================
coffee_cups = np.where(sleep_hours < 6.5, 
                       np.random.poisson(3, n_days),
                       np.random.poisson(2, n_days))
coffee_cups = np.clip(coffee_cups, 0, 6)

# ============================================
# 🍔 EATING OUT: Lifestyle indicator
# - More on weekends
# - Correlates with social time
# ============================================
meals_out = np.where(is_weekend,
                     np.random.poisson(2, n_days),
                     np.random.poisson(0.5, n_days))
meals_out = np.clip(meals_out, 0, 5)

# ============================================
# 😊 LIFE SATISFACTION: The outcome variable!
# This is what we're trying to optimize!
# Complex formula creates REAL correlations
# ============================================
life_satisfaction = (
    0.35 * (sleep_hours - 5) +                    # Sleep is crucial
    0.25 * (exercise_minutes / 20) +              # Exercise helps
    0.30 * (10 - work_stress) +                   # Stress kills happiness
    0.15 * social_hours +                          # Social connection matters
    -0.20 * phone_hours +                          # Too much screen = unhappy
    -0.10 * np.abs(coffee_cups - 2) +             # Extreme coffee = stress
    0.10 * meals_out +                             # Treats boost mood
    np.random.normal(0, 1.2, n_days)              # Random life events
)
# Normalize to 1-10 scale
life_satisfaction = 5 + (life_satisfaction / life_satisfaction.std()) * 2
life_satisfaction = np.clip(life_satisfaction, 1, 10)

# Create comprehensive DataFrame
life_df = pd.DataFrame({
    'Date': dates,
    'Day': day_names,
    'Weekend': is_weekend,
    'Sleep_Hours': sleep_hours,
    'Exercise_Min': exercise_minutes,
    'Stress_Level': work_stress,
    'Social_Hours': social_hours,
    'Screen_Hours': phone_hours,
    'Coffee_Cups': coffee_cups,
    'Meals_Out': meals_out,
    'Life_Satisfaction': life_satisfaction,
    'Day_Number': range(1, n_days + 1)
})

print(f"✅ Generated {n_days} days of data tracking {len(life_df.columns)-2} metrics")
print(f"📅 {dates[0].strftime('%b %d')} to {dates[-1].strftime('%b %d, %Y')}\n")

# ----------------- Data Generation 

# ============================================
# 🎨 CREATE MULTI-PANEL STATISTICAL NARRATIVE
# ============================================
print("🎨 Building your statistical narrative dashboard...\n")

# Create figure with GridSpec for flexible layout
fig = plt.figure(figsize=(22, 16))
fig.suptitle('📊 LIFE OPTIMIZATION: A Complete Statistical Narrative\n' +
             'Telling the story of what makes you happy over 90 days', 
             fontsize=20, fontweight='bold', y=0.995)

# GridSpec allows different sized panels
gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)

# ============================================
# PANEL 1: THE HERO SHOT (Main Relationship)
# "What's the primary story?"
# ============================================
ax_main = fig.add_subplot(gs[0:2, 0:2])  # 2x2 grid = large panel

# Multi-dimensional scatter plot
# X-axis: Sleep (primary factor)
# Y-axis: Satisfaction (outcome)
# Size: Exercise (secondary factor)
# Color: Stress (context)
scatter = ax_main.scatter(
    life_df['Sleep_Hours'], 
    life_df['Life_Satisfaction'],
    s=life_df['Exercise_Min'] * 2,     # Bigger dots = more exercise
    c=life_df['Stress_Level'],          # Red = stressed, Green = relaxed
    cmap='RdYlGn_r',                    # Red-Yellow-Green reversed
    alpha=0.6,
    edgecolors='black',
    linewidth=0.5
)

ax_main.set_xlabel('Sleep Hours', fontsize=13, fontweight='bold')
ax_main.set_ylabel('Life Satisfaction (1-10)', fontsize=13, fontweight='bold')
ax_main.set_title('💡 THE BIG PICTURE: Life Satisfaction Optimization\n' +
                  '(Size = Exercise, Color = Stress Level)', 
                  fontsize=14, fontweight='bold', pad=15)
ax_main.grid(True, alpha=0.3, linestyle='--')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax_main)
cbar.set_label('Stress Level (1-10)', fontsize=11, fontweight='bold')

# Add trend line with equation
z = np.polyfit(life_df['Sleep_Hours'], life_df['Life_Satisfaction'], 1)
p = np.poly1d(z)
sleep_sorted = np.sort(life_df['Sleep_Hours'])
ax_main.plot(sleep_sorted, p(sleep_sorted),
            "r--", alpha=0.8, linewidth=2.5, 
            label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
ax_main.legend(fontsize=11, loc='lower right')

# Add text annotation
corr = life_df['Sleep_Hours'].corr(life_df['Life_Satisfaction'])
ax_main.text(0.02, 0.98, f'Correlation: {corr:.3f}',
            transform=ax_main.transAxes,
            fontsize=12, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ============================================
# PANEL 2: THE NETWORK (Correlation Matrix)
# "How is everything connected?"
# ============================================
ax_corr = fig.add_subplot(gs[0, 2:])  # Top right

# Select key metrics
corr_cols = ['Sleep_Hours', 'Exercise_Min', 'Stress_Level', 
             'Social_Hours', 'Screen_Hours', 'Coffee_Cups', 
             'Life_Satisfaction']
corr_matrix = life_df[corr_cols].corr()

# Create heatmap
sns.heatmap(corr_matrix, 
            annot=True,           # Show numbers
            fmt='.2f',            # 2 decimal places
            cmap='RdYlBu_r',      # Red-Yellow-Blue reversed
            center=0,              # White at zero
            vmin=-1, vmax=1,      # Correlation range
            ax=ax_corr,
            linewidths=1,
            linecolor='white',
            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8})

ax_corr.set_title('🔗 THE NETWORK: How Everything Connects\n' +
                  '(Dark red = strong positive, Dark blue = strong negative)',
                  fontsize=14, fontweight='bold', pad=15)

# Rotate labels
ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, ha='right')
ax_corr.set_yticklabels(ax_corr.get_yticklabels(), rotation=0)

# ============================================
# PANEL 3: THE DISTRIBUTIONS (What's Normal?)
# "What does typical look like?"
# ============================================
ax_dist = fig.add_subplot(gs[1, 2:])  # Second row right

# Prepare data for violin plot
metrics_subset = life_df[['Sleep_Hours', 'Exercise_Min', 'Stress_Level', 'Life_Satisfaction']].copy()
metrics_subset.columns = ['Sleep\n(hours)', 'Exercise\n(min)', 'Stress\n(1-10)', 'Satisfaction\n(1-10)']

# Melt for seaborn
metrics_melted = metrics_subset.melt(var_name='Metric', value_name='Value')

# Create violin plots
sns.violinplot(data=metrics_melted, x='Metric', y='Value', 
               ax=ax_dist, palette='Set2', inner='box')

ax_dist.set_title('📊 THE DISTRIBUTIONS: What\'s Normal vs Unusual?\n' +
                  '(Width shows frequency - wider = more common)',
                  fontsize=14, fontweight='bold', pad=15)
ax_dist.set_xlabel('')
ax_dist.set_ylabel('Value', fontsize=11, fontweight='bold')
ax_dist.grid(axis='y', alpha=0.3, linestyle='--')

# Add mean lines
for i, col in enumerate(metrics_subset.columns):
    mean_val = metrics_subset[col].mean()
    ax_dist.hlines(mean_val, i-0.4, i+0.4, colors='red', linewidth=2, linestyles='--')

# ============================================
# PANEL 4: THE PATTERN (Weekly Rhythm)
# "Are there repeating patterns?"
# ============================================
ax_weekly = fig.add_subplot(gs[2, :2])  # Third row left (wide)

# Weekly pattern
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.boxplot(data=life_df, x='Day', y='Life_Satisfaction',
           order=day_order, ax=ax_weekly, 
           palette=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
                   '#98D8C8', '#C7F0DB', '#E8F8F5'])

# Add overall mean line
overall_mean = life_df['Life_Satisfaction'].mean()
ax_weekly.axhline(y=overall_mean, color='red', linestyle='--', 
                 linewidth=2.5, label=f'Overall Average: {overall_mean:.2f}')

ax_weekly.set_title('📅 THE WEEKLY PATTERN: Do Certain Days Make You Happier?\n' +
                    '(Box shows middle 50%, line shows median)',
                    fontsize=14, fontweight='bold', pad=15)
ax_weekly.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
ax_weekly.set_ylabel('Life Satisfaction (1-10)', fontsize=11, fontweight='bold')
ax_weekly.legend(fontsize=10)
ax_weekly.grid(axis='y', alpha=0.3, linestyle='--')
ax_weekly.tick_params(axis='x', rotation=45)

# ============================================
# PANEL 5: THE TRAJECTORY (Time Trend)
# "Are things getting better or worse?"
# ============================================
ax_trend = fig.add_subplot(gs[2, 2:])  # Third row right (wide)

# Plot daily values
ax_trend.plot(life_df['Day_Number'], life_df['Life_Satisfaction'],
             'o-', alpha=0.3, color='steelblue', markersize=4, 
             linewidth=1, label='Daily satisfaction')

# Add 7-day moving average (smoothed trend)
life_df['MA7'] = life_df['Life_Satisfaction'].rolling(window=7, center=True).mean()
ax_trend.plot(life_df['Day_Number'], life_df['MA7'],
             color='darkblue', linewidth=3, label='7-day moving average')

# Add overall trend line
z_trend = np.polyfit(life_df['Day_Number'], life_df['Life_Satisfaction'], 1)
p_trend = np.poly1d(z_trend)
trend_direction = "↑ Improving" if z_trend[0] > 0 else "↓ Declining" if z_trend[0] < 0 else "→ Stable"
ax_trend.plot(life_df['Day_Number'], p_trend(life_df['Day_Number']),
             'g--', linewidth=2.5, 
             label=f'Overall trend: {trend_direction} ({z_trend[0]:+.4f}/day)')

ax_trend.set_title('📈 THE TRAJECTORY: Are You Getting Happier?\n' +
                   '(Blue dots = daily, Dark line = trend, Green = direction)',
                   fontsize=14, fontweight='bold', pad=15)
ax_trend.set_xlabel('Day Number (1-90)', fontsize=11, fontweight='bold')
ax_trend.set_ylabel('Life Satisfaction (1-10)', fontsize=11, fontweight='bold')
ax_trend.legend(fontsize=9, loc='best')
ax_trend.grid(True, alpha=0.3, linestyle='--')
ax_trend.set_xlim(0, n_days+1)

# ============================================
# PANELS 6-9: THE DEEP DIVES (Category Analysis)
# "What specific changes would help most?"
# ============================================

# Define categories with meaningful thresholds
life_df['Sleep_Cat'] = pd.cut(life_df['Sleep_Hours'], 
                              bins=[0, 6, 7.5, 12], 
                              labels=['Poor\n(<6h)', 'Good\n(6-7.5h)', 'Great\n(>7.5h)'])

life_df['Exercise_Cat'] = pd.cut(life_df['Exercise_Min'], 
                                 bins=[-1, 0, 30, 200], 
                                 labels=['None\n(0min)', 'Light\n(<30min)', 'Active\n(>30min)'])

life_df['Screen_Cat'] = pd.cut(life_df['Screen_Hours'], 
                               bins=[0, 3, 5, 15], 
                               labels=['Low\n(<3h)', 'Medium\n(3-5h)', 'High\n(>5h)'])

life_df['Stress_Cat'] = pd.cut(life_df['Stress_Level'], 
                               bins=[0, 4, 7, 11], 
                               labels=['Low\n(<4)', 'Medium\n(4-7)', 'High\n(>7)'])

# Panel 6: Sleep Impact
ax_cat1 = fig.add_subplot(gs[3, 0])
sns.barplot(data=life_df, x='Sleep_Cat', y='Life_Satisfaction',
           ax=ax_cat1, palette='Blues', ci=95, capsize=0.1, errwidth=2)
ax_cat1.set_title('😴 Sleep Quality\nImpact on Happiness', 
                  fontsize=12, fontweight='bold', pad=10)
ax_cat1.set_xlabel('')
ax_cat1.set_ylabel('Avg Satisfaction', fontsize=10)
ax_cat1.grid(axis='y', alpha=0.3, linestyle='--')
# Add sample sizes
for i, cat in enumerate(life_df['Sleep_Cat'].cat.categories):
    count = (life_df['Sleep_Cat'] == cat).sum()
    ax_cat1.text(i, 0.5, f'n={count}', ha='center', fontsize=9, fontweight='bold')

# Panel 7: Exercise Impact
ax_cat2 = fig.add_subplot(gs[3, 1])
sns.barplot(data=life_df, x='Exercise_Cat', y='Life_Satisfaction',
           ax=ax_cat2, palette='Greens', ci=95, capsize=0.1, errwidth=2)
ax_cat2.set_title('💪 Exercise Amount\nImpact on Happiness', 
                  fontsize=12, fontweight='bold', pad=10)
ax_cat2.set_xlabel('')
ax_cat2.set_ylabel('')
ax_cat2.grid(axis='y', alpha=0.3, linestyle='--')
for i, cat in enumerate(life_df['Exercise_Cat'].cat.categories):
    count = (life_df['Exercise_Cat'] == cat).sum()
    ax_cat2.text(i, 0.5, f'n={count}', ha='center', fontsize=9, fontweight='bold')

# Panel 8: Screen Time Impact
ax_cat3 = fig.add_subplot(gs[3, 2])
sns.barplot(data=life_df, x='Screen_Cat', y='Life_Satisfaction',
           ax=ax_cat3, palette='Oranges_r', ci=95, capsize=0.1, errwidth=2)
ax_cat3.set_title('📱 Screen Time\nImpact on Happiness', 
                  fontsize=12, fontweight='bold', pad=10)
ax_cat3.set_xlabel('')
ax_cat3.set_ylabel('')
ax_cat3.grid(axis='y', alpha=0.3, linestyle='--')
for i, cat in enumerate(life_df['Screen_Cat'].cat.categories):
    count = (life_df['Screen_Cat'] == cat).sum()
    ax_cat3.text(i, 0.5, f'n={count}', ha='center', fontsize=9, fontweight='bold')

# Panel 9: Stress Impact
ax_cat4 = fig.add_subplot(gs[3, 3])
sns.barplot(data=life_df, x='Stress_Cat', y='Life_Satisfaction',
           ax=ax_cat4, palette='Reds_r', ci=95, capsize=0.1, errwidth=2)
ax_cat4.set_title('😰 Stress Level\nImpact on Happiness', 
                  fontsize=12, fontweight='bold', pad=10)
ax_cat4.set_xlabel('')
ax_cat4.set_ylabel('')
ax_cat4.grid(axis='y', alpha=0.3, linestyle='--')
for i, cat in enumerate(life_df['Stress_Cat'].cat.categories):
    count = (life_df['Stress_Cat'] == cat).sum()
    ax_cat4.text(i, 0.5, f'n={count}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

print("✅ Statistical narrative complete!\n")

# ============================================
# 📖 THE STORY: Insights & Recommendations
# ============================================
print("="*70)
print("📖 THE COMPLETE STORY: What Your Data Reveals")
print("="*70)

print("\n" + "─"*70)
print("CHAPTER 1: The Correlation Story")
print("─"*70)

# Calculate all correlations
correlations = {
    'Sleep': life_df['Sleep_Hours'].corr(life_df['Life_Satisfaction']),
    'Exercise': life_df['Exercise_Min'].corr(life_df['Life_Satisfaction']),
    'Stress': life_df['Stress_Level'].corr(life_df['Life_Satisfaction']),
    'Social Time': life_df['Social_Hours'].corr(life_df['Life_Satisfaction']),
    'Screen Time': life_df['Screen_Hours'].corr(life_df['Life_Satisfaction']),
    'Coffee': life_df['Coffee_Cups'].corr(life_df['Life_Satisfaction'])
}

# Sort by impact
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n🔗 Ranked by impact on your happiness:\n")
for rank, (factor, corr) in enumerate(sorted_corr, 1):
    direction = "boosts" if corr > 0 else "hurts"
    strength = "STRONGLY" if abs(corr) > 0.5 else "moderately" if abs(corr) > 0.3 else "slightly"
    emoji = "✅" if (corr > 0 and abs(corr) > 0.3) else "⚠️" if (corr < 0 and abs(corr) > 0.3) else "ℹ️"
    
    print(f"   {rank}. {emoji} {factor:15} {corr:+.3f}  → {strength} {direction} happiness")

print("\n" + "─"*70)
print("CHAPTER 2: The Weekly Pattern Story")
print("─"*70)

weekday_sat = life_df[~life_df['Weekend']]['Life_Satisfaction']
weekend_sat = life_df[life_df['Weekend']]['Life_Satisfaction']
t_stat, p_val = stats.ttest_ind(weekday_sat, weekend_sat)

print(f"\n   Weekday satisfaction: {weekday_sat.mean():.2f} ± {weekday_sat.std():.2f}")
print(f"   Weekend satisfaction: {weekend_sat.mean():.2f} ± {weekend_sat.std():.2f}")
print(f"   Difference: {weekend_sat.mean() - weekday_sat.mean():+.2f} points")
print(f"   Statistical test: p = {p_val:.4f}", end="")

if p_val < 0.05:
    print(" ✅ SIGNIFICANT - weekends are genuinely different!")
    if weekend_sat.mean() > weekday_sat.mean():
        print("   💡 Insight: Weekends make you happier - bring weekend habits to weekdays!")
    else:
        print("   💡 Insight: Weekdays are better - your weekend routine might need work!")
else:
    print(" → Not significant - no real weekday/weekend difference")

print("\n" + "─"*70)
print("CHAPTER 3: The Optimization Recipe")
print("─"*70)

# Find best and worst days
high_sat = life_df[life_df['Life_Satisfaction'] > life_df['Life_Satisfaction'].quantile(0.75)]
low_sat = life_df[life_df['Life_Satisfaction'] < life_df['Life_Satisfaction'].quantile(0.25)]

print(f"\n   🌟 YOUR TOP 25% HAPPIEST DAYS (n={len(high_sat)}):")
print(f"      Average satisfaction: {high_sat['Life_Satisfaction'].mean():.2f}")
print(f"      Sleep:        {high_sat['Sleep_Hours'].mean():.1f}h ± {high_sat['Sleep_Hours'].std():.1f}")
print(f"      Exercise:     {high_sat['Exercise_Min'].mean():.0f}min ± {high_sat['Exercise_Min'].std():.0f}")
print(f"      Stress:       {high_sat['Stress_Level'].mean():.1f}/10 ± {high_sat['Stress_Level'].std():.1f}")
print(f"      Social time:  {high_sat['Social_Hours'].mean():.1f}h ± {high_sat['Social_Hours'].std():.1f}")
print(f"      Screen time:  {high_sat['Screen_Hours'].mean():.1f}h ± {high_sat['Screen_Hours'].std():.1f}")

print(f"\n   😔 YOUR BOTTOM 25% WORST DAYS (n={len(low_sat)}):")
print(f"      Average satisfaction: {low_sat['Life_Satisfaction'].mean():.2f}")
print(f"      Sleep:        {low_sat['Sleep_Hours'].mean():.1f}h ± {low_sat['Sleep_Hours'].std():.1f}")
print(f"      Exercise:     {low_sat['Exercise_Min'].mean():.0f}min ± {low_sat['Exercise_Min'].std():.0f}")
print(f"      Stress:       {low_sat['Stress_Level'].mean():.1f}/10 ± {low_sat['Stress_Level'].std():.1f}")
print(f"      Social time:  {low_sat['Social_Hours'].mean():.1f}h ± {low_sat['Social_Hours'].std():.1f}")
print(f"      Screen time:  {low_sat['Screen_Hours'].mean():.1f}h ± {low_sat['Screen_Hours'].std():.1f}")

print("\n   📊 THE DIFFERENCES (Best vs Worst):")
print(f"      Sleep:   {high_sat['Sleep_Hours'].mean() - low_sat['Sleep_Hours'].mean():+.1f}h more on happy days")
print(f"      Exercise: {high_sat['Exercise_Min'].mean() - low_sat['Exercise_Min'].mean():+.0f}min more on happy days")
print(f"      Stress:   {high_sat['Stress_Level'].mean() - low_sat['Stress_Level'].mean():+.1f} lower on happy days")
print(f"      Screen:   {high_sat['Screen_Hours'].mean() - low_sat['Screen_Hours'].mean():+.1f}h less on happy days")

print("\n" + "─"*70)
print("CHAPTER 4: Your Trajectory")
print("─"*70)

trend_slope = np.polyfit(life_df['Day_Number'], life_df['Life_Satisfaction'], 1)[0]
total_change = trend_slope * n_days

print(f"\n   📈 Overall trend: {trend_slope:+.5f} points per day")
print(f"   📊 Total change over 90 days: {total_change:+.2f} points")

if abs(trend_slope) < 0.005:
    print("   → Stable trajectory - you're maintaining consistency")
elif trend_slope > 0:
    print(f"   ✅ Improving! You gained {total_change:.2f} points - keep it up!")
else:
    print(f"   ⚠️  Declining. You lost {abs(total_change):.2f} points - time to intervene")

print("\n" + "="*70)
print("💪 ACTIONABLE RECOMMENDATIONS")
print("="*70)

recommendations = []

# Sleep recommendation
if correlations['Sleep'] > 0.3:
    sleep_diff = high_sat['Sleep_Hours'].mean() - low_sat['Sleep_Hours'].mean()
    if sleep_diff > 0.5:
        target_sleep = high_sat['Sleep_Hours'].mean()
        recommendations.append(
            f"😴 SLEEP: Aim for {target_sleep:.1f}h per night " +
            f"(currently averaging {life_df['Sleep_Hours'].mean():.1f}h)\n" +
            f"   Impact: Could boost happiness by ~{sleep_diff * correlations['Sleep']:.1f} points"
        )

# Exercise recommendation
if correlations['Exercise'] > 0.3:
    exercise_diff = high_sat['Exercise_Min'].mean() - low_sat['Exercise_Min'].mean()
    if exercise_diff > 15:
        target_exercise = high_sat['Exercise_Min'].mean()
        recommendations.append(
            f"💪 EXERCISE: Target {target_exercise:.0f}min per day " +
            f"(currently averaging {life_df['Exercise_Min'].mean():.0f}min)\n" +
            f"   Impact: Could boost happiness by ~{(exercise_diff/60) * correlations['Exercise']:.1f} points"
        )

# Screen time recommendation
if correlations['Screen Time'] < -0.3:
    screen_diff = low_sat['Screen_Hours'].mean() - high_sat['Screen_Hours'].mean()
    if screen_diff > 0.5:
        target_screen = high_sat['Screen_Hours'].mean()
        recommendations.append(
            f"📱 SCREEN TIME: Reduce to {target_screen:.1f}h per day " +
            f"(currently averaging {life_df['Screen_Hours'].mean():.1f}h)\n" +
            f"   Impact: Could boost happiness by ~{screen_diff * abs(correlations['Screen Time']):.1f} points"
        )

# Stress recommendation
if correlations['Stress'] < -0.3:
    recommendations.append(
        f"😰 STRESS MANAGEMENT: Critical for your happiness\n" +
        f"   High stress days average {low_sat['Stress_Level'].mean():.1f}/10 stress\n" +
        f"   Consider: meditation, time management, delegation, therapy"
    )

# Weekend recommendation
if weekend_sat.mean() > weekday_sat.mean() + 0.5:
    recommendations.append(
        f"📅 WEEKEND HABITS: You're {weekend_sat.mean() - weekday_sat.mean():.1f} points happier on weekends\n" +
        f"   What weekend habits can you bring to weekdays?\n" +
        f"   (More sleep? More social time? Less stress?)"
    )

# Print recommendations
if recommendations:
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")
else:
    print("\n✅ Your habits are well-balanced! Keep doing what you're doing.")

print("\n" + "="*70)
print("📚 KEY LESSONS FROM THIS STATISTICAL NARRATIVE")
print("="*70)
print("""
1. MULTIPLE PERSPECTIVES: Each panel tells part of the story
   - Main plot shows relationships
   - Heatmap reveals connections
   - Distributions show what's normal
   - Time trends show trajectory
   - Categories enable comparisons

2. COORDINATED DESIGN: All panels work together
   - Same color schemes
   - Consistent styling
   - Clear hierarchy (big plot = main story)
   - Each panel answers a specific question

3. FROM DATA TO DECISIONS: Numbers → Insights → Actions
   - Correlations show what matters
   - Statistical tests prove it's real
   - Comparisons reveal optimal ranges
   - Recommendations drive behavior change

4. TELL A STORY: Guide the viewer's journey
   - Start with big picture (Panel 1)
   - Show connections (Panel 2)
   - Reveal patterns (Panels 3-5)
   - Provide specifics (Panels 6-9)
   - End with clear takeaways

This is how data scientists communicate complex findings!
""")
print("="*70 + "\n")

print("🎓 Now you understand multi-panel statistical narratives!")
print("💡 Try adapting this to YOUR data - health, finances, productivity, etc.")
print("🚀 This is production-level data visualization!\n")