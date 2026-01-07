import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("="*70)
print("EXERCIȚIUL 6: Barplot - Compararea Salariilor")
print("="*70)

# ==================== STEP 1: Generate Employee Dataset ====================
print("\n👥 STEP 1: Generare dataset angajați")
print("="*70)

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic employee data
df_employees = pd.DataFrame({
    "sex": np.random.choice(["Male", "Female"], size=100),
    "age": np.random.normal(35, 10, size=100),
    "salary": np.random.normal(5000, 1500, size=100),
    "department": np.random.choice(["IT", "HR", "Marketing", "Sales"], size=100),
    "experience": np.random.randint(1, 15, 100),
})

# Ensure age is positive and reasonable
df_employees['age'] = df_employees['age'].clip(22, 65)

# Make salary somewhat correlated with age (older = slightly higher)
df_employees['salary'] = df_employees['salary'] + (df_employees['age'] - 35) * 30

print(f"✓ Dataset generat: {len(df_employees)} angajați")

print("\n🔍 Primele 5 rânduri:")
print(df_employees.head())

print("\n📊 Statistici descriptive:")
print(df_employees.describe())

print(f"\n👥 Distribuție pe sexe:")
print(df_employees['sex'].value_counts())

print(f"\n🏢 Distribuție pe departamente:")
print(df_employees['department'].value_counts())

# ==================== STEP 2: Create Age Groups ====================
print("\n" + "="*70)
print("🔢 STEP 2: Creare grupuri de vârstă")
print("="*70)

# Create age groups using lambda function
df_employees['age_group'] = df_employees['age'].apply(
    lambda x: 'Sub 30 ani' if x < 30 else 'Peste 30 ani'
)

print("✓ Grupuri de vârstă create: 'Sub 30 ani' și 'Peste 30 ani'")

print(f"\n📊 Distribuție pe grupe de vârstă:")
print(df_employees['age_group'].value_counts())

# Calculate statistics per group
print(f"\n💰 Statistici salariu per grup de vârstă:")
for group in df_employees['age_group'].unique():
    group_data = df_employees[df_employees['age_group'] == group]['salary']
    print(f"\n{group}:")
    print(f"  Număr angajați: {len(group_data)}")
    print(f"  Salariu mediu: {group_data.mean():.2f} RON")
    print(f"  Salariu median: {group_data.median():.2f} RON")
    print(f"  Min/Max: {group_data.min():.2f} / {group_data.max():.2f} RON")

# ==================== STEP 3: Basic Barplot (Without Hue) ====================
print("\n" + "="*70)
print("📊 STEP 3: Barplot simplu (fără hue)")
print("="*70)

# Set theme
sns.set_theme(style="whitegrid")

# Create figure
plt.figure(figsize=(10, 6))

# Basic barplot - Seaborn calculates mean automatically!
sns.barplot(data=df_employees, 
            x="age_group", 
            y="salary",
            palette="Set2",
            edgecolor='black',
            linewidth=1.5)

# Customize
plt.title("Salariul Mediu pe Grupe de Vârstă", 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Grupa de Vârstă", fontsize=12, fontweight='bold')
plt.ylabel("Salariu Mediu (RON)", fontsize=12, fontweight='bold')
plt.ylim(0, df_employees['salary'].max() * 1.2)  # Add some space at top

# Add value labels on bars
for i, container in enumerate(plt.gca().containers):
    plt.gca().bar_label(container, fmt='%.0f RON', fontsize=10, fontweight='bold')

plt.tight_layout()
print("✓ Barplot simplu creat")
print("  📊 Seaborn a calculat media AUTOMAT!")
print("  📏 Barele de eroare arată intervalul de încredere")

# Save
plt.savefig('salary_barplot_simple.png', dpi=300, bbox_inches='tight')
print("💾 Salvat ca 'salary_barplot_simple.png'")

plt.show()

# ==================== STEP 4: Barplot WITH Hue (BONUS!) ====================
print("\n" + "="*70)
print("✨ STEP 4: Barplot cu HUE (comparație bărbați vs femei)")
print("="*70)

# Create figure
plt.figure(figsize=(12, 6))

# Barplot with hue - split by sex!
sns.barplot(data=df_employees, 
            x="age_group", 
            y="salary",
            hue="sex",  # THE MAGIC! Split by sex
            palette={"Male": "#4A90E2", "Female": "#E94B3C"},
            edgecolor='black',
            linewidth=1.2)

# Customize
plt.title("Salariul Mediu pe Grupe de Vârstă și Sex", 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Grupa de Vârstă", fontsize=12, fontweight='bold')
plt.ylabel("Salariu Mediu (RON)", fontsize=12, fontweight='bold')
plt.legend(title='Sex', fontsize=11, title_fontsize=12)
plt.ylim(0, df_employees['salary'].max() * 1.2)

# Add grid for better readability
plt.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
print("✓ Barplot cu hue creat")
print("  🔵 Bărbați - albastru")
print("  🔴 Femei - roșu")
print("  📊 Comparație directă între sexe!")

# Save
plt.savefig('salary_barplot_with_hue.png', dpi=300, bbox_inches='tight')
print("💾 Salvat ca 'salary_barplot_with_hue.png'")

plt.show()

# ==================== STEP 5: Detailed Statistics by Group ====================
print("\n" + "="*70)
print("📊 STEP 5: Statistici detaliate per grup")
print("="*70)

# Group by age_group and sex
grouped_stats = df_employees.groupby(['age_group', 'sex'])['salary'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('median', 'median'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
]).round(2)

print("\n📈 Statistici complete:")
print(grouped_stats)

# Calculate differences
print("\n💡 Analiză comparativă:")

# Compare young vs old
young_avg = df_employees[df_employees['age_group'] == 'Sub 30 ani']['salary'].mean()
old_avg = df_employees[df_employees['age_group'] == 'Peste 30 ani']['salary'].mean()
age_diff = old_avg - young_avg
age_diff_pct = (age_diff / young_avg) * 100

print(f"\n🔹 Diferență pe vârstă:")
print(f"  Sub 30: {young_avg:.2f} RON")
print(f"  Peste 30: {old_avg:.2f} RON")
print(f"  Diferență: {age_diff:.2f} RON ({age_diff_pct:+.1f}%)")

# Compare male vs female
male_avg = df_employees[df_employees['sex'] == 'Male']['salary'].mean()
female_avg = df_employees[df_employees['sex'] == 'Female']['salary'].mean()
gender_diff = male_avg - female_avg
gender_diff_pct = (gender_diff / female_avg) * 100

print(f"\n🔹 Diferență pe sex:")
print(f"  Bărbați: {male_avg:.2f} RON")
print(f"  Femei: {female_avg:.2f} RON")
print(f"  Diferență: {gender_diff:.2f} RON ({gender_diff_pct:+.1f}%)")

# ==================== STEP 6: Multiple Comparisons (BONUS) ====================
print("\n" + "="*70)
print("🎨 STEP 6: Comparații multiple (departamente + vârstă)")
print("="*70)

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# LEFT: By age group with hue
sns.barplot(data=df_employees, x="age_group", y="salary", hue="sex",
            palette={"Male": "#4A90E2", "Female": "#E94B3C"},
            ax=axes[0], edgecolor='black', linewidth=1)
axes[0].set_title('Salariu per Vârstă și Sex', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Grupa de Vârstă', fontweight='bold')
axes[0].set_ylabel('Salariu Mediu (RON)', fontweight='bold')
axes[0].legend(title='Sex')
axes[0].grid(axis='y', alpha=0.3)

# RIGHT: By department
sns.barplot(data=df_employees, x="department", y="salary",
            palette="viridis", ax=axes[1], edgecolor='black', linewidth=1)
axes[1].set_title('Salariu per Departament', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Departament', fontweight='bold')
axes[1].set_ylabel('Salariu Mediu (RON)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
print("✓ Două comparații create în paralel")

# Save
plt.savefig('salary_barplot_multiple.png', dpi=300, bbox_inches='tight')
print("💾 Salvat ca 'salary_barplot_multiple.png'")

plt.show()