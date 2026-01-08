import plotly.express as px
import pandas as pd
import numpy as np

print("="*70)
print("EXERCIȚIUL 1: Scatter Plot Interactiv - Vârstă vs Experiență")
print("="*70)

# ==================== STEP 1: Create Personal Dataset ====================
print("\n👥 STEP 1: Creare dataset personal")
print("="*70)

# Create data about you and friends/colleagues
df_personal = pd.DataFrame({
    'Nume': ['Alex', 'Maria', 'Ionuț', 'Ana', 'Mihai', 'Elena', 
             'Cristian', 'Laura', 'Andrei', 'Diana', 'Bogdan', 'Ioana',
             'Răzvan', 'Mădălina', 'Florin'],
    'Vârsta': [25, 28, 22, 30, 27, 24, 26, 29, 23, 31, 25, 28, 24, 27, 32],
    'Ani_Experiență': [3, 5, 1, 7, 4, 2, 3, 6, 2, 8, 3, 5, 2, 4, 9],
    'Salariu_Estimat': [3000, 4500, 2500, 6000, 4000, 3200, 3500, 5000, 
                        2800, 6500, 3300, 4800, 2900, 4200, 7000],
    'Domeniu': ['IT', 'Marketing', 'IT', 'Finance', 'IT', 'HR',
                'Marketing', 'Finance', 'IT', 'Finance', 'Marketing',
                'IT', 'HR', 'Marketing', 'Finance']
})

print(f"✓ Dataset creat: {len(df_personal)} persoane")

print("\n🔍 Primele 5 rânduri:")
print(df_personal.head())

print("\n📊 Statistici:")
print(f"  Vârstă medie: {df_personal['Vârsta'].mean():.1f} ani")
print(f"  Experiență medie: {df_personal['Ani_Experiență'].mean():.1f} ani")
print(f"  Salariu mediu: {df_personal['Salariu_Estimat'].mean():.0f} RON")

print(f"\n🏢 Distribuție pe domenii:")
print(df_personal['Domeniu'].value_counts())

# ==================== STEP 2: Basic Scatter Plot ====================
print("\n" + "="*70)
print("📊 STEP 2: Scatter plot simplu (fără customizări)")
print("="*70)

# Create basic scatter plot
fig_basic = px.scatter(
    df_personal,
    x='Vârsta',
    y='Ani_Experiență',
    title='📊 Vârstă vs Experiență (Basic)'
)

print("✓ Scatter plot basic creat")
print("  🎮 Se va deschide în browser!")
print("  💡 Încearcă să:")
print("     - Pui mouse-ul peste puncte (hover)")
print("     - Faci zoom (drag rectangle)")
print("     - Dai pan (click + drag)")

# Show the plot
fig_basic.show()

# ==================== STEP 3: Scatter with Color ====================
print("\n" + "="*70)
print("🎨 STEP 3: Scatter plot cu CULORI (color by domain)")
print("="*70)

# Scatter plot with colors by domain
fig_color = px.scatter(
    df_personal,
    x='Vârsta',
    y='Ani_Experiență',
    color='Domeniu',  # THIS IS THE MAGIC! 🎨
    title='📊 Vârstă vs Experiență - Colorat pe Domenii'
)

# Customize layout
fig_color.update_layout(
    xaxis_title="Vârsta (ani)",
    yaxis_title="Ani de Experiență",
    font=dict(size=12),
    hovermode='closest'
)

print("✓ Scatter plot cu culori creat")
print("  🎨 Fiecare domeniu are culoare diferită!")
print("  👁️ Click pe legendă pentru a ascunde/arăta domenii!")

fig_color.show()

# ==================== STEP 4: Scatter with Hover Data ====================
print("\n" + "="*70)
print("💬 STEP 4: Scatter plot cu HOVER DATA (info extra)")
print("="*70)

# Scatter plot with custom hover information
fig_hover = px.scatter(
    df_personal,
    x='Vârsta',
    y='Ani_Experiență',
    color='Domeniu',
    hover_data=['Nume', 'Salariu_Estimat'],  # Show name and salary on hover!
    title='📊 Vârstă vs Experiență - Cu Info la Hover'
)

# Customize hover template for better formatting
fig_hover.update_traces(
    hovertemplate='<b>%{customdata[0]}</b><br>' +
                  'Vârstă: %{x} ani<br>' +
                  'Experiență: %{y} ani<br>' +
                  'Salariu: %{customdata[1]:,} RON<br>' +
                  '<extra></extra>'
)

fig_hover.update_layout(
    xaxis_title="Vârsta (ani)",
    yaxis_title="Ani de Experiență",
    font=dict(size=12)
)

print("✓ Scatter plot cu hover data creat")
print("  💬 Pune mouse-ul peste un punct!")
print("  📋 Vei vedea: Nume + Salariu!")

fig_hover.show()

# ==================== STEP 5: BONUS - Bubble Chart (Size by Salary) ====================
print("\n" + "="*70)
print("🎈 STEP 5: BONUS - BUBBLE CHART (size = salary)")
print("="*70)

# Bubble chart with size based on salary
fig_bubble = px.scatter(
    df_personal,
    x='Vârsta',
    y='Ani_Experiență',
    color='Domeniu',
    size='Salariu_Estimat',  # BUBBLE SIZE! 🎈
    hover_data=['Nume', 'Salariu_Estimat'],
    title='🎈 Vârstă vs Experiență - Bubble Chart (Size = Salariu)',
    size_max=50  # Maximum bubble size
)

# Customize
fig_bubble.update_layout(
    xaxis_title="Vârsta (ani)",
    yaxis_title="Ani de Experiență",
    font=dict(size=12)
)

# Custom hover
fig_bubble.update_traces(
    hovertemplate='<b>%{customdata[0]}</b><br>' +
                  'Vârstă: %{x} ani<br>' +
                  'Experiență: %{y} ani<br>' +
                  'Salariu: %{customdata[1]:,} RON<br>' +
                  'Domeniu: %{fullData.name}<br>' +
                  '<extra></extra>'
)

print("✓ Bubble chart creat")
print("  🎈 Puncte mai mari = salarii mai mari!")
print("  💰 Observă vizual cine câștigă mai mult!")

fig_bubble.show()

# ==================== STEP 6: Analysis & Insights ====================
print("\n" + "="*70)
print("🔍 STEP 6: Analiză și Insights")
print("="*70)

# Calculate correlation
from scipy import stats
correlation, p_value = stats.pearsonr(df_personal['Vârsta'], df_personal['Ani_Experiență'])

print(f"\n📊 Corelația vârstă-experiență: {correlation:.3f}")
if correlation > 0.7:
    print("  ✓ Corelație PUTERNICĂ pozitivă - mai mare vârsta, mai multă experiență!")
elif correlation > 0.4:
    print("  ✓ Corelație MODERATĂ pozitivă")
else:
    print("  ≈ Corelație SLABĂ")

# Domain analysis
print(f"\n💰 Salariu mediu pe domenii:")
salary_by_domain = df_personal.groupby('Domeniu')['Salariu_Estimat'].agg(['mean', 'min', 'max'])
for domain, row in salary_by_domain.iterrows():
    print(f"  {domain:12s}: {row['mean']:6.0f} RON (min: {row['min']:.0f}, max: {row['max']:.0f})")

# Experience vs Age ratio
df_personal['Exp_per_Year'] = df_personal['Ani_Experiență'] / (df_personal['Vârsta'] - 18)
print(f"\n📈 Cei mai productivi (experiență/an):")
top_productive = df_personal.nlargest(3, 'Exp_per_Year')[['Nume', 'Vârsta', 'Ani_Experiență', 'Exp_per_Year']]
for idx, row in top_productive.iterrows():
    print(f"  {row['Nume']:12s}: {row['Exp_per_Year']:.2f} ani exp/an de viață")

# ==================== STEP 7: Advanced - Faceted Plot ====================
print("\n" + "="*70)
print("🎯 STEP 7: BONUS ADVANCED - Facet per Domain")
print("="*70)

# Create separate plot for each domain
fig_facet = px.scatter(
    df_personal,
    x='Vârsta',
    y='Ani_Experiență',
    color='Domeniu',
    size='Salariu_Estimat',
    facet_col='Domeniu',  # Separate panel per domain!
    facet_col_wrap=2,  # 2 columns
    hover_data=['Nume', 'Salariu_Estimat'],
    title='📊 Vârstă vs Experiență - Separat pe Domenii (Faceted)'
)

fig_facet.update_layout(
    height=600,
    showlegend=False  # Legend not needed with facets
)

print("✓ Faceted plot creat")
print("  📊 Fiecare domeniu are panoul său!")
print("  🔍 Compară vizual între domenii!")

fig_facet.show()