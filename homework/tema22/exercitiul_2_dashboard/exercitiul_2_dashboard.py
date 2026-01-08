import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

print("="*70)
print("EXERCIȚIUL 2: Dashboard cu 3 Grafice - Buget Personal")
print("="*70)

# ==================== STEP 1: Generate Financial Data ====================
print("\n💰 STEP 1: Generare date financiare (6 luni)")
print("="*70)

# Monthly data for 6 months
months = ['Ian', 'Feb', 'Mar', 'Apr', 'Mai', 'Iun']

# Income with slight growth
np.random.seed(42)
venituri = [3000, 3200, 3500, 3400, 3600, 3800]

# Expenses (should be less than income)
cheltuieli = [2500, 2700, 2800, 2600, 2900, 2700]

# Savings = Income - Expenses
economii = [v - c for v, c in zip(venituri, cheltuieli)]

print(f"✓ Date generate pentru {len(months)} luni")
print(f"\n📊 Rezumat financiar:")
print(f"  Venituri totale: {sum(venituri):,} RON")
print(f"  Cheltuieli totale: {sum(cheltuieli):,} RON")
print(f"  Economii totale: {sum(economii):,} RON")
print(f"  Rată economisire: {sum(economii)/sum(venituri)*100:.1f}%")

# Expense breakdown for June (current month)
cheltuieli_categorii = {
    'Mâncare': 800,
    'Chirie': 1200,
    'Transport': 300,
    'Distracție': 400
}

print(f"\n🥧 Breakdown cheltuieli Iunie:")
for cat, val in cheltuieli_categorii.items():
    pct = (val / sum(cheltuieli_categorii.values())) * 100
    print(f"  {cat:12s}: {val:4d} RON ({pct:5.1f}%)")

# ==================== STEP 2: Create Subplot Structure ====================
print("\n" + "="*70)
print("🏗️ STEP 2: Creare structură dashboard (2x2 grid)")
print("="*70)

# Create 2x2 grid with custom specs
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        '💰 Venituri vs Cheltuieli', 
        '📈 Trend Economii', 
        '🥧 Breakdown Cheltuieli'
    ),
    specs=[
        [{"type": "bar"}, {"type": "scatter"}],  # Top row: bar and line
        [{"type": "pie", "colspan": 2}, None]     # Bottom: pie spans both columns
    ],
    row_heights=[0.5, 0.5],  # Equal height for rows
    vertical_spacing=0.15,   # Space between rows
    horizontal_spacing=0.1   # Space between columns
)

print("✓ Grid creat: 2 rânduri × 2 coloane")
print("  • Top-left: Bar chart (venituri vs cheltuieli)")
print("  • Top-right: Line chart (economii)")
print("  • Bottom: Pie chart (ocupă ambele coloane)")

# ==================== STEP 3: SUBPLOT 1 - Bar Chart ====================
print("\n" + "="*70)
print("📊 STEP 3: Subplot 1 - Bar Chart (Venituri vs Cheltuieli)")
print("="*70)

# Add Income bars
fig.add_trace(
    go.Bar(
        x=months,
        y=venituri,
        name='Venituri',
        marker_color='#10b981',  # Green
        marker_line_color='black',
        marker_line_width=1.5,
        text=venituri,
        textposition='outside',
        texttemplate='%{text} RON',
        hovertemplate='<b>%{x}</b><br>Venituri: %{y:,} RON<extra></extra>'
    ),
    row=1, col=1
)

# Add Expenses bars
fig.add_trace(
    go.Bar(
        x=months,
        y=cheltuieli,
        name='Cheltuieli',
        marker_color='#ef4444',  # Red
        marker_line_color='black',
        marker_line_width=1.5,
        text=cheltuieli,
        textposition='outside',
        texttemplate='%{text} RON',
        hovertemplate='<b>%{x}</b><br>Cheltuieli: %{y:,} RON<extra></extra>'
    ),
    row=1, col=1
)

print("✓ Bar chart adăugat")
print("  🟢 Verde = Venituri")
print("  🔴 Roșu = Cheltuieli")
print("  📊 Valori afișate pe bare")

# ==================== STEP 4: SUBPLOT 2 - Line Chart ====================
print("\n" + "="*70)
print("📈 STEP 4: Subplot 2 - Line Chart (Trend Economii)")
print("="*70)

# Add Savings line chart
fig.add_trace(
    go.Scatter(
        x=months,
        y=economii,
        name='Economii',
        mode='lines+markers',
        line=dict(color='#3b82f6', width=3),  # Blue
        marker=dict(
            size=12,
            color='#3b82f6',
            line=dict(color='white', width=2)
        ),
        text=economii,
        textposition='top center',
        texttemplate='%{text} RON',
        hovertemplate='<b>%{x}</b><br>Economii: %{y:,} RON<extra></extra>',
        fill='tozeroy',  # Fill area under line
        fillcolor='rgba(59, 130, 246, 0.1)'  # Light blue transparent
    ),
    row=1, col=2
)

# Add average line (reference)
avg_economii = sum(economii) / len(economii)
fig.add_hline(
    y=avg_economii,
    line_dash="dash",
    line_color="orange",
    annotation_text=f"Medie: {avg_economii:.0f} RON",
    annotation_position="right",
    row=1, col=2
)

print("✓ Line chart adăugat")
print(f"  📈 Trend economii cu markeri")
print(f"  📏 Linie medie: {avg_economii:.0f} RON")
print(f"  🎨 Fill sub linie pentru vizual mai bun")

# ==================== STEP 5: SUBPLOT 3 - Pie Chart ====================
print("\n" + "="*70)
print("🥧 STEP 5: Subplot 3 - Pie Chart (Breakdown Cheltuieli)")
print("="*70)

# Add Pie chart for expense breakdown
fig.add_trace(
    go.Pie(
        labels=list(cheltuieli_categorii.keys()),
        values=list(cheltuieli_categorii.values()),
        marker=dict(
            colors=['#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6'],  # Custom colors
            line=dict(color='white', width=2)
        ),
        textposition='inside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>%{value:,} RON<br>%{percent}<extra></extra>',
        hole=0.3,  # Donut chart (hole in middle)
        pull=[0.05, 0, 0, 0]  # Pull out first slice slightly
    ),
    row=2, col=1
)

print("✓ Pie chart adăugat")
print("  🍩 Donut style (hole=0.3)")
print("  🎨 Culori custom per categorie")
print("  📊 Afișare procente și label-uri")
print(f"  💰 Total cheltuieli: {sum(cheltuieli_categorii.values())} RON")

# ==================== STEP 6: Layout Customization ====================
print("\n" + "="*70)
print("✨ STEP 6: Customizare layout general")
print("="*70)

# Update overall layout
fig.update_layout(
    title={
        'text': '💼 Dashboard Financiar Personal - 6 Luni',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'color': '#1f2937', 'family': 'Arial Black'}
    },
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=11)
    ),
    height=800,
    paper_bgcolor='#f9fafb',  # Light gray background
    plot_bgcolor='white',
    font=dict(family='Arial', size=11)
)

# Update axes for bar chart
fig.update_xaxes(
    title_text="Luna",
    showgrid=False,
    row=1, col=1
)
fig.update_yaxes(
    title_text="Suma (RON)",
    showgrid=True,
    gridcolor='lightgray',
    row=1, col=1
)

# Update axes for line chart
fig.update_xaxes(
    title_text="Luna",
    showgrid=False,
    row=1, col=2
)
fig.update_yaxes(
    title_text="Economii (RON)",
    showgrid=True,
    gridcolor='lightgray',
    row=1, col=2
)

print("✓ Layout customizat")
print("  📐 Înălțime: 800px")
print("  🎨 Background: Light gray")
print("  📊 Grid pentru axe Y")
print("  🏷️ Titlu centrat și stilizat")

# ==================== STEP 7: Show Dashboard ====================
print("\n" + "="*70)
print("🎉 STEP 7: Afișare dashboard")
print("="*70)

print("\n🚀 Dashboard se deschide în browser!")
print("\n💡 Features interactive de testat:")
print("  • Hover peste bare/puncte/felii pentru detalii")
print("  • Click pe legendă pentru hide/show serii")
print("  • Zoom pe line chart (drag rectangle)")
print("  • Pan după zoom (click + drag)")
print("  • Download ca PNG (buton toolbar)")

# Show the dashboard
fig.show()

# ==================== STEP 8: Save Dashboard ====================
print("\n" + "="*70)
print("💾 STEP 8: Salvare dashboard")
print("="*70)

# Save as HTML
filename = "dashboard_financiar.html"
fig.write_html(filename, config={'displayModeBar': True})

print(f"✅ Dashboard salvat ca '{filename}'")
print(f"📂 Poți partaja acest fișier - e complet interactiv!")

# ==================== STEP 9: Analysis & Insights ====================
print("\n" + "="*70)
print("🔍 STEP 9: Analiză și Insights")
print("="*70)

print("\n📊 Observații din dashboard:")

# Best/worst months
best_month_idx = economii.index(max(economii))
worst_month_idx = economii.index(min(economii))

print(f"\n1️⃣ Cea mai bună lună:")
print(f"   {months[best_month_idx]}: {economii[best_month_idx]} RON economii")
print(f"   (Venituri: {venituri[best_month_idx]}, Cheltuieli: {cheltuieli[best_month_idx]})")

print(f"\n2️⃣ Cea mai slabă lună:")
print(f"   {months[worst_month_idx]}: {economii[worst_month_idx]} RON economii")
print(f"   (Venituri: {venituri[worst_month_idx]}, Cheltuieli: {cheltuieli[worst_month_idx]})")

# Trend
if economii[-1] > economii[0]:
    trend = "POZITIV 📈"
else:
    trend = "NEGATIV 📉"
print(f"\n3️⃣ Trend economii: {trend}")
print(f"   Prima lună: {economii[0]} RON")
print(f"   Ultima lună: {economii[-1]} RON")
print(f"   Diferență: {economii[-1] - economii[0]:+d} RON")

# Biggest expense category
biggest_cat = max(cheltuieli_categorii, key=cheltuieli_categorii.get)
print(f"\n4️⃣ Cea mai mare cheltuială:")
print(f"   {biggest_cat}: {cheltuieli_categorii[biggest_cat]} RON")
print(f"   ({cheltuieli_categorii[biggest_cat]/sum(cheltuieli_categorii.values())*100:.1f}% din total)")

# Savings rate
savings_rate = (sum(economii) / sum(venituri)) * 100
print(f"\n5️⃣ Rata de economisire:")
print(f"   {savings_rate:.1f}% din venituri")
if savings_rate >= 20:
    print(f"   ✅ EXCELENT! (target: 20%+)")
elif savings_rate >= 10:
    print(f"   👍 BUN (target: 20%)")
else:
    print(f"   ⚠️ ÎMBUNĂTĂȚEȘTE (sub 10%)")