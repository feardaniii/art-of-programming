import plotly.graph_objects as go
import numpy as np
import pandas as pd

print("="*70)
print("EXERCIȚIUL 4: Heatmap - Matrice de Corelații 10x10")
print("="*70)

# ==================== STEP 1: Generate Random 10x10 Matrix ====================
print("\n🎲 STEP 1: Generare matrice random 10×10")
print("="*70)

# Set seed
np.random.seed(42)

# Generate random matrix (0-100)
matrix = np.random.randint(0, 100, size=(10, 10))

# Create labels
labels = [f'Var{i+1}' for i in range(10)]

print(f"✓ Matrice generată: {matrix.shape[0]}×{matrix.shape[1]}")
print(f"✓ Valori: 0-100")
print(f"✓ Labels: {labels}")

print("\n🔍 Primele 5x5 valori:")
print(matrix[:5, :5])

# ==================== STEP 2: Basic Heatmap ====================
print("\n" + "="*70)
print("🔥 STEP 2: Heatmap simplu")
print("="*70)

# Create basic heatmap
fig_basic = go.Figure(data=go.Heatmap(
    z=matrix,
    x=labels,
    y=labels,
    colorscale='Viridis',  # Color scheme
    colorbar=dict(title="Valoare")
))

fig_basic.update_layout(
    title='🔥 Heatmap 10x10 - Basic',
    xaxis_title="Variabile (Coloane)",
    yaxis_title="Variabile (Rânduri)",
    width=700,
    height=700
)

print("✓ Heatmap basic creat")
print("  🎨 Colorscale: Viridis (purple → yellow)")

fig_basic.show()

# ==================== STEP 3: Heatmap with Text (Values Displayed) ====================
print("\n" + "="*70)
print("📊 STEP 3: Heatmap cu VALORI afișate în celule")
print("="*70)

# Create heatmap with text annotations
fig_text = go.Figure(data=go.Heatmap(
    z=matrix,
    x=labels,
    y=labels,
    colorscale='Viridis',
    text=matrix,  # Display values
    texttemplate='%{text}',  # Format
    textfont={"size": 10},
    colorbar=dict(title="Valoare")
))

fig_text.update_layout(
    title='🔥 Heatmap 10x10 - Cu Valori Afișate',
    xaxis_title="Variabile (Coloane)",
    yaxis_title="Variabile (Rânduri)",
    width=800,
    height=800
)

print("✓ Heatmap cu text creat")
print("  📊 text=matrix - valorile sunt afișate!")
print("  🔍 Hover pentru info detaliată")

fig_text.show()

# ==================== STEP 4: BONUS - Real Correlation Matrix ====================
print("\n" + "="*70)
print("🎁 STEP 4: BONUS - Matrice de Corelații REALĂ")
print("="*70)

# Generate 10 variables with correlations
print("⏳ Generare 10 variabile corelate...")

# Generate correlated data
n_samples = 100
n_vars = 10

# Start with random data
data = np.random.randn(n_samples, n_vars)

# Create some correlations by making some variables depend on others
# Var2 correlates with Var1
data[:, 1] = data[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2

# Var3 correlates with Var1
data[:, 2] = data[:, 0] * 0.6 + np.random.randn(n_samples) * 0.4

# Var5 correlates with Var4
data[:, 4] = data[:, 3] * 0.7 + np.random.randn(n_samples) * 0.3

# Var7 correlates with Var6
data[:, 6] = data[:, 5] * 0.9 + np.random.randn(n_samples) * 0.1

# Create DataFrame
df_corr = pd.DataFrame(data, columns=labels)

# Calculate correlation matrix
corr_matrix = df_corr.corr()

print(f"✓ Matrice de corelații calculată")
print(f"  📊 {n_samples} observații, {n_vars} variabile")
print(f"  🔗 Corelații artificiale create între unele variabile")

print("\n🔍 Matrice corelații (primele 5×5):")
print(corr_matrix.iloc[:5, :5].round(2))

# ==================== STEP 5: Correlation Heatmap (Final Version) ====================
print("\n" + "="*70)
print("🎨 STEP 5: Heatmap Corelații (versiune finală)")
print("="*70)

# Create correlation heatmap with diverging colorscale
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',  # Red-Blue diverging (perfect for correlations!)
    zmid=0,  # Center colorscale at 0
    zmin=-1,
    zmax=1,
    text=np.round(corr_matrix.values, 2),  # Round to 2 decimals
    texttemplate='%{text}',
    textfont={"size": 11, "color": "white"},
    colorbar=dict(
        title="Corelație",
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0']
    ),
    hovertemplate='%{y} ↔ %{x}<br>Corelație: %{z:.3f}<extra></extra>'
))

# Customize layout
fig.update_layout(
    title={
        'text': '🔥 Matrice de Corelații 10×10 (Interactive)',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16, 'family': 'Arial Black'}
    },
    xaxis_title="Variabile",
    yaxis_title="Variabile",
    width=900,
    height=900,
    xaxis=dict(
        tickangle=-45,
        side='bottom'
    ),
    yaxis=dict(
        autorange='reversed'  # Start from top
    )
)

print("✓ Heatmap final de corelații creat")
print("  🎨 Colorscale RdBu: roșu (negativ) → alb (0) → albastru (pozitiv)")
print("  📊 Valori rotunjite la 2 zecimale")
print("  🔍 Hover pentru corelație exactă")
print("  💡 Diagonala = 1.0 (fiecare variabilă cu ea însăși)")

fig.show()

# ==================== STEP 6: Analysis ====================
print("\n" + "="*70)
print("🔍 STEP 6: Analiză corelații")
print("="*70)

# Find strongest correlations (excluding diagonal)
print("\n💪 Cele mai puternice corelații (|r| > 0.5):")

# Get upper triangle (avoid duplicates)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
upper_triangle = corr_matrix.where(mask)

# Find strong correlations
strong_corrs = []
for col in upper_triangle.columns:
    for idx in upper_triangle.index:
        value = upper_triangle.loc[idx, col]
        if pd.notna(value) and abs(value) > 0.5:
            strong_corrs.append((idx, col, value))

# Sort by absolute value
strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

if strong_corrs:
    for var1, var2, corr in strong_corrs[:10]:  # Top 10
        emoji = "📈" if corr > 0 else "📉"
        strength = "PUTERNICĂ" if abs(corr) > 0.7 else "MODERATĂ"
        print(f"  {emoji} {var1} ↔ {var2}: {corr:+.3f} ({strength})")
else:
    print("  Nicio corelație puternică găsită")

# Statistics
print("\n📊 Statistici generale:")
# Flatten correlation matrix, exclude diagonal
corr_values = corr_matrix.values[~np.eye(10, dtype=bool)]
print(f"  Media corelațiilor: {corr_values.mean():.3f}")
print(f"  Std dev: {corr_values.std():.3f}")
print(f"  Min: {corr_values.min():.3f}")
print(f"  Max: {corr_values.max():.3f}")

# ==================== STEP 7: Different Colorscales Comparison ====================
print("\n" + "="*70)
print("🎨 STEP 7: Comparație colorscale-uri")
print("="*70)

# Show different colorscales for comparison
colorscales = ['Viridis', 'RdBu', 'Blues', 'Reds', 'RdYlGn', 'Plasma']

print("\n🎨 Colorscale-uri disponibile pentru heatmap:")
for cs in colorscales:
    print(f"  • {cs}")

print("\n💡 Recomandări:")
print("  • Corelații (-1 to 1): RdBu, RdYlGn (diverging)")
print("  • Valori pozitive (0+): Viridis, Blues, Plasma (sequential)")
print("  • Heatmaps generale: Viridis (colorblind-safe)")

# ==================== STEP 8: Save ====================
print("\n" + "="*70)
print("💾 STEP 8: Salvare heatmap")
print("="*70)

# Save both versions
fig_text.write_html("heatmap_random.html")
fig.write_html("heatmap_correlations.html")

print("✅ Heatmap-uri salvate:")
print("  • heatmap_random.html (matrice random)")
print("  • heatmap_correlations.html (corelații)")