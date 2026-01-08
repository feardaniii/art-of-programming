import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

print("="*70)
print("EXERCIȚIUL 3: Box Plot - Performance pe Echipe")
print("="*70)

# ==================== STEP 1: Generate Team Scores ====================
print("\n🏆 STEP 1: Generare scoruri pentru 4 echipe")
print("="*70)

# Set seed for reproducibility
np.random.seed(42)

# Generate scores for 4 teams with different characteristics
echipe = []
scoruri = []

# Team configurations: (name, mean_score, std_dev, num_scores)
team_configs = [
    ('Echipa A', 80, 10, 25),  # High performers, consistent
    ('Echipa B', 70, 15, 25),  # Medium, variable
    ('Echipa C', 85, 8, 25),   # Best, very consistent
    ('Echipa D', 65, 12, 25)   # Lower, moderate variation
]

for echipa, mean_score, std_dev, n_scores in team_configs:
    # Generate scores with normal distribution
    scores = np.random.normal(mean_score, std_dev, n_scores)
    # Clip to reasonable range (0-100)
    scores = np.clip(scores, 0, 100)
    
    echipe.extend([echipa] * n_scores)
    scoruri.extend(scores)

# Create DataFrame
df_scoruri = pd.DataFrame({
    'Echipa': echipe,
    'Scor': scoruri
})

print(f"✓ Generate {len(df_scoruri)} scoruri ({len(team_configs)} echipe × 25)")

# Show statistics per team
print("\n📊 Statistici per echipă:")
for team in ['Echipa A', 'Echipa B', 'Echipa C', 'Echipa D']:
    team_scores = df_scoruri[df_scoruri['Echipa'] == team]['Scor']
    print(f"\n{team}:")
    print(f"  Media: {team_scores.mean():.1f}")
    print(f"  Mediană: {team_scores.median():.1f}")
    print(f"  Std Dev: {team_scores.std():.1f}")
    print(f"  Min/Max: {team_scores.min():.1f} / {team_scores.max():.1f}")

# ==================== STEP 2: Basic Box Plot ====================
print("\n" + "="*70)
print("📦 STEP 2: Box plot simplu")
print("="*70)

# Create basic box plot with Plotly Express
fig_basic = px.box(
    df_scoruri,
    x='Echipa',
    y='Scor',
    title='📦 Distribuția Scorurilor pe Echipe (Basic)'
)

print("✓ Box plot simplu creat")
print("  📊 Arată mediană, Q1, Q3, min, max")

fig_basic.show()

# ==================== STEP 3: Box Plot with All Points ====================
print("\n" + "="*70)
print("🎯 STEP 3: Box plot cu TOATE punctele vizibile")
print("="*70)

# Box plot with all individual points
fig_points = px.box(
    df_scoruri,
    x='Echipa',
    y='Scor',
    points='all',  # Show ALL individual points! 
    color='Echipa',  # Different color per team
    title='📦 Distribuția Scorurilor - Cu Toate Punctele Vizibile'
)

# Customize
fig_points.update_traces(
    boxmean='sd',  # Show mean and standard deviation
    marker=dict(size=6, opacity=0.6),
    line=dict(width=2)
)

fig_points.update_layout(
    yaxis_title="Scor (puncte)",
    xaxis_title="Echipa",
    showlegend=True,
    hovermode='closest'
)

print("✓ Box plot cu puncte creat")
print("  🎯 points='all' - vezi fiecare scor individual!")
print("  📊 boxmean='sd' - medie + std dev afișate")
print("  🎨 Culori diferite per echipă")

fig_points.show()

# ==================== STEP 4: BONUS - Add Target Line ====================
print("\n" + "="*70)
print("🎁 STEP 4: BONUS - Adăugare linie target (75 puncte)")
print("="*70)

# Create box plot with Graph Objects for more control
fig = go.Figure()

# Add box plot for each team
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

for idx, team in enumerate(['Echipa A', 'Echipa B', 'Echipa C', 'Echipa D']):
    team_data = df_scoruri[df_scoruri['Echipa'] == team]['Scor']
    
    fig.add_trace(go.Box(
        y=team_data,
        name=team,
        boxmean='sd',  # Show mean and std dev
        marker_color=colors[idx],
        boxpoints='all',  # Show all points
        jitter=0.3,  # Spread points horizontally
        pointpos=-1.5,  # Position points to the left
        marker=dict(
            size=5,
            opacity=0.5,
            line=dict(width=0.5, color='white')
        ),
        line=dict(width=2),
        fillcolor=colors[idx],
        opacity=0.7
    ))

# Add horizontal target line at 75
fig.add_hline(
    y=75,
    line_dash="dash",
    line_color="red",
    line_width=3,
    annotation_text="🎯 Target: 75 puncte",
    annotation_position="right",
    annotation=dict(
        font=dict(size=14, color="red", family="Arial Black")
    )
)

# Customize layout
fig.update_layout(
    title={
        'text': '📦 Distribuția Scorurilor pe Echipe (cu Target Line)',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16, 'family': 'Arial Black'}
    },
    yaxis_title="Scor (puncte)",
    xaxis_title="Echipa",
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),
    plot_bgcolor='#f9fafb',
    height=600,
    hovermode='closest'
)

# Add grid
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='lightgray'
)

print("✓ Box plot final cu target line creat")
print("  🎯 Linie roșie la 75 puncte (target)")
print("  📊 Toate echipele comparate vizual")
print("  🎨 Culori custom + grid pentru claritate")

fig.show()

# ==================== STEP 5: Analysis ====================
print("\n" + "="*70)
print("🔍 STEP 5: Analiză performanță")
print("="*70)

# Calculate statistics per team
print("\n📊 Comparație echipe:")

target_score = 75
results = []

for team in ['Echipa A', 'Echipa B', 'Echipa C', 'Echipa D']:
    team_scores = df_scoruri[df_scoruri['Echipa'] == team]['Scor']
    
    median = team_scores.median()
    mean = team_scores.mean()
    std = team_scores.std()
    above_target = (team_scores > target_score).sum()
    total = len(team_scores)
    
    results.append({
        'Team': team,
        'Median': median,
        'Mean': mean,
        'Std': std,
        'Above_Target': above_target,
        'Pct_Above': (above_target/total)*100
    })
    
    print(f"\n{team}:")
    print(f"  Mediană: {median:.1f}")
    print(f"  Media: {mean:.1f}")
    print(f"  Std Dev: {std:.1f}")
    print(f"  Peste target (75): {above_target}/{total} ({(above_target/total)*100:.1f}%)")

# Best team analysis
results_df = pd.DataFrame(results)
best_median = results_df.loc[results_df['Median'].idxmax(), 'Team']
most_consistent = results_df.loc[results_df['Std'].idxmin(), 'Team']
most_above_target = results_df.loc[results_df['Pct_Above'].idxmax(), 'Team']

print("\n" + "="*70)
print("🏆 CÂȘTIGĂTORI")
print("="*70)
print(f"\n🥇 Cea mai bună mediană: {best_median}")
print(f"🎯 Cea mai consistentă (std dev mic): {most_consistent}")
print(f"📈 Cei mai mulți peste target: {most_above_target}")

# Outlier detection
print("\n" + "="*70)
print("🔍 OUTLIERS (scoruri neobișnuite)")
print("="*70)

for team in ['Echipa A', 'Echipa B', 'Echipa C', 'Echipa D']:
    team_scores = df_scoruri[df_scoruri['Echipa'] == team]['Scor']
    
    Q1 = team_scores.quantile(0.25)
    Q3 = team_scores.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = team_scores[(team_scores < lower_bound) | (team_scores > upper_bound)]
    
    if len(outliers) > 0:
        print(f"\n{team}: {len(outliers)} outliers")
        print(f"  Scoruri: {outliers.values}")
    else:
        print(f"\n{team}: Niciun outlier")

# ==================== STEP 6: Save ====================
print("\n" + "="*70)
print("💾 STEP 6: Salvare box plot")
print("="*70)

filename = "box_plot_teams.html"
fig.write_html(filename)

print(f"✅ Box plot salvat ca '{filename}'")