import plotly.express as px
import pandas as pd
import json
import os

print("="*70)
print("EXERCIȚIUL 5: Salvare și Partajare - Export HTML")
print("="*70)

# ==================== STEP 1: Recreate a Chart (from Ex 1) ====================
print("\n📊 STEP 1: Recreăm un grafic din Exercițiul 1")
print("="*70)

# Recreate the dataset
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

# Create the bubble chart (our best one from Ex 1)
fig = px.scatter(
    df_personal,
    x='Vârsta',
    y='Ani_Experiență',
    color='Domeniu',
    size='Salariu_Estimat',
    hover_data=['Nume', 'Salariu_Estimat'],
    title='🎈 Vârstă vs Experiență - Bubble Chart Interactive',
    size_max=50
)

# Customize
fig.update_layout(
    xaxis_title="Vârsta (ani)",
    yaxis_title="Ani de Experiență",
    font=dict(size=12)
)

print("✓ Grafic recreat (bubble chart cu toate features)")

# ==================== STEP 2: Basic HTML Save ====================
print("\n" + "="*70)
print("💾 STEP 2: Salvare HTML simplă")
print("="*70)

# Save as HTML - SIMPLEST WAY
filename_basic = "my_interactive_chart.html"
fig.write_html(filename_basic)

print(f"✅ Grafic salvat ca '{filename_basic}'")
print(f"📂 Locație: {os.path.abspath(filename_basic)}")
print(f"💡 Deschide fișierul manual în browser și testează!")
print(f"   → Hover, zoom, pan - totul funcționează OFFLINE!")

# ==================== STEP 3: Advanced HTML Save (with options) ====================
print("\n" + "="*70)
print("🎯 STEP 3: Salvare HTML cu opțiuni avansate")
print("="*70)

# Save with advanced options
filename_advanced = "my_chart_standalone.html"

fig.write_html(
    filename_advanced,
    include_plotlyjs='cdn',  # Load plotly.js from internet (smaller file)
    config={
        'displayModeBar': True,  # Show toolbar
        'displaylogo': False,    # Hide Plotly logo
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],  # Remove some tools
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'my_chart',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    }
)

print(f"✅ Grafic avansat salvat ca '{filename_advanced}'")
print(f"📊 Diferențe față de basic:")
print(f"   • Plotly.js încărcat de pe CDN (fișier mai mic)")
print(f"   • Toolbar customizat (unele butoane eliminate)")
print(f"   • Logo Plotly ascuns")
print(f"   • Setări download PNG optimizate")

# ==================== STEP 4: Compare File Sizes ====================
print("\n" + "="*70)
print("📏 STEP 4: Comparație dimensiuni fișiere")
print("="*70)

size_basic = os.path.getsize(filename_basic) / 1024  # KB
size_advanced = os.path.getsize(filename_advanced) / 1024  # KB

print(f"📊 Dimensiuni:")
print(f"   {filename_basic:30s}: {size_basic:8.1f} KB")
print(f"   {filename_advanced:30s}: {size_advanced:8.1f} KB")
print(f"   Diferență: {abs(size_basic - size_advanced):8.1f} KB")

print(f"\n💡 De ce diferență?")
if size_basic > size_advanced:
    print(f"   Basic include Plotly.js în fișier (~3MB)")
    print(f"   Advanced încarcă Plotly.js de pe internet (mai mic)")
else:
    print(f"   Ambele au dimensiuni similare")

# ==================== STEP 5: BONUS - Export as JSON ====================
print("\n" + "="*70)
print("🎁 STEP 5: BONUS - Export ca JSON")
print("="*70)

# Export as JSON (for web apps, React, Vue, etc.)
chart_json = fig.to_json()
filename_json = "my_chart.json"

with open(filename_json, 'w', encoding='utf-8') as f:
    json.dump(json.loads(chart_json), f, indent=2)

print(f"✅ Grafic exportat ca JSON: '{filename_json}'")

# Show a snippet of the JSON
json_data = json.loads(chart_json)
print(f"\n🔍 Structura JSON (primele chei):")
for key in list(json_data.keys())[:5]:
    print(f"   • {key}")

print(f"\n💡 La ce e util JSON?")
print(f"   • Integrare în React/Vue/Angular apps")
print(f"   • Flask/Django backends")
print(f"   • API responses")
print(f"   • Stocare în baze de date")

size_json = os.path.getsize(filename_json) / 1024
print(f"\n📏 Dimensiune JSON: {size_json:.1f} KB")

# ==================== STEP 6: Different Include Options ====================
print("\n" + "="*70)
print("📚 STEP 6: Opțiuni include_plotlyjs")
print("="*70)

print("\n🎯 Opțiuni disponibile pentru include_plotlyjs:")
print("\n1️⃣ include_plotlyjs=True (default)")
print("   • Plotly.js inclus în fișier")
print("   • Fișier mare (~3.5 MB)")
print("   • ✅ Funcționează 100% offline")
print("   • ✅ Nu necesită internet")

print("\n2️⃣ include_plotlyjs='cdn'")
print("   • Plotly.js încărcat de pe internet")
print("   • Fișier mic (~50 KB)")
print("   • ⚠️ Necesită conexiune internet")
print("   • ✅ Ideal pentru email/share")

print("\n3️⃣ include_plotlyjs='directory'")
print("   • Plotly.js în fișier separat")
print("   • Fișiere mici dacă ai multe grafice")
print("   • ⚠️ Trebuie să trimiți ambele fișiere")

print("\n4️⃣ include_plotlyjs=False")
print("   • Fără Plotly.js")
print("   • Cel mai mic fișier")
print("   • ⚠️ Trebuie inclus manual în HTML")

# ==================== STEP 7: Create Comparison Examples ====================
print("\n" + "="*70)
print("🎨 STEP 7: Creează 3 versiuni pentru comparație")
print("="*70)

# Version 1: Full offline (large file)
fig.write_html(
    "chart_offline.html",
    include_plotlyjs=True,
    config={'displayModeBar': True}
)
print("✓ chart_offline.html - Full offline (fișier mare)")

# Version 2: CDN (small file, needs internet)
fig.write_html(
    "chart_cdn.html",
    include_plotlyjs='cdn',
    config={'displayModeBar': True}
)
print("✓ chart_cdn.html - CDN (fișier mic, necesită net)")

# Version 3: No plotly (minimal)
fig.write_html(
    "chart_minimal.html",
    include_plotlyjs=False,
    config={'displayModeBar': True}
)
print("✓ chart_minimal.html - Minimal (cel mai mic)")

# Compare sizes
print(f"\n📊 Comparație dimensiuni:")
for fname in ['chart_offline.html', 'chart_cdn.html', 'chart_minimal.html']:
    if os.path.exists(fname):
        size = os.path.getsize(fname) / 1024
        print(f"   {fname:25s}: {size:8.1f} KB")