# 🚀 PLOTLY FUNDAMENTALS - How Interactive Charts Actually Work
"""
📚 CE ÎNVĂȚĂM:
1. Ce este Plotly și cum funcționează (Python → HTML + JavaScript)
2. Două moduri de a crea grafice: Express (rapid) vs Graph Objects (control)
3. Trei moduri de a folosi graficele: Browser, HTML file, Web app

🌍 APLICAȚIE REALĂ: Grafice pentru rapoarte financiare personale
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ============================================================================
# SECȚIUNEA 1: Înțelegerea arhitecturii Plotly
# ============================================================================
print("🔍 ARHITECTURA PLOTLY:")
print("1. Scrii Python code → creezi un obiect Figure")
print("2. Plotly convertește Figure în HTML + JavaScript (plotly.js)")
print("3. Browser-ul renderează interactivitatea")
print("4. De aceea se deschide browser-ul când rulezi fig.show()!\n")

# ============================================================================
# SECȚIUNEA 2: PLOTLY EXPRESS - Quick & Simple
# ============================================================================
print("📊 MODUL 1: Plotly Express (px) - pentru vizualizări rapide\n")

# Data realistă
df = pd.DataFrame({
    'Luna': ['Ian', 'Feb', 'Mar', 'Apr', 'Mai', 'Iun'],
    'Venituri': [850, 650, 900, 1200, 750, 950],
    'Cheltuieli': [820, 590, 875, 1030, 720, 870]
})

# Express = O singură linie pentru grafic complet
fig_express = px.bar(df, x='Luna', y=['Venituri', 'Cheltuieli'],
                     title='💰 Finance Dashboard - Made with Express',
                     color_discrete_map={'Venituri': '#10B981', 'Cheltuieli': '#EF4444'})

# Customizări simple
fig_express.update_layout(
    hovermode='x unified',
    xaxis_title="Luna",
    yaxis_title="Suma (€)"
)

# ============================================================================
# SECȚIUNEA 3: GRAPH OBJECTS - Control Complet
# ============================================================================
print("🎨 MODUL 2: Graph Objects (go) - pentru control maxim\n")

# Același grafic, dar cu control total asupra fiecărui element
fig_objects = go.Figure()

# Adaugă fiecare bară manual (mai multe linii, dar mai mult control)
fig_objects.add_trace(go.Bar(
    x=df['Luna'],
    y=df['Venituri'],
    name='Venituri',
    marker_color='#10B981',
    hovertemplate='<b>%{x}</b><br>Venituri: €%{y}<extra></extra>'
))

fig_objects.add_trace(go.Bar(
    x=df['Luna'],
    y=df['Cheltuieli'],
    name='Cheltuieli',
    marker_color='#EF4444',
    hovertemplate='<b>%{x}</b><br>Cheltuieli: €%{y}<extra></extra>'
))

fig_objects.update_layout(
    title='💰 Finance Dashboard - Made with Graph Objects',
    xaxis_title="Luna",
    yaxis_title="Suma (€)",
    hovermode='x unified'
)

# ============================================================================
# SECȚIUNEA 4: Trei moduri de output (REAL-WORLD)
# ============================================================================
print("🌍 TREI MODURI DE A FOLOSI GRAFICELE:\n")

# MODUL 1: Browser temporar (pentru development/testing)
print("1️⃣ fig.show() → Deschide în browser (doar pentru testare)")
print("   - Server temporar la http://127.0.0.1:8050")
print("   - Se închide când închizi script-ul")
print("   - Perfect pentru: development, explorare, debugging\n")

# MODUL 2: Static HTML file (pentru partajare)
print("2️⃣ fig.write_html() → Salvează ca HTML file")
print("   - File independent, funcționează offline")
print("   - Poți trimite prin email sau hosta pe orice server")
print("   - Perfect pentru: rapoarte, prezentări\n")

fig_express.write_html("finance_report.html")
print("   ✅ Salvat ca 'finance_report.html' - deschide-l manual!\n")

# MODUL 3: JSON pentru web apps (pentru producție)
print("3️⃣ fig.to_json() → JSON pentru integrare în web apps")
print("   - Perfect pentru Flask/FastAPI/React apps")
print("   - Vezi finance_dashboard.py pentru exemplu complet\n")

# Exemplu rapid de JSON output
import json
chart_json = fig_express.to_json()
print(f"   JSON preview: {json.loads(chart_json)['layout']['title']['text']}\n")

# ============================================================================
# SECȚIUNEA 5: Interactivitate - Ce poți face în browser?
# ============================================================================
print("🎮 INTERACTIVITATE AUTOMATĂ (încearcă în browser):")
print("   📌 Hover → Vezi detalii")
print("   📌 Click pe legendă → Hide/show categorii")
print("   📌 Scroll → Zoom in/out")
print("   📌 Drag → Pan (mișcă graficul)")
print("   📌 Buton top-right → Download PNG/Zoom/Reset\n")

# ============================================================================
# EXECUȚIE: Deschide graficul în browser
# ============================================================================
print("🚀 Lansăm graficul în browser...\n")
print("💡 TIP: Când închizi script-ul, server-ul se oprește.")
print("         Pentru acces permanent, folosește .write_html() sau web app!\n")

# Alege pe care vrei să-l vezi (decomentează):
fig_express.show()  # Express version
# fig_objects.show()  # Graph Objects version

# ============================================================================
# 🎯 RECAP:
# ============================================================================
# ✅ Plotly = Python → HTML + JavaScript (de aceea browser)
# ✅ Express (px) = rapid, simplu, mai puțin control
# ✅ Graph Objects (go) = mai multe linii, control total
# ✅ .show() = development | .write_html() = partajare | JSON = web apps
#
# NEXT STEP: finance_dashboard.py → Vezi cum integrezi cu Flask pentru real app!