# 💰 PRODUCTION-READY DASHBOARD - Dynamic Data + Web Integration
"""
📚 CE ÎNVĂȚĂM:
1. Cum să încarci date dinamic (CSV în loc de hardcoded)
2. Cum să integrezi Plotly în web apps (Flask)
3. Două patterns: HTML embedding vs JSON API
4. Dashboard cu multiple subplots (financial intelligence)

🌍 APLICAȚIE REALĂ: Dashboard financiar pentru web app
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json

# ============================================================================
# SECȚIUNEA 1: DYNAMIC DATA LOADING (Real-World Pattern)
# ============================================================================
print("📂 LOADING DATA FROM CSV (not hardcoded!):\n")

# În real-world, datele vin din CSV/Database/API, nu hardcoded
# Creăm un CSV exemplu (de obicei ai deja acest file)
sample_csv = """Luna,Locuinta,Mancare,Transport,Entertainment,Sanatate
Ian,1200,680,200,300,120
Feb,1200,750,150,450,80
Mar,1250,820,300,500,150
Apr,1250,650,180,320,200
Mai,1250,900,250,600,100
Iun,1300,780,220,420,180"""

# Salvează CSV (de obicei ai deja acest file)
with open('expenses.csv', 'w') as f:
    f.write(sample_csv)

# ÎNCARCĂ din CSV (așa faci în real-world!)
df = pd.read_csv('expenses.csv')
print(f"✅ Loaded {len(df)} months from CSV")
print(df.head(), "\n")

# Extrage datele pentru grafice
months = df['Luna'].tolist()
expense_data = {
    'Locuinta': df['Locuinta'].tolist(),
    'Mancare': df['Mancare'].tolist(),
    'Transport': df['Transport'].tolist(),
    'Entertainment': df['Entertainment'].tolist(),
    'Sanatate': df['Sanatate'].tolist()
}

# ============================================================================
# SECȚIUNEA 2: BUILDING THE DASHBOARD (Multi-Plot Layout)
# ============================================================================
print("📊 CREATING MULTI-PLOT DASHBOARD:\n")

# Subplot-uri pentru dashboard complet
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('📊 Trend Cheltuieli', '🥧 Breakdown Categorii',
                   '📈 Comparație cu Buget', '⚠️ Alertă Categorii'),
    specs=[[{"type": "scatter"}, {"type": "pie"}],
           [{"type": "bar"}, {"type": "scatter"}]]
)

# 1. Trend line interactiv pentru fiecare categorie
for category, values in expense_data.items():
    fig.add_trace(
        go.Scatter(x=months, y=values, name=category,
                  mode='lines+markers',
                  line=dict(width=3),
                  hovertemplate='%{fullData.name}<br>%{y}€<br><extra></extra>'),
        row=1, col=1
    )

# 2. Pie chart interactiv pentru ultima lună
last_month_total = {cat: vals[-1] for cat, vals in expense_data.items()}
fig.add_trace(
    go.Pie(values=list(last_month_total.values()),
           labels=list(last_month_total.keys()),
           name="Breakdown Iun",
           hovertemplate='%{label}<br>€%{value}<br>%{percent}<extra></extra>'),
    row=1, col=2
)

# 3. Comparație cu bugetul planificat
budget = {'Locuinta': 1300, 'Mancare': 700, 'Transport': 200, 
          'Entertainment': 400, 'Sanatate': 150}

categories = list(budget.keys())
actual = [expense_data[cat][-1] for cat in categories]
planned = list(budget.values())

fig.add_trace(go.Bar(x=categories, y=planned, name='Buget Planificat',
                    marker_color='lightblue'), row=2, col=1)
fig.add_trace(go.Bar(x=categories, y=actual, name='Cheltuit Real',
                    marker_color='darkred'), row=2, col=1)

# 4. Alertă pentru categoriile cu creștere > 20%
growth_rates = []
alert_categories = []
for cat in categories:
    if len(expense_data[cat]) >= 2:
        growth = ((expense_data[cat][-1] - expense_data[cat][-2]) / expense_data[cat][-2]) * 100
        growth_rates.append(growth)
        if growth > 20:
            alert_categories.append(cat)

fig.add_trace(
    go.Scatter(x=categories, y=growth_rates, 
              mode='markers+text',
              marker=dict(size=15, color=['red' if cat in alert_categories else 'green' 
                                        for cat in categories]),
              text=[f'{rate:.1f}%' for rate in growth_rates],
              textposition="top center",
              name='Creștere %'),
    row=2, col=2
)

# Layout pentru dashboard profesional
fig.update_layout(
    title_text="💰 Personal Finance Intelligence Dashboard",
    title_x=0.5,
    height=800,
    showlegend=True,
    hovermode='closest'
)

# Configurări specifice pentru fiecare subplot
fig.update_xaxes(title_text="Luna", row=2, col=1)
fig.update_yaxes(title_text="Suma (€)", row=2, col=1)
fig.update_xaxes(title_text="Categorie", row=2, col=2)
fig.update_yaxes(title_text="Creștere (%)", row=2, col=2)

print("✅ Dashboard created with 4 subplots!\n")

# ============================================================================
# SECȚIUNEA 3: WEB INTEGRATION - Cum folosești în real app
# ============================================================================
print("🌍 REAL-WORLD DEPLOYMENT OPTIONS:\n")

# OPȚIUNE 1: Static HTML (simplest)
print("1️⃣ STATIC HTML FILE:")
print("   - Perfect pentru rapoarte, prezentări, email")
fig.write_html("finance_dashboard.html")
print("   ✅ Saved: finance_dashboard.html\n")

# OPȚIUNE 2: JSON API (pentru React/Vue/frontend frameworks)
print("2️⃣ JSON API (pentru frontend frameworks):")
chart_json = json.loads(fig.to_json())
print(f"   - Chart data size: {len(fig.to_json())} characters")
print("   - Use case: Flask/FastAPI endpoint → React/Vue frontend\n")

# OPȚIUNE 3: Flask Integration (full example below)
print("3️⃣ FLASK WEB APP (vezi codul de mai jos):\n")

# ============================================================================
# BONUS: FLASK WEB APP EXAMPLE (copy-paste ready!)
# ============================================================================
flask_example = '''
# 🌐 COPY-PASTE FLASK APP (rulează ca fișier separat!)

from flask import Flask, render_template, jsonify
import plotly
import json

app = Flask(__name__)

@app.route('/')
def index():
    """Servește HTML cu grafic embedded"""
    # Recreează figura (în real-world, apeși o funcție)
    # ... (codul de creare a fig de mai sus)

    # Convertește la JSON pentru template
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('dashboard.html', graphJSON=graphJSON)

@app.route('/api/chart-data')
def chart_data():
    """API endpoint pentru frontend frameworks"""
    return jsonify(json.loads(fig.to_json()))

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# Template HTML (salvează ca templates/dashboard.html):
"""
<!DOCTYPE html>
<html>
<head>
    <title>Finance Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>💰 Your Finance Dashboard</h1>
    <div id="chart"></div>
    <script>
        var graphs = {{graphJSON | safe}};
        Plotly.plot('chart', graphs.data, graphs.layout);
    </script>
</body>
</html>
"""
'''

print(flask_example)

# ============================================================================
# EXECUȚIE: Vezi dashboard-ul în browser
# ============================================================================
print("\n🚀 Opening dashboard in browser...\n")
print("💡 TIP: Pentru production:")
print("   - Folosește .write_html() pentru rapoarte statice")
print("   - Folosește Flask/FastAPI pentru web apps cu date dinamice")
print("   - Vezi health_tracker.py pentru Dash (interactive callbacks)\n")

fig.show()

# ============================================================================
# 🎯 RECAP:
# ============================================================================
# ✅ Data dinamică din CSV (nu hardcoded) → scalabil
# ✅ Multiple subplots → dashboard complet
# ✅ Trei deployment options: HTML, JSON API, Flask embedding
# ✅ Production-ready pattern pentru web integration
#
# NEXT: health_tracker.py → Interactivitate reală cu Dash callbacks!