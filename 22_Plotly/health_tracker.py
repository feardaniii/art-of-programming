# 🏥 INTERACTIVE DASHBOARDS WITH DASH - Real User Interactions
"""
📚 CE ÎNVĂȚĂM:
1. Diferența dintre Plotly static și Dash (interactive callbacks)
2. Cum funcționează callbacks: User input → Python function → Updated chart
3. Cum să creezi dashboard-uri care răspund la acțiuni user
4. Real-world: health tracking app cu filtre interactive

🌍 APLICAȚIE REALĂ: Dashboard de sănătate cu interactivitate reală

⚠️ REQUIREMENT: pip install dash
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# SECȚIUNEA 1: PREGĂTIREA DATELOR
# ============================================================================
print("🏥 HEALTH TRACKER - Interactive Dashboard\n")

# Generarea de date realiste pentru 2 săptămâni
dates = [datetime.now() - timedelta(days=13-i) for i in range(14)]
date_strings = [d.strftime('%d/%m') for d in dates]

# Data de health tracking
health_data = pd.DataFrame({
    'Date': date_strings,
    'Sleep (hours)': [7.5, 6.8, 7.2, 8.0, 6.5, 7.8, 8.2, 7.1, 6.9, 7.6, 8.1, 7.4, 6.7, 7.9],
    'Steps': [8500, 7200, 9100, 10200, 6800, 9500, 11200, 8800, 7600, 9800, 10500, 8900, 7400, 9200],
    'Water (L)': [2.1, 1.8, 2.5, 2.8, 1.5, 2.3, 3.0, 2.2, 1.9, 2.6, 2.9, 2.4, 1.7, 2.5],
    'Mood (1-10)': [8, 6, 7, 9, 5, 8, 9, 7, 6, 8, 9, 8, 6, 8],
    'Energy (1-10)': [7, 5, 6, 9, 4, 8, 9, 6, 5, 8, 9, 7, 5, 7]
})

print(f"✅ Loaded {len(health_data)} days of health data")
print(health_data.head(), "\n")

# ============================================================================
# SECȚIUNEA 2: PLOTLY STATIC (pentru comparație)
# ============================================================================
print("📊 PART 1: Creating a STATIC Plotly chart\n")

# Static chart - trebuie să alegi metrica înainte
fig_static = go.Figure()
fig_static.add_trace(go.Scatter(
    x=health_data['Date'],
    y=health_data['Sleep (hours)'],
    mode='lines+markers',
    name='Sleep Hours',
    line=dict(color='#6366f1', width=3)
))

fig_static.update_layout(
    title='🌙 Sleep Tracking (Static)',
    xaxis_title='Date',
    yaxis_title='Hours',
    hovermode='x unified'
)

# Salvează ca HTML
fig_static.write_html("health_static.html")
print("✅ Static chart saved as 'health_static.html'")
print("   ❌ Problem: Dacă vrei să vezi Steps în loc de Sleep, trebuie să")
print("      modifici codul și să rulezi din nou!\n")

# ============================================================================
# SECȚIUNEA 3: DASH INTERACTIVE (REAL MAGIC!)
# ============================================================================
print("🎮 PART 2: Creating an INTERACTIVE Dash app\n")
print("⚡ CE ESTE DASH?")
print("   - Dash = Plotly + Callbacks (user interactions)")
print("   - Callbacks = Functions care răspund la acțiuni user")
print("   - Pattern: User clicks dropdown → Python runs → Chart updates\n")

# ============================================================================
# DASH APP EXAMPLE (copy-paste și rulează separat!)
# ============================================================================
dash_app_code = '''
# 🚀 INTERACTIVE HEALTH DASHBOARD - Rulează acest cod!
# Instalare: pip install dash

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

# Data (același ca mai sus)
dates = [datetime.now() - timedelta(days=13-i) for i in range(14)]
date_strings = [d.strftime('%d/%m') for d in dates]

health_data = pd.DataFrame({
    'Date': date_strings,
    'Sleep (hours)': [7.5, 6.8, 7.2, 8.0, 6.5, 7.8, 8.2, 7.1, 6.9, 7.6, 8.1, 7.4, 6.7, 7.9],
    'Steps': [8500, 7200, 9100, 10200, 6800, 9500, 11200, 8800, 7600, 9800, 10500, 8900, 7400, 9200],
    'Water (L)': [2.1, 1.8, 2.5, 2.8, 1.5, 2.3, 3.0, 2.2, 1.9, 2.6, 2.9, 2.4, 1.7, 2.5],
    'Mood (1-10)': [8, 6, 7, 9, 5, 8, 9, 7, 6, 8, 9, 8, 6, 8],
    'Energy (1-10)': [7, 5, 6, 9, 4, 8, 9, 6, 5, 8, 9, 7, 5, 7]
})

# Creează Dash app
app = Dash(__name__)

# Layout (HTML structure)
app.layout = html.Div([
    html.H1("🏥 Interactive Health Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Label("📊 Select Metric to Track:"),
        dcc.Dropdown(
            id='metric-dropdown',
            options=[
                {'label': '🌙 Sleep Hours', 'value': 'Sleep (hours)'},
                {'label': '🚶 Steps', 'value': 'Steps'},
                {'label': '💧 Water Intake', 'value': 'Water (L)'},
                {'label': '😊 Mood Score', 'value': 'Mood (1-10)'},
                {'label': '⚡ Energy Level', 'value': 'Energy (1-10)'}
            ],
            value='Sleep (hours)',  # Default
            style={'width': '50%', 'margin': 'auto'}
        )
    ], style={'marginBottom': '30px'}),

    # Aici se va afișa graficul
    dcc.Graph(id='health-chart'),

    # Statistics panel
    html.Div(id='stats-panel', style={'textAlign': 'center', 'fontSize': '20px'})
])

# ✨ CALLBACK MAGIC - Aici se întâmplă interactivitatea!
@app.callback(
    [Output('health-chart', 'figure'),
     Output('stats-panel', 'children')],
    Input('metric-dropdown', 'value')
)
def update_chart(selected_metric):
    """
    🎯 Această funcție se execută AUTOMAT când user schimbă dropdown-ul!

    Flow:
    1. User selectează "Steps" din dropdown
    2. Dash vede că Input('metric-dropdown', 'value') s-a schimbat
    3. Dash apelează această funcție cu selected_metric='Steps'
    4. Funcția creează un grafic nou
    5. Dash updatează Output('health-chart', 'figure') automat
    """

    # Creează grafic bazat pe metrica selectată
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=health_data['Date'],
        y=health_data[selected_metric],
        mode='lines+markers',
        name=selected_metric,
        line=dict(width=3),
        marker=dict(size=10)
    ))

    # Customizări pentru fiecare metrică
    colors = {
        'Sleep (hours)': '#6366f1',
        'Steps': '#10b981',
        'Water (L)': '#3b82f6',
        'Mood (1-10)': '#f59e0b',
        'Energy (1-10)': '#ef4444'
    }

    fig.update_traces(line_color=colors.get(selected_metric, '#6366f1'))

    fig.update_layout(
        title=f'📈 {selected_metric} Timeline',
        xaxis_title='Date',
        yaxis_title=selected_metric,
        hovermode='x unified',
        height=500
    )

    # Calculează statistici
    avg = health_data[selected_metric].mean()
    max_val = health_data[selected_metric].max()
    min_val = health_data[selected_metric].min()

    stats_text = html.Div([
        html.H3("📊 Statistics:"),
        html.P(f"Average: {avg:.1f} | Max: {max_val:.1f} | Min: {min_val:.1f}")
    ])

    return fig, stats_text

# Rulează app
if __name__ == '__main__':
    print("\\n🚀 Starting Dash app at http://127.0.0.1:8050")
    print("💡 Open your browser and interact with the dropdown!")
    print("⚡ The chart updates INSTANTLY without reloading the page!\\n")
    app.run(debug=True, port=8050)
'''

# Salvează Dash app ca fișier separat
with open('health_dashboard_interactive.py', 'w') as f:
    f.write(dash_app_code)

print("✅ Interactive Dash app saved as 'health_dashboard_interactive.py'")
print("\n📖 HOW TO RUN THE INTERACTIVE DASHBOARD:")
print("   1. pip install dash")
print("   2. python health_dashboard_interactive.py")
print("   3. Open http://127.0.0.1:8050 in browser")
print("   4. Change dropdown → Chart updates instantly!\n")

# ============================================================================
# SECȚIUNEA 4: CONCEPTE CHEIE - Callbacks Explained
# ============================================================================
print("🎓 ÎNȚELEGEREA CALLBACKS:\n")

callback_explanation = """
┌─────────────────────────────────────────────────────────────────┐
│                    CALLBACK FLOW (DASH)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. USER ACTION                                                 │
│     └─> Clicks dropdown, moves slider, types in input          │
│                                                                 │
│  2. DASH DETECTS CHANGE                                         │
│     └─> Input('component-id', 'property') changed               │
│                                                                 │
│  3. PYTHON FUNCTION RUNS                                        │
│     └─> Your @app.callback function executes                    │
│                                                                 │
│  4. OUTPUT UPDATES                                              │
│     └─> Output('component-id', 'property') updated              │
│                                                                 │
│  5. BROWSER UPDATES                                             │
│     └─> User sees new chart/text instantly (no page reload!)   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

💡 KEY INSIGHT: Callbacks transform static charts into APPLICATIONS!
"""

print(callback_explanation)

# ============================================================================
# SECȚIUNEA 5: COMPARAȚIE - Static vs Interactive
# ============================================================================
print("\n📊 STATIC vs INTERACTIVE:\n")

comparison = """
╔═══════════════════════════════════════════════════════════════════╗
║                   PLOTLY STATIC vs DASH                           ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  PLOTLY (Static):                                                 ║
║    ✅ Simple - mai puțin cod                                      ║
║    ✅ Perfect pentru rapoarte, prezentări                         ║
║    ❌ Trebuie să modifici codul pentru a vedea date diferite      ║
║    ❌ Nu răspunde la input user în timp real                      ║
║                                                                   ║
║  DASH (Interactive):                                              ║
║    ✅ User poate explora datele fără cod                          ║
║    ✅ Updates în timp real (no page reload)                       ║
║    ✅ Perfect pentru web apps, dashboards production              ║
║    ❌ Mai mult cod (callbacks, layout)                            ║
║    ❌ Necesită server running (Flask backend)                     ║
║                                                                   ║
╠═══════════════════════════════════════════════════════════════════╣
║  WHEN TO USE WHAT?                                                ║
║                                                                   ║
║  Use PLOTLY (.show(), .write_html()) when:                        ║
║    → One-time analysis/reports                                    ║
║    → Email attachments                                            ║
║    → Simple data exploration                                      ║
║                                                                   ║
║  Use DASH when:                                                   ║
║    → Users need to filter/explore data                            ║
║    → Real-time monitoring dashboards                              ║
║    → Production web applications                                  ║
║    → Multiple user interactions needed                            ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""

print(comparison)

# ============================================================================
# EXECUȚIE: Vezi graficul static
# ============================================================================
print("\n🚀 Opening static chart in browser...\n")
print("💡 TIP: After viewing this, run health_dashboard_interactive.py")
print("         to see the REAL power of Dash callbacks!\n")

fig_static.show()

# ============================================================================
# 🎯 RECAP & NEXT STEPS:
# ============================================================================
print("\n" + "="*70)
print("🎯 RECAP:")
print("="*70)
print("✅ Plotly static = Quick charts, reports, one-time analysis")
print("✅ Dash = Interactive apps cu callbacks pentru user input")
print("✅ Callbacks = Input → Python function → Output update")
print("✅ Dash perfect pentru production dashboards\n")

print("🚀 NEXT STEPS:")
print("1. Run: python health_dashboard_interactive.py")
print("2. Play with the dropdown and watch the chart update!")
print("3. Experiment: Add a date range slider (advanced)")
print("4. Deploy: Heroku, Railway, or any Python hosting\n")

print("📚 LEARNING PATH COMPLETED:")
print("   Script 1 (basic_setup.py) → Plotly fundamentals")
print("   Script 2 (finance_dashboard.py) → Web integration")
print("   Script 3 (health_tracker.py) → Interactive callbacks")
print("\n   🎉 You now understand the full Plotly ecosystem!")
