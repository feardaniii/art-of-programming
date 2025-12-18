
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
    print("\n🚀 Starting Dash app at http://127.0.0.1:8050")
    print("💡 Open your browser and interact with the dropdown!")
    print("⚡ The chart updates INSTANTLY without reloading the page!\n")
    app.run(debug=True, port=8050)
