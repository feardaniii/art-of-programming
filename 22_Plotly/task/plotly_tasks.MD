# 📊 PLOTLY INTERACTIVE VISUALIZATION - TASK

---

## 📚 Introducere

În acest task vei învăța să creezi vizualizări **interactive** cu Plotly. Spre deosebire de matplotlib/seaborn (statice), graficele Plotly permit:
- 🎮 **Hover** pentru detalii
- 🔍 **Zoom** și pan
- 👁️ **Click pe legendă** pentru hide/show
- 💾 **Export** ca PNG direct din browser

---

## 🎯 Ce este Plotly?

Plotly transformă codul tău Python în **HTML + JavaScript** interactiv:

```
Python Code → Plotly Figure → HTML + plotly.js → Browser
```

**Două moduri principale:**
- **Plotly Express (px)**: Rapid, o singură linie, perfect pentru început
- **Graph Objects (go)**: Control complet, mai multe linii, pentru customizări avansate

---

## 🚀 Cod de Start

Rulează acest cod pentru a testa instalarea:
```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Test rapid
fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6], title="🎉 Plotly Works!")
fig.show()  # Deschide în browser
```

**💡 Observație:** Plotly deschide un server temporar în browser. Când închizi script-ul, server-ul se oprește!

---

# PARTEA 1: Fundamente Plotly Express

---

## 📝 EXERCIȚIUL 1: Scatter Plot Personal - Vârsta vs Experiență

**Obiectiv:** Creează un scatter plot cu datele tale și ale prietenilor/colegilor

### 📖 Concepte noi:

**`scatter plot`** = grafic cu puncte, perfect pentru a vedea relații între două variabile numerice
**`hover_data`** = informații extra care apar când pui mouse-ul peste un punct
**`color`** = colorează punctele bazat pe o categorie (ex: sex, departament)

**TODO:**
1. Creează un DataFrame cu date despre 10-15 persoane (tine, prieteni, colegi)
   - Coloane: `Nume`, `Vârsta`, `Ani_Experiență`, `Salariu_Estimat`, `Domeniu`
2. Folosește `px.scatter()` pentru a vizualiza `Vârsta` (X) vs `Ani_Experiență` (Y)
3. Colorează punctele după `Domeniu` folosind parametrul `color`
4. Adaugă `hover_data` pentru a vedea `Nume` și `Salariu_Estimat` la hover
5. **BONUS:** Schimbă dimensiunea punctelor bazat pe `Salariu_Estimat` cu `size`

### 💻 Cod de pornire:
```python
import plotly.express as px
import pandas as pd

# Creează date (exemplu - modifică cu datele tale!)
df_personal = pd.DataFrame({
    'Nume': ['Alex', 'Maria', 'Ionut', 'Ana', 'Mihai', 'Elena'],
    'Vârsta': [25, 28, 22, 30, 27, 24],
    'Ani_Experiență': [3, 5, 1, 7, 4, 2],
    'Salariu_Estimat': [3000, 4500, 2500, 6000, 4000, 3200],
    'Domeniu': ['IT', 'Marketing', 'IT', 'Finance', 'IT', 'HR']
})

# Creează scatter plot
fig = px.scatter(
    df_personal,
    x='Vârsta',
    y='Ani_Experiență',
    color='Domeniu',  # Colorează după domeniu
    hover_data=['Nume', 'Salariu_Estimat'],  # Info la hover
    title='📊 Vârstă vs Experiență - Analiza Prietenilor'
)

# Customizări
fig.update_layout(
    xaxis_title="Vârsta (ani)",
    yaxis_title="Ani de Experiență",
    hovermode='closest'
)

fig.show()

# TODO: Adaugă size='Salariu_Estimat' pentru dimensiuni variate ale punctelor
```

**💡 Hint pentru BONUS:** Adaugă `size='Salariu_Estimat'` în `px.scatter()` și observă cum punctele mai mari = salarii mai mari!

**❓ Întrebare de reflecție:**
Ce corelație observi între vârstă și experiență? Este liniară?

_Răspunsul tău:_ _______________________________________________

---

## 📝 EXERCIȚIUL 2: Dashboard cu 3 Grafice - Buget Personal

**Obiectiv:** Creează un mini-dashboard financiar cu Bar, Line și Pie chart

### 📖 Concepte noi:

**`make_subplots`** = creează grile cu multiple grafice în același layout
**`add_trace`** = adaugă un grafic într-un subplot
**`row, col`** = specifică poziția subplot-ului în grilă

**TODO:**
1. Creează date pentru 6 luni: Venituri, Cheltuieli, Economii
2. **Subplot 1 (Bar Chart):** Comparație Venituri vs Cheltuieli pe luni
3. **Subplot 2 (Line Chart):** Trend economii în timp
4. **Subplot 3 (Pie Chart):** Breakdown cheltuieli pe categorii (Mâncare, Chirie, Transport, Distracție)
5. Adaugă titlu general și customizează layout-ul

### 💻 Cod de pornire:
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Date financiare (6 luni)
months = ['Ian', 'Feb', 'Mar', 'Apr', 'Mai', 'Iun']
venituri = [3000, 3200, 3500, 3400, 3600, 3800]
cheltuieli = [2500, 2700, 2800, 2600, 2900, 2700]
economii = [v - c for v, c in zip(venituri, cheltuieli)]

# Breakdown cheltuieli pentru luna curentă (Iunie)
cheltuieli_categorii = {
    'Mâncare': 800,
    'Chirie': 1200,
    'Transport': 300,
    'Distracție': 400
}

# Creează grid 2x2 (3 subplot-uri folosite)
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('💰 Venituri vs Cheltuieli', '📈 Trend Economii', '🥧 Breakdown Cheltuieli'),
    specs=[[{"type": "bar"}, {"type": "scatter"}],
           [{"type": "pie", "colspan": 2}, None]]  # Pie ocupă ambele coloane
)

# SUBPLOT 1: Bar Chart - Venituri vs Cheltuieli
fig.add_trace(
    go.Bar(x=months, y=venituri, name='Venituri', marker_color='#10b981'),
    row=1, col=1
)
fig.add_trace(
    go.Bar(x=months, y=cheltuieli, name='Cheltuieli', marker_color='#ef4444'),
    row=1, col=1
)

# SUBPLOT 2: Line Chart - Trend Economii
# TODO: Adaugă un line chart cu economiile (folosește go.Scatter cu mode='lines+markers')

# SUBPLOT 3: Pie Chart - Breakdown Cheltuieli
# TODO: Adaugă un pie chart cu cheltuieli_categorii (folosește go.Pie)

# Layout general
fig.update_layout(
    title_text="💼 Dashboard Financiar Personal - 6 Luni",
    title_x=0.5,
    height=700,
    showlegend=True
)

fig.show()
```

**💡 Hint:**
- Line chart: `go.Scatter(x=months, y=economii, mode='lines+markers', name='Economii')`
- Pie chart: `go.Pie(labels=list(cheltuieli_categorii.keys()), values=list(cheltuieli_categorii.values()))`

**❓ Întrebare de reflecție:**
În ce lună ai economisit cel mai mult? Ce insight-uri îți oferă dashboard-ul?

_Răspunsul tău:_ _______________________________________________

---

# PARTEA 2: Vizualizări Avansate

---

## 📝 EXERCIȚIUL 3: Box Plot - Performance pe Echipe

**Obiectiv:** Compară distribuția scorurilor pentru 4 echipe sportive/de muncă

### 📖 Concepte noi:

**`box plot`** = arată distribuția datelor: mediană, quartile, outliers
**`points='all'`** = afișează toate punctele individuale peste box
**`notched=True`** = adaugă "notch" pentru interval de confidență al medianei

**TODO:**
1. Generează scoruri pentru 4 echipe (20-30 de scoruri per echipă)
2. Creează un box plot cu `px.box()` sau `go.Box()`
3. Afișează punctele individuale pentru a vedea distribuția completă
4. Customizează culorile pentru fiecare echipă
5. **BONUS:** Adaugă o linie orizontală pentru "target score" (ex: 75 puncte)

### 💻 Cod de pornire:
```python
import plotly.express as px
import numpy as np
import pandas as pd

# Generează scoruri pentru 4 echipe
np.random.seed(42)
echipe = []
scoruri = []

for echipa, mean_score in [('Echipa A', 80), ('Echipa B', 70), ('Echipa C', 85), ('Echipa D', 65)]:
    scores = np.random.normal(mean_score, 10, 25)  # 25 de scoruri
    echipe.extend([echipa] * 25)
    scoruri.extend(scores)

df_scoruri = pd.DataFrame({
    'Echipa': echipe,
    'Scor': scoruri
})

# Creează box plot
fig = px.box(
    df_scoruri,
    x='Echipa',
    y='Scor',
    title='📦 Distribuția Scorurilor pe Echipe',
    points='all',  # Afișează toate punctele
    color='Echipa'  # Culori diferite per echipă
)

# TODO: Adaugă o linie orizontală pentru target score (75)
# Hint: fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Target")

fig.update_layout(
    yaxis_title="Scor (puncte)",
    xaxis_title="Echipa"
)

fig.show()
```

**💡 Hint pentru BONUS:** Folosește `fig.add_hline(y=75, line_dash="dash", line_color="red")` pentru linia target.

**❓ Întrebare de reflecție:**
Care echipă are cea mai consistentă performanță (variație mică)? Care are outliers?

_Răspunsul tău:_ _______________________________________________

---

## 📝 EXERCIȚIUL 4: Heatmap - Matrice de Corelații 10x10

**Obiectiv:** Vizualizează o matrice de date ca heatmap cu culori

### 📖 Concepte noi:

**`heatmap`** = hartă de culoare unde intensitatea culorii = valoarea numerică
**`colorscale`** = schema de culori (ex: 'Viridis', 'RdBu', 'Blues')
**`text`** = afișează valori în fiecare celulă
**`texttemplate`** = formatare pentru text (ex: '.2f' pentru 2 zecimale)

**TODO:**
1. Generează o matrice 10x10 cu valori random între 0 și 100
2. Creează etichete pentru rânduri și coloane (ex: "Var1", "Var2", ...)
3. Folosește `go.Heatmap()` pentru a vizualiza matricea
4. Adaugă `text` cu valori și `texttemplate` pentru formatare
5. **BONUS:** Calculează și afișează o matrice de corelații reală (genera 10 variabile corelate)

### 💻 Cod de pornire:
```python
import plotly.graph_objects as go
import numpy as np

# Generează matrice random 10x10
np.random.seed(42)
matrix = np.random.randint(0, 100, size=(10, 10))

# Etichete
labels = [f'Var{i+1}' for i in range(10)]

# Creează heatmap
fig = go.Figure(data=go.Heatmap(
    z=matrix,
    x=labels,  # Coloane
    y=labels,  # Rânduri
    colorscale='Viridis',  # Schema de culori
    text=matrix,  # Afișează valori
    texttemplate='%{text}',  # Format
    textfont={"size": 10},
    colorbar=dict(title="Valoare")
))

fig.update_layout(
    title='🔥 Heatmap 10x10 - Matrice Random',
    xaxis_title="Variabile (Coloane)",
    yaxis_title="Variabile (Rânduri)",
    width=700,
    height=700
)

fig.show()

# TODO BONUS: Generează 10 variabile corelate și calculează matricea de corelații
```

**💡 Hint pentru BONUS:**
```python
# Generează 10 variabile cu corelații
data = np.random.randn(100, 10)  # 100 observații, 10 variabile
df_corr = pd.DataFrame(data, columns=labels)
corr_matrix = df_corr.corr()  # Calculează corelațiile

# Apoi folosește corr_matrix.values pentru z în heatmap
# Schimbă colorscale='RdBu' pentru albastru-roșu (perfect pentru corelații -1 to 1)
```

**❓ Întrebare de reflecție:**
În cazul matricei de corelații (BONUS), ce perechi de variabile sunt puternic corelate?

_Răspunsul tău:_ _______________________________________________

---

# PARTEA 3: Real-World Application

---

## 📝 EXERCIȚIUL 5: Salvează și Partajează - HTML Export

**Obiectiv:** Învață să salvezi graficele pentru partajare (email, prezentări)

### 📖 Concepte noi:

**`.write_html()`** = salvează graficul ca fișier HTML independent (funcționează offline!)
**`.to_json()`** = convertește în JSON pentru integrare în web apps
**`include_plotlyjs`** = control asupra cum e încărcat plotly.js

**TODO:**
1. Ia unul din graficele tale de mai sus (scatter, dashboard, sau heatmap)
2. Salvează-l ca fișier HTML folosind `.write_html()`
3. Deschide fișierul în browser și verifică interactivitatea
4. **BONUS:** Exportă ca JSON și inspectează structura

### 💻 Cod de pornire:
```python
# Ia figura din Exercițiul 1 (sau orice altă figură)
# fig = ... (figura ta)

# Salvează ca HTML
fig.write_html("my_interactive_chart.html")
print("✅ Chart salvat ca 'my_interactive_chart.html'")
print("📂 Deschide fișierul manual în browser!")

# Salvare cu opțiuni avansate
fig.write_html(
    "my_chart_standalone.html",
    include_plotlyjs='cdn',  # Încarcă plotly.js de pe internet (fișier mai mic)
    config={'displayModeBar': True}  # Afișează bara de tools
)

# TODO BONUS: Exportă ca JSON
import json
chart_json = fig.to_json()
with open('my_chart.json', 'w') as f:
    json.dump(json.loads(chart_json), f, indent=2)
print("✅ Chart JSON salvat ca 'my_chart.json'")
```

**💡 Observații:**
- Fișierul HTML funcționează **offline** (poți trimite prin email!)
- Toate feature-urile interactive (hover, zoom, pan) funcționează
- JSON e util pentru integrare în React/Vue/Flask apps

**❓ Întrebare de reflecție:**
Când ai folosi `.write_html()` vs `.show()` în proiecte reale?

_Răspunsul tău:_ _______________________________________________

---

# 🎓 Întrebări de Reflecție Finale

După ce ai terminat exercițiile, răspunde la următoarele întrebări:

### 1. Ce diferențe majore ai observat între Plotly și matplotlib/seaborn?

_Răspunsul tău:_
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

### 2. De ce crezi că interactivitatea e importantă în vizualizarea datelor?

_Răspunsul tău:_
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

### 3. Când ai folosi Plotly Express (px) vs Graph Objects (go)?

_Răspunsul tău:_
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

### 4. Ce tip de proiect real ai putea construi folosind Plotly?

_Răspunsul tău:_
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

---

# 💡 Resurse Utile

- 📚 **Documentație Plotly:** https://plotly.com/python/
- 📊 **Galerie de exemple:** https://plotly.com/python/plotly-express/
- 🎨 **Customizare:** https://plotly.com/python/styling-plotly-express/
- 🌐 **Dash (interactive apps):** https://dash.plotly.com/
- 🔥 **Cheat Sheet:** https://images.plot.ly/plotly-documentation/images/python_cheat_sheet.pdf

---

# 🔥 PROVOCARE EXTRA (opțional)

Dacă ai terminat toate exercițiile și vrei să exersezi mai mult:

### 🚀 Challenge 1: Dashboard Interactive cu Dash
Transformă dashboard-ul din Exercițiul 2 într-o aplicație Dash cu:
- Dropdown pentru a selecta luna
- Slider pentru a filtra categoriile de cheltuieli
- Chart care se updatează automat (callbacks!)

**Starter code:**
```python
# pip install dash
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

app = Dash(__name__)

app.layout = html.Div([
    html.H1("💼 Dashboard Financiar Interactive"),
    dcc.Dropdown(
        id='month-dropdown',
        options=[{'label': m, 'value': m} for m in months],
        value='Iun'
    ),
    dcc.Graph(id='expense-chart')
])

@app.callback(
    Output('expense-chart', 'figure'),
    Input('month-dropdown', 'value')
)
def update_chart(selected_month):
    # TODO: Creează grafic bazat pe luna selectată
    pass

if __name__ == '__main__':
    app.run(debug=True)
```

### 📊 Challenge 2: Animated Chart
Creează un scatter plot animat care arată evoluția în timp:
```python
# Exemplu: Evoluția salariilor în timp
fig = px.scatter(
    df_time_series,
    x='Age',
    y='Salary',
    animation_frame='Year',  # Animație pe ani
    color='Department',
    size='Experience',
    title='📈 Evoluția Salariilor 2020-2024'
)
fig.show()
```

### 🗺️ Challenge 3: Geographic Map
Afișează date pe hartă (ex: salarii medii pe județe):
```python
fig = px.choropleth(
    df_counties,
    locations='County',
    color='Average_Salary',
    title='🗺️ Salarii Medii pe Județe'
)
```

---

## ✅ Checklist Final

Înainte de a preda task-ul, verifică:

- [ ] Am completat toate cele 5 exerciții obligatorii
- [ ] Am adăugat titluri și etichete la toate graficele
- [ ] Graficele mele sunt **interactive** (hover funcționează)
- [ ] Am salvat cel puțin un grafic ca HTML
- [ ] Am răspuns la toate întrebările de reflecție
- [ ] Codul meu rulează fără erori
- [ ] Am testat bonusurile (dacă am apucat)
- [ ] Am experimentat cu culori și layout-uri diferite

---

## 📦 Ce să Predai

Organizează-ți munca astfel:

```
task_plotly_[NumeTau]/
├── exercitiul_1_scatter.py
├── exercitiul_2_dashboard.py
├── exercitiul_3_boxplot.py
├── exercitiul_4_heatmap.py
├── exercitiul_5_export.py
├── my_interactive_chart.html (exportat)
└── raspunsuri_reflectii.md (întrebările de reflecție)
```

---

**🎉 Mult succes cu task-ul! 🎉**

> **Reminder:** Plotly e despre **interactivitate**! Asigură-te că folosești hover, zoom, și click pe legendă pentru a explora datele. Nu ezita să experimentezi cu parametri diferiți - asta e frumusețea visualizării interactive!
