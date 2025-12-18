# 📊 PLOTLY INTERACTIVE VISUALIZATION - TASK

---

## 📚 Introduction

In this task you'll learn to create **interactive** visualizations with Plotly. Unlike matplotlib/seaborn (static), Plotly charts allow:
- 🎮 **Hover** for details
- 🔍 **Zoom** and pan
- 👁️ **Click on legend** to hide/show
- 💾 **Export** as PNG directly from browser

---

## 🎯 What is Plotly?

Plotly transforms your Python code into **HTML + JavaScript** interactive content:

```
Python Code → Plotly Figure → HTML + plotly.js → Browser
```

**Two main modes:**
- **Plotly Express (px)**: Fast, one-liner, perfect for beginners
- **Graph Objects (go)**: Complete control, more lines, for advanced customizations

---

## 🚀 Starting Code

Run this code to test your installation:
```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Quick test
fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6], title="🎉 Plotly Works!")
fig.show()  # Opens in browser
```

**💡 Note:** Plotly opens a temporary server in your browser. When you close the script, the server stops!

---

# PART 1: Plotly Express Fundamentals

---

## 📝 EXERCISE 1: Personal Scatter Plot - Age vs Experience

**Objective:** Create a scatter plot with data about yourself and friends/colleagues

### 📖 New concepts:

**`scatter plot`** = chart with points, perfect for seeing relationships between two numeric variables
**`hover_data`** = extra information that appears when you hover over a point
**`color`** = colors points based on a category (e.g., gender, department)

**TODO:**
1. Create a DataFrame with data about 10-15 people (yourself, friends, colleagues)
   - Columns: `Name`, `Age`, `Years_Experience`, `Estimated_Salary`, `Field`
2. Use `px.scatter()` to visualize `Age` (X) vs `Years_Experience` (Y)
3. Color the points by `Field` using the `color` parameter
4. Add `hover_data` to see `Name` and `Estimated_Salary` on hover
5. **BONUS:** Change point sizes based on `Estimated_Salary` with `size`

### 💻 Starting code:
```python
import plotly.express as px
import pandas as pd

# Create data (example - modify with your data!)
df_personal = pd.DataFrame({
    'Name': ['Alex', 'Maria', 'John', 'Anna', 'Mike', 'Elena'],
    'Age': [25, 28, 22, 30, 27, 24],
    'Years_Experience': [3, 5, 1, 7, 4, 2],
    'Estimated_Salary': [3000, 4500, 2500, 6000, 4000, 3200],
    'Field': ['IT', 'Marketing', 'IT', 'Finance', 'IT', 'HR']
})

# Create scatter plot
fig = px.scatter(
    df_personal,
    x='Age',
    y='Years_Experience',
    color='Field',  # Color by field
    hover_data=['Name', 'Estimated_Salary'],  # Info on hover
    title='📊 Age vs Experience - Friends Analysis'
)

# Customizations
fig.update_layout(
    xaxis_title="Age (years)",
    yaxis_title="Years of Experience",
    hovermode='closest'
)

fig.show()

# TODO: Add size='Estimated_Salary' for varied point sizes
```

**💡 Hint for BONUS:** Add `size='Estimated_Salary'` in `px.scatter()` and observe how larger points = higher salaries!

**❓ Reflection question:**
What correlation do you observe between age and experience? Is it linear?

_Your answer:_ _______________________________________________

---

## 📝 EXERCISE 2: 3-Chart Dashboard - Personal Budget

**Objective:** Create a mini financial dashboard with Bar, Line, and Pie charts

### 📖 New concepts:

**`make_subplots`** = creates grids with multiple charts in the same layout
**`add_trace`** = adds a chart to a subplot
**`row, col`** = specifies the subplot position in the grid

**TODO:**
1. Create data for 6 months: Income, Expenses, Savings
2. **Subplot 1 (Bar Chart):** Compare Income vs Expenses by month
3. **Subplot 2 (Line Chart):** Savings trend over time
4. **Subplot 3 (Pie Chart):** Expense breakdown by categories (Food, Rent, Transport, Entertainment)
5. Add a general title and customize the layout

### 💻 Starting code:
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Financial data (6 months)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
income = [3000, 3200, 3500, 3400, 3600, 3800]
expenses = [2500, 2700, 2800, 2600, 2900, 2700]
savings = [i - e for i, e in zip(income, expenses)]

# Expense breakdown for current month (June)
expense_categories = {
    'Food': 800,
    'Rent': 1200,
    'Transport': 300,
    'Entertainment': 400
}

# Create 2x2 grid (3 subplots used)
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('💰 Income vs Expenses', '📈 Savings Trend', '🥧 Expense Breakdown'),
    specs=[[{"type": "bar"}, {"type": "scatter"}],
           [{"type": "pie", "colspan": 2}, None]]  # Pie spans both columns
)

# SUBPLOT 1: Bar Chart - Income vs Expenses
fig.add_trace(
    go.Bar(x=months, y=income, name='Income', marker_color='#10b981'),
    row=1, col=1
)
fig.add_trace(
    go.Bar(x=months, y=expenses, name='Expenses', marker_color='#ef4444'),
    row=1, col=1
)

# SUBPLOT 2: Line Chart - Savings Trend
# TODO: Add a line chart with savings (use go.Scatter with mode='lines+markers')

# SUBPLOT 3: Pie Chart - Expense Breakdown
# TODO: Add a pie chart with expense_categories (use go.Pie)

# General layout
fig.update_layout(
    title_text="💼 Personal Financial Dashboard - 6 Months",
    title_x=0.5,
    height=700,
    showlegend=True
)

fig.show()
```

**💡 Hint:**
- Line chart: `go.Scatter(x=months, y=savings, mode='lines+markers', name='Savings')`
- Pie chart: `go.Pie(labels=list(expense_categories.keys()), values=list(expense_categories.values()))`

**❓ Reflection question:**
In which month did you save the most? What insights does the dashboard provide?

_Your answer:_ _______________________________________________

---

# PART 2: Advanced Visualizations

---

## 📝 EXERCISE 3: Box Plot - Team Performance

**Objective:** Compare score distribution for 4 sports/work teams

### 📖 New concepts:

**`box plot`** = shows data distribution: median, quartiles, outliers
**`points='all'`** = displays all individual points over the box
**`notched=True`** = adds "notch" for median confidence interval

**TODO:**
1. Generate scores for 4 teams (20-30 scores per team)
2. Create a box plot with `px.box()` or `go.Box()`
3. Display individual points to see complete distribution
4. Customize colors for each team
5. **BONUS:** Add a horizontal line for "target score" (e.g., 75 points)

### 💻 Starting code:
```python
import plotly.express as px
import numpy as np
import pandas as pd

# Generate scores for 4 teams
np.random.seed(42)
teams = []
scores = []

for team, mean_score in [('Team A', 80), ('Team B', 70), ('Team C', 85), ('Team D', 65)]:
    team_scores = np.random.normal(mean_score, 10, 25)  # 25 scores
    teams.extend([team] * 25)
    scores.extend(team_scores)

df_scores = pd.DataFrame({
    'Team': teams,
    'Score': scores
})

# Create box plot
fig = px.box(
    df_scores,
    x='Team',
    y='Score',
    title='📦 Score Distribution by Teams',
    points='all',  # Display all points
    color='Team'  # Different colors per team
)

# TODO: Add a horizontal line for target score (75)
# Hint: fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Target")

fig.update_layout(
    yaxis_title="Score (points)",
    xaxis_title="Team"
)

fig.show()
```

**💡 Hint for BONUS:** Use `fig.add_hline(y=75, line_dash="dash", line_color="red")` for the target line.

**❓ Reflection question:**
Which team has the most consistent performance (small variation)? Which has outliers?

_Your answer:_ _______________________________________________

---

## 📝 EXERCISE 4: Heatmap - 10x10 Correlation Matrix

**Objective:** Visualize a data matrix as a colored heatmap

### 📖 New concepts:

**`heatmap`** = color map where color intensity = numeric value
**`colorscale`** = color scheme (e.g., 'Viridis', 'RdBu', 'Blues')
**`text`** = displays values in each cell
**`texttemplate`** = formatting for text (e.g., '.2f' for 2 decimals)

**TODO:**
1. Generate a 10x10 matrix with random values between 0 and 100
2. Create labels for rows and columns (e.g., "Var1", "Var2", ...)
3. Use `go.Heatmap()` to visualize the matrix
4. Add `text` with values and `texttemplate` for formatting
5. **BONUS:** Calculate and display a real correlation matrix (generate 10 correlated variables)

### 💻 Starting code:
```python
import plotly.graph_objects as go
import numpy as np

# Generate random 10x10 matrix
np.random.seed(42)
matrix = np.random.randint(0, 100, size=(10, 10))

# Labels
labels = [f'Var{i+1}' for i in range(10)]

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=matrix,
    x=labels,  # Columns
    y=labels,  # Rows
    colorscale='Viridis',  # Color scheme
    text=matrix,  # Display values
    texttemplate='%{text}',  # Format
    textfont={"size": 10},
    colorbar=dict(title="Value")
))

fig.update_layout(
    title='🔥 10x10 Heatmap - Random Matrix',
    xaxis_title="Variables (Columns)",
    yaxis_title="Variables (Rows)",
    width=700,
    height=700
)

fig.show()

# TODO BONUS: Generate 10 correlated variables and calculate correlation matrix
```

**💡 Hint for BONUS:**
```python
# Generate 10 variables with correlations
data = np.random.randn(100, 10)  # 100 observations, 10 variables
df_corr = pd.DataFrame(data, columns=labels)
corr_matrix = df_corr.corr()  # Calculate correlations

# Then use corr_matrix.values for z in heatmap
# Change colorscale='RdBu' for blue-red (perfect for correlations -1 to 1)
```

**❓ Reflection question:**
In the correlation matrix case (BONUS), which pairs of variables are strongly correlated?

_Your answer:_ _______________________________________________

---

# PART 3: Real-World Application

---

## 📝 EXERCISE 5: Save and Share - HTML Export

**Objective:** Learn to save charts for sharing (email, presentations)

### 📖 New concepts:

**`.write_html()`** = saves the chart as a standalone HTML file (works offline!)
**`.to_json()`** = converts to JSON for integration in web apps
**`include_plotlyjs`** = controls how plotly.js is loaded

**TODO:**
1. Take one of your charts from above (scatter, dashboard, or heatmap)
2. Save it as an HTML file using `.write_html()`
3. Open the file in a browser and verify interactivity
4. **BONUS:** Export as JSON and inspect the structure

### 💻 Starting code:
```python
# Take the figure from Exercise 1 (or any other figure)
# fig = ... (your figure)

# Save as HTML
fig.write_html("my_interactive_chart.html")
print("✅ Chart saved as 'my_interactive_chart.html'")
print("📂 Open the file manually in your browser!")

# Save with advanced options
fig.write_html(
    "my_chart_standalone.html",
    include_plotlyjs='cdn',  # Load plotly.js from internet (smaller file)
    config={'displayModeBar': True}  # Display tools bar
)

# TODO BONUS: Export as JSON
import json
chart_json = fig.to_json()
with open('my_chart.json', 'w') as f:
    json.dump(json.loads(chart_json), f, indent=2)
print("✅ Chart JSON saved as 'my_chart.json'")
```

**💡 Notes:**
- The HTML file works **offline** (you can send it via email!)
- All interactive features (hover, zoom, pan) work
- JSON is useful for integration in React/Vue/Flask apps

**❓ Reflection question:**
When would you use `.write_html()` vs `.show()` in real projects?

_Your answer:_ _______________________________________________

---

# 🎓 Final Reflection Questions

After completing the exercises, answer the following questions:

### 1. What major differences did you observe between Plotly and matplotlib/seaborn?

_Your answer:_
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

### 2. Why do you think interactivity is important in data visualization?

_Your answer:_
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

### 3. When would you use Plotly Express (px) vs Graph Objects (go)?

_Your answer:_
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

### 4. What type of real project could you build using Plotly?

_Your answer:_
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

---

# 💡 Useful Resources

- 📚 **Plotly Documentation:** https://plotly.com/python/
- 📊 **Example Gallery:** https://plotly.com/python/plotly-express/
- 🎨 **Customization:** https://plotly.com/python/styling-plotly-express/
- 🌐 **Dash (interactive apps):** https://dash.plotly.com/
- 🔥 **Cheat Sheet:** https://images.plot.ly/plotly-documentation/images/python_cheat_sheet.pdf

---

# 🔥 EXTRA CHALLENGE (optional)

If you've finished all exercises and want to practice more:

### 🚀 Challenge 1: Interactive Dashboard with Dash
Transform the dashboard from Exercise 2 into a Dash application with:
- Dropdown to select month
- Slider to filter expense categories
- Chart that updates automatically (callbacks!)

**Starter code:**
```python
# pip install dash
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

app = Dash(__name__)

app.layout = html.Div([
    html.H1("💼 Interactive Financial Dashboard"),
    dcc.Dropdown(
        id='month-dropdown',
        options=[{'label': m, 'value': m} for m in months],
        value='Jun'
    ),
    dcc.Graph(id='expense-chart')
])

@app.callback(
    Output('expense-chart', 'figure'),
    Input('month-dropdown', 'value')
)
def update_chart(selected_month):
    # TODO: Create chart based on selected month
    pass

if __name__ == '__main__':
    app.run(debug=True)
```

### 📊 Challenge 2: Animated Chart
Create an animated scatter plot showing evolution over time:
```python
# Example: Salary evolution over time
fig = px.scatter(
    df_time_series,
    x='Age',
    y='Salary',
    animation_frame='Year',  # Animation by years
    color='Department',
    size='Experience',
    title='📈 Salary Evolution 2020-2024'
)
fig.show()
```

### 🗺️ Challenge 3: Geographic Map
Display data on a map (e.g., average salaries by counties):
```python
fig = px.choropleth(
    df_counties,
    locations='County',
    color='Average_Salary',
    title='🗺️ Average Salaries by County'
)
```

---

## ✅ Final Checklist

Before submitting the task, verify:

- [ ] I've completed all 5 mandatory exercises
- [ ] I've added titles and labels to all charts
- [ ] My charts are **interactive** (hover works)
- [ ] I've saved at least one chart as HTML
- [ ] I've answered all reflection questions
- [ ] My code runs without errors
- [ ] I've tested the bonuses (if I had time)
- [ ] I've experimented with different colors and layouts

---

## 📦 What to Submit

Organize your work as follows:

```
task_plotly_[YourName]/
├── exercise_1_scatter.py
├── exercise_2_dashboard.py
├── exercise_3_boxplot.py
├── exercise_4_heatmap.py
├── exercise_5_export.py
├── my_interactive_chart.html (exported)
└── reflection_answers.md (reflection questions)
```

---

**🎉 Good luck with the task! 🎉**

> **Reminder:** Plotly is about **interactivity**! Make sure you use hover, zoom, and click on legend to explore the data. Don't hesitate to experiment with different parameters - that's the beauty of interactive visualization!