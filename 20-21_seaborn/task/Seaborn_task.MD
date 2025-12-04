# 📊 SEABORN DATA VISUALIZATION - TASK

---

## 📚 Introducere

În acest task vei învăța să folosești Seaborn pentru vizualizări avansate de date. Vei lucra atât cu dataset-uri predefinite, cât și cu date generate sintetic.

---

## 🎯 Ce sunt dataseturile Seaborn?

Seaborn vine cu mai multe dataset-uri incluse, perfecte pentru învățare:

- **tips**: bacșișuri la restaurant (total_bill, tip, sex, smoker, day, time, size)
- **iris**: măsurători flori iris (sepal_length, sepal_width, petal_length, petal_width, species)
- **penguins**: date despre pinguini (species, island, bill_length_mm, flipper_length_mm, etc.)
- **titanic**: pasagerii de pe Titanic (survived, pclass, sex, age, fare, etc.)
- **diamonds**: caracteristici diamante (carat, cut, color, clarity, price, etc.)

### 📌 Cum să încarci un dataset:
```python
df = sns.load_dataset("tips")  # înlocuiește "tips" cu numele dorit
```

### 📌 Cum să vezi ce dataset-uri sunt disponibile:
```python
print(sns.get_dataset_names())
```

---

## 🚀 Cod de Start

Rulează acest cod pentru a începe:
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Exemplu de încărcare
df = sns.load_dataset("tips")
print("🔍 Primele 5 rânduri din dataset-ul 'tips':")
print(df.head())
print("\n📋 Informații despre coloane:")
print(df.info())
```

---

# PARTEA 1: Explorarea Dataseturilor Seaborn

---

## 📝 EXERCIȚIUL 1: Histogramă cu alt dataset

**Obiectiv:** Vizualizează distribuția vârstei pasagerilor de pe Titanic

### 📖 Concepte noi:

**`hue`** = parametru care colorează graficul pe baza unei coloane categorice (ex: separare pe "survived" - supraviețuitori vs. nesupraviețuitori)

**TODO:**
1. Încarcă dataset-ul "titanic" folosind `sns.load_dataset()`
2. Folosește `sns.histplot()` pentru a vizualiza distribuția vârstei (age)
3. Adaugă titlu și etichete axelor
4. **BONUS:** Separă histograma pe `survived` folosind parametrul `hue`

### 💻 Cod de pornire:
```python
# Încarcă datasetul
df_titanic = sns.load_dataset("titanic")

# Creează histograma
sns.histplot(data=df_titanic, x="age")
plt.title("Distribuția vârstei pasagerilor Titanic")
plt.xlabel("Vârsta")
plt.ylabel("Număr de pasageri")
plt.show()

# BONUS: Adaugă hue="survived" pentru a separa pe supraviețuitori
```

**💡 Hint pentru BONUS:** Adaugă parametrul `hue="survived"` în `sns.histplot()` și observă diferența!

---

## 📝 EXERCIȚIUL 2: Pairplot pentru Iris

**Obiectiv:** Explorează relațiile dintre variabilele florii Iris

### 📖 Concepte noi:

**`pairplot`** = creează o matrice de grafice care arată relațiile între toate perechile de variabile numerice

**TODO:**
1. Încarcă dataset-ul "iris"
2. Creează un pairplot folosind `sns.pairplot()`
3. Colorează punctele după `species` (hint: parametrul `hue`)
4. **Observă:** ce relații interesante vezi între variabile?

### 💻 Cod de pornire:
```python
# Încarcă datasetul
df_iris = sns.load_dataset("iris")

```

**❓ Întrebare de reflecție:**  
Ce perechi de variabile sunt cel mai puternic corelate?

_Răspunsul tău:_ _______________________________________________

---

## 📝 EXERCIȚIUL 3: Heatmap de corelații pentru Penguins

**Obiectiv:** Vizualizează corelațiile dintre măsurătorile pinguinilor

### 📖 Concepte noi:

**`heatmap`** = hartă de culoare care arată intensitatea valorilor într-o matrice (perfect pentru corelații)  
**`annot=True`** = afișează valorile numerice în fiecare celulă  
**`cmap`** = schema de culori (ex: "coolwarm" = albastru-roșu)

**TODO:**
1. Încarcă dataset-ul "penguins"
2. Calculează matricea de corelații: `df.corr(numeric_only=True)`
3. Creează o heatmap folosind `sns.heatmap()`
4. Adaugă `annot=True` pentru a vedea valorile
5. **BONUS:** Folosește `cmap="coolwarm"` pentru culori mai clare

### 💻 Cod de pornire:
```python
# Încarcă datasetul
df_penguins = sns.load_dataset("penguins")

# Calculează corelațiile
correlation_matrix = df_penguins.corr(numeric_only=True)
```

---

# PARTEA 2: Lucrul cu Date Generate

---

## 🎲 Generăm un Dataset Sintetic

Rulează acest cod pentru a genera date despre angajați:
```python
# Generăm un dataset sintetic cu Pandas
np.random.seed(42)
df_employees = pd.DataFrame({
    "sex": np.random.choice(["Male", "Female"], size=100),
    "age": np.random.normal(35, 10, size=100),
    "salary": np.random.normal(5000, 1500, size=100),
    "department": np.random.choice(["IT", "HR", "Marketing", "Sales"], size=100),
    "experience": np.random.randint(1, 15, 100),
})

print("🔍 Primele 5 rânduri din dataset-ul generat:")
print(df_employees.head())
print("\n📊 Statistici descriptive:")
print(df_employees.describe())
```

---

## 📝 EXERCIȚIUL 4: Violinplot pentru salariu și experiență

**Obiectiv:** Analizează distribuția salariilor în funcție de experiență și sex

### 📖 Concepte noi:

**`violinplot`** = combină boxplot cu distribuția densității (arată forma distribuției)  
**`split=True`** = când ai `hue`, împarte fiecare violin în două părți pentru comparație

**TODO:**
1. Creează categorii de experiență (Junior: 1-5 ani, Mid: 6-10 ani, Senior: 11-15 ani)
2. Creează un violinplot cu `sns.violinplot()`
3. Axă X: categorii de experiență
4. Axă Y: `salary`
5. Folosește `split=True` și `hue="sex"` pentru a separa pe sexe

### 💻 Cod de pornire:
```python
# Creează categorii de experiență
df_employees['exp_category'] = pd.cut(
    df_employees['experience'], 
    bins=[0, 5, 10, 15], 
    labels=['Junior (1-5)', 'Mid (6-10)', 'Senior (11-15)']
)

# Creează violinplot

# Adaugă hue="sex" și split=True pentru a compara bărbați vs femei
```

---

## 📝 EXERCIȚIUL 5: FacetGrid pentru departamente și sexe

**Obiectiv:** Compară distribuția salariilor pe departamente și sexe

### 📖 Concepte noi:

**`FacetGrid`** = creează multiple subgrafice (facete) bazate pe valori categorice  
**`col`** = parametru care creează coloane separate pentru fiecare valoare  
**`.map()`** = aplică o funcție de plot pe fiecare facetă

**TODO:**
1. Creează un `FacetGrid` cu `col="department"`, `hue="sex"`
2. Mapează un `histplot` pentru `salary`
3. Adaugă legendă cu `.add_legend()`
4. **Observă:** există diferențe de salariu între departamente?

### 💻 Cod de pornire:
```python
# Creează FacetGrid
g = sns.FacetGrid(df_employees, col="department", height=4)
g.map(sns.histplot, "salary")
plt.show()

# TODO: Adaugă hue="sex" la FacetGrid și .add_legend() la final
```

**❓ Întrebare de reflecție:**  
Ce departament pare să aibă cele mai mari salarii?

_Răspunsul tău:_ _______________________________________________

---

## 📝 EXERCIȚIUL 6: Barplot pentru compararea salariilor

**Obiectiv:** Compară salariul mediu între angajații tineri și cei mai experimentați

### 📖 Concepte noi:

**`barplot`** = afișează valoarea medie (sau altă statistică) pentru fiecare categorie  
**`apply()`** = aplică o funcție pe fiecare element dintr-o coloană

**TODO:**
1. Creează o coloană nouă `"age_group"` care clasifică angajații:
   - `"Sub 30 ani"` dacă `age < 30`
   - `"Peste 30 ani"` dacă `age >= 30`
2. Creează un barplot cu `sns.barplot()`
3. Axă X: `age_group`
4. Axă Y: `salary` (va calcula automat media)
5. **BONUS:** Adaugă `hue="sex"` pentru a compara și pe sexe

### 💻 Cod de pornire:
```python
# Creează grupuri de vârstă
df_employees['age_group'] = df_employees['age'].apply(
    lambda x: 'Sub 30 ani' if x < 30 else 'Peste 30 ani'
)

# Creează barplot
sns.barplot(data=df_employees, x="age_group", y="salary")
plt.title("Salariul mediu pe grupe de vârstă")
plt.xlabel("Grupa de vârstă")
plt.ylabel("Salariu mediu")
plt.show()

# TODO BONUS: Adaugă hue="sex" pentru a compara bărbați vs femei
```

---

# 🎓 Întrebări de Reflecție

După ce ai terminat exercițiile, răspunde la următoarele întrebări:

### 1. Ce diferență ai observat între histplot și violinplot?

_Răspunsul tău:_  
_______________________________________________________________  
_______________________________________________________________  
_______________________________________________________________

### 2. Ce tip de relații poți identifica mai ușor cu un pairplot?

_Răspunsul tău:_  
_______________________________________________________________  
_______________________________________________________________  
_______________________________________________________________

### 3. La ce este util un FacetGrid în comparație cu un plot simplu?

_Răspunsul tău:_  
_______________________________________________________________  
_______________________________________________________________  
_______________________________________________________________

### 4. Ce ai învățat din analiza dataset-ului de angajați generat?

_Răspunsul tău:_  
_______________________________________________________________  
_______________________________________________________________  
_______________________________________________________________

---

# 💡 Resurse Utile

- 📚 **Documentație Seaborn:** https://seaborn.pydata.org/
- 📊 **Galerie de exemple:** https://seaborn.pydata.org/examples/index.html
- 🎨 **Palete de culori:** https://seaborn.pydata.org/tutorial/color_palettes.html
- 📖 **Tutorial oficial:** https://seaborn.pydata.org/tutorial.html

---

# 🔥 PROVOCARE EXTRA (opțional)

Dacă ai terminat toate exercițiile și vrei să exersezi mai mult:

1. **Combină vizualizările:** Creează o figură cu subplots care să conțină mai multe tipuri de plot-uri
2. **Explorează stilurile:** Încearcă diferite teme cu `sns.set_style()` (darkgrid, whitegrid, dark, white, ticks)
3. **Salvează vizualizările:** Folosește `plt.savefig("nume_fisier.png", dpi=300, bbox_inches='tight')`
4. **Analizează relații complexe:** Folosește `sns.jointplot()` pentru a combina scatter plots cu distribuții marginale
5. **Personalizează culorile:** Explorează paletele custom cu `sns.color_palette()`

---

## ✅ Checklist Final

Înainte de a preda task-ul, verifică:

- [ ] Am completat toate cele 6 exerciții
- [ ] Am adăugat titluri și etichete la toate graficele
- [ ] Am răspuns la întrebările de reflecție
- [ ] Codul meu rulează fără erori
- [ ] Am înțeles ce reprezintă fiecare vizualizare
- [ ] Am testat bonusurile (dacă am apucat)

---

**🎉 Mult succes cu task-ul! 🎉**

> **Reminder:** Dacă întâmpini probleme, revizitează documentația Seaborn și exemplele din curs. 
Nu ezita să experimentezi cu parametri diferiți!