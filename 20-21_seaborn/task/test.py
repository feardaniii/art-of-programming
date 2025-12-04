import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ==========================================
# 🚀 COD DE START
# ==========================================
# Exemplu de încărcare
df = sns.load_dataset("tips")
print("🔍 Primele 5 rânduri din dataset-ul 'tips':")
print(df.head())
print("\n📋 Informații despre coloane:")
print(df.info())


# ==========================================
# PARTEA 1: Explorarea Dataseturilor Seaborn
# ==========================================

# --- EXERCIȚIUL 1: Histogramă Titanic ---
print("\n--- Exercițiul 1 ---")
df_titanic = sns.load_dataset("titanic")

plt.figure(figsize=(10, 6))
# Include BONUS: hue="survived"
sns.histplot(data=df_titanic, x="age", hue="survived", multiple="stack")
plt.title("Distribuția vârstei pasagerilor Titanic (Supraviețuitori vs. Decedați)")
plt.xlabel("Vârsta")
plt.ylabel("Număr de pasageri")
plt.show()


# --- EXERCIȚIUL 2: Pairplot Iris ---
print("\n--- Exercițiul 2 ---")
df_iris = sns.load_dataset("iris")

# Include BONUS: hue="species"
sns.pairplot(df_iris, hue="species")
plt.show()


# --- EXERCIȚIUL 3: Heatmap Penguins ---
print("\n--- Exercițiul 3 ---")
df_penguins = sns.load_dataset("penguins")

# Calculează corelațiile
correlation_matrix = df_penguins.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
# Include BONUS: cmap="coolwarm" și annot=True
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Matrice de corelație - Pinguini")
plt.show()


# ==========================================
# PARTEA 2: Lucrul cu Date Generate
# ==========================================

# --- GENERARE DATASET SINTETIC ---
np.random.seed(42)
df_employees = pd.DataFrame({
    "sex": np.random.choice(["Male", "Female"], size=100),
    "age": np.random.normal(35, 10, size=100),
    "salary": np.random.normal(5000, 1500, size=100),
    "department": np.random.choice(["IT", "HR", "Marketing", "Sales"], size=100),
    "experience": np.random.randint(1, 15, 100),
})

print("\n🔍 Primele 5 rânduri din dataset-ul generat:")
print(df_employees.head())
print("\n📊 Statistici descriptive:")
print(df_employees.describe())


# --- EXERCIȚIUL 4: Violinplot ---
print("\n--- Exercițiul 4 ---")
# Creează categorii de experiență
df_employees['exp_category'] = pd.cut(
    df_employees['experience'], 
    bins=[0, 5, 10, 15], 
    labels=['Junior (1-5)', 'Mid (6-10)', 'Senior (11-15)']
)

plt.figure(figsize=(10, 6))
# Include BONUS: hue="sex" și split=True
sns.violinplot(data=df_employees, x="exp_category", y="salary", hue="sex", split=True)
plt.title("Distribuția salariilor în funcție de experiență și sex")
plt.xlabel("Nivel de experiență")
plt.ylabel("Salariu")
plt.show()


# --- EXERCIȚIUL 5: FacetGrid ---
print("\n--- Exercițiul 5 ---")
# Include BONUS: hue="sex"
g = sns.FacetGrid(df_employees, col="department", hue="sex", height=4)
g.map(sns.histplot, "salary")
g.add_legend()
plt.show()


# --- EXERCIȚIUL 6: Barplot ---
print("\n--- Exercițiul 6 ---")
# Creează grupuri de vârstă
df_employees['age_group'] = df_employees['age'].apply(
    lambda x: 'Sub 30 ani' if x < 30 else 'Peste 30 ani'
)

plt.figure(figsize=(8, 6))
# Include BONUS: hue="sex"
sns.barplot(data=df_employees, x="age_group", y="salary", hue="sex")
plt.title("Salariul mediu pe grupe de vârstă și sex")
plt.xlabel("Grupa de vârstă")
plt.ylabel("Salariu mediu")
plt.show()