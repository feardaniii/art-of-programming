import pandas as pd
import numpy as np

# Creăm DataFrame-ul cu date problematice
data = {
    'id': [1, 2, 3, 4, 5],
    'nume': ['Alex', None, 'Maria', 'Ion', 'Ana'],
    'varsta': [28, 34, None, 45, 29],
    'salariu': [None, 4500, 5200, 3800, 4200],
    'oras': ['București', 'Cluj', '', 'Timișoara', 'București']
}
df = pd.DataFrame(data)

print("Date brute:")
print(df)
print("\n" + "="*50 + "\n")

# Pasul 1: Identificăm valorile lipsă
print("Valorile lipsă pe coloane:")
print(df.isnull().sum())
print()

# Pasul 2: Curățăm datele
# Completăm numele lipsă
df['nume'].fillna('Necunoscut', inplace=True)

# Completăm vârsta cu media
varsta_medie = df['varsta'].mean()
df['varsta'].fillna(varsta_medie, inplace=True)

# Completăm salariul cu media
salariu_mediu = df['salariu'].mean()
df['salariu'].fillna(salariu_mediu, inplace=True)

# Completăm orașul lipsă
df['oras'].replace('', 'București', inplace=True)

print("Date curate:")
print(df)
print("\nGata! Datele sunt curate și ready pentru analiză! 🎉")