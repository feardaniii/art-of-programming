import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=== PIPELINE AUTOMAT DE PROCESARE DATE ===\n")

# STEP 1: Simulăm citirea unui CSV "murdar"
print("📄 STEP 1: Citesc fișierul CSV cu date problematice...")
raw_data = {
    'id': [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10],  # 5 duplicat!
    'nume': ['Alex', None, 'Maria', 'Ion', 'Ana', 'Ana', 'George', '', 'Diana', 'Paul', 'Laura'],
    'email': ['alex@ex.com', 'invalid', 'maria@ex.com', 'ion@ex.com', None,
              'ana@ex.com', 'george@ex.com', 'test@ex.com', None, 'paul@ex.com', 'laura@ex.com'],
    'varsta': [28, 34, None, 45, 29, 29, -5, 150, 32, None, 27],  # Valori invalide!
    'salariu': [None, 4500, 5200, 3800, 4200, 4200, 3000, 5500, None, 4800, 5000],
    'oras': ['București', 'Cluj', '', 'Timișoara', 'bucurești', 'București',
             'cluj', 'Timișoara', 'Iași', 'București', ''],
    'data_angajare': ['2020-01-15', '2019-05-20', '2021-03-10', 'invalid',
                      '2020-08-15', '2020-08-15', '2022-01-01', '2018-12-01',
                      '2021-06-15', None, '2023-02-01']
}

df = pd.DataFrame(raw_data)
initial_rows = len(df)
print(f"Date citite: {initial_rows} rânduri, {len(df.columns)} coloane")
print("\nPrimele 5 rânduri (RAW):")
print(df.head())
print("\n" + "="*60 + "\n")

# STEP 2: Curățarea datelor
print("🧹 STEP 2: Curăț datele lipsă și duplicate...")

# 2.1: Elimină duplicate EXACTE
df_cleaned = df.drop_duplicates()
duplicates_removed = initial_rows - len(df_cleaned)
print(f"   ✓ {duplicates_removed} duplicate eliminate")

# 2.2: Tratează valorile lipsă
# Nume lipsă
df_cleaned['nume'].fillna('Necunoscut', inplace=True)
df_cleaned['nume'].replace('', 'Necunoscut', inplace=True)

# Email lipsă
df_cleaned['email'].fillna('unknown@example.com', inplace=True)

# Vârstă lipsă (cu media)
varsta_medie = df_cleaned['varsta'].mean()
df_cleaned['varsta'].fillna(varsta_medie, inplace=True)

# Salariu lipsă (cu media)
salariu_mediu = df_cleaned['salariu'].mean()
df_cleaned['salariu'].fillna(salariu_mediu, inplace=True)

# Oraș lipsă
df_cleaned['oras'].replace('', 'Necunoscut', inplace=True)

missing_filled = df.isnull().sum().sum()
print(f"   ✓ {missing_filled} valori lipsă completate")
print()

# STEP 3: Validarea și normalizarea datelor
print("🔢 STEP 3: Convertesc tipurile de date și validez...")

# 3.1: Normalizează oraș (lowercase inconsistencies)
df_cleaned['oras'] = df_cleaned['oras'].str.title()

# 3.2: Validează vârsta (trebuie între 18 și 70)
invalid_ages = df_cleaned[(df_cleaned['varsta'] < 18) | (df_cleaned['varsta'] > 70)]
print(f"   ⚠️  {len(invalid_ages)} vârste invalide detectate")
df_cleaned.loc[(df_cleaned['varsta'] < 18) | (df_cleaned['varsta'] > 70), 'varsta'] = varsta_medie
print(f"   ✓ Vârste invalide înlocuite cu media: {varsta_medie:.1f}")

# 3.3: Convertește data_angajare în datetime
df_cleaned['data_angajare'] = pd.to_datetime(df_cleaned['data_angajare'], errors='coerce')
invalid_dates = df_cleaned['data_angajare'].isnull().sum()
print(f"   ✓ Date convertite în format datetime ({invalid_dates} invalide → NaT)")
print()

# STEP 4: Feature Engineering
print("📊 STEP 4: Calculez metrici noi...")

# Calculează ani de experiență
df_cleaned['ani_experienta'] = (datetime.now() - df_cleaned['data_angajare']).dt.days / 365.25
df_cleaned['ani_experienta'] = df_cleaned['ani_experienta'].fillna(0).round(1)

# Calculează salariu pe an experiență
df_cleaned['salariu_per_exp'] = (df_cleaned['salariu'] /
                                  df_cleaned['ani_experienta'].replace(0, 1)).round(2)

print(f"   ✓ Adăugate 2 coloane noi: ani_experienta, salariu_per_exp")
print()

# STEP 5: Statistici finale
print("📈 STEP 5: Generez statistici...")
stats = {
    'Total angajați': len(df_cleaned),
    'Vârsta medie': df_cleaned['varsta'].mean(),
    'Salariu mediu': df_cleaned['salariu'].mean(),
    'Experiență medie': df_cleaned['ani_experienta'].mean(),
    'Orașe unice': df_cleaned['oras'].nunique()
}

for key, value in stats.items():
    if isinstance(value, float):
        print(f"   • {key}: {value:.2f}")
    else:
        print(f"   • {key}: {value}")
print()

# STEP 6: Export rezultate
print("💾 STEP 6: Export rezultate...")
print("\nDate CURATE (primele 5 rânduri):")
print(df_cleaned[['nume', 'varsta', 'salariu', 'oras', 'ani_experienta']].head())
print()

# Salvează în CSV (opțional - decomentează dacă vrei să salvezi)
# df_cleaned.to_csv('date_curate.csv', index=False)
# print("   ✓ Fișier salvat: date_curate.csv")

print("\n" + "="*60)
print("✅ PIPELINE COMPLET! Date prelucrate și gata de analiză!")
print("="*60)

# BONUS: Raport rezumativ
print("\n📋 RAPORT FINAL:")
print(f"   Rânduri procesate: {initial_rows} → {len(df_cleaned)}")
print(f"   Duplicate eliminate: {duplicates_removed}")
print(f"   Valori invalide corectate: {len(invalid_ages) + invalid_dates}")
print(f"   Calitatea datelor: {(len(df_cleaned) / initial_rows * 100):.1f}%")
