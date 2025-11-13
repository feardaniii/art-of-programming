import pandas as pd
import numpy as np

# Creăm date reale de comenzi e-commerce
comenzi_data = {
    'data': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18',
             '2024-01-19', '2024-01-20', '2024-01-21', '2024-01-22',
             '2024-02-01', '2024-02-02', '2024-02-03', '2024-02-04'],
    'produs': ['Laptop', 'Mouse', 'Tastatură', 'Monitor',
               'Laptop', 'Mouse', 'Tastatură', 'Monitor',
               'Laptop', 'Mouse', 'Tastatură', 'Monitor'],
    'pret': [2500, 150, 200, 800,
             2500, 150, 200, 800,
             2500, 150, 200, 800],
    'cantitate': [2, 5, 3, 1,
                  1, 8, 4, 2,
                  3, 6, 5, 1],
    'oras': ['București', 'Cluj', 'București', 'Timișoara',
             'Cluj', 'București', 'Timișoara', 'București',
             'București', 'Cluj', 'Timișoara', 'Cluj'],
    'categorie': ['Electronics', 'Accessories', 'Accessories', 'Electronics',
                  'Electronics', 'Accessories', 'Accessories', 'Electronics',
                  'Electronics', 'Accessories', 'Accessories', 'Electronics']
}

df = pd.DataFrame(comenzi_data)
df['data'] = pd.to_datetime(df['data'])
df['valoare_totala'] = df['pret'] * df['cantitate']

print("=== PIVOT TABLES: DIN HAOS LA RAPOARTE ===\n")
print("Date brute (primele 5 rânduri):")
print(df.head())
print("\n" + "="*60 + "\n")

# 1. PIVOT BASIC - Vânzări pe oraș și produs
print("1. PIVOT TABLE - Vânzări totale pe Oraș × Produs:")
pivot1 = df.pivot_table(
    values='valoare_totala',
    index='oras',
    columns='produs',
    aggfunc='sum',
    fill_value=0
)
print(pivot1)
print()

# 2. MULTI-INDEX PIVOT - Oraș × Categorie cu statistici
print("2. PIVOT AVANSAT - Oraș × Categorie cu statistici multiple:")
pivot2 = df.pivot_table(
    values='valoare_totala',
    index='oras',
    columns='categorie',
    aggfunc=['sum', 'mean', 'count'],
    fill_value=0
)
print(pivot2)
print()

# 3. PIVOT CU MARGINI (TOTALS) - Raport executive
print("3. PIVOT CU TOTALS - Raport executive complet:")
pivot3 = df.pivot_table(
    values='valoare_totala',
    index='oras',
    columns='produs',
    aggfunc='sum',
    fill_value=0,
    margins=True,  # Adaugă totals
    margins_name='TOTAL'
)
print(pivot3)
print()

# 4. CROSS-TAB - Analiza cantităților
print("4. CROSS-TAB - Număr de comenzi pe Oraș × Produs:")
crosstab = pd.crosstab(
    df['oras'],
    df['produs'],
    values=df['cantitate'],
    aggfunc='sum',
    margins=True
)
print(crosstab)
print()

# 5. GROUPBY + PIVOT - Analiza complexă
print("5. ANALIZA COMPLEXĂ - Top produse pe oraș:")
grouped = df.groupby(['oras', 'produs']).agg({
    'valoare_totala': 'sum',
    'cantitate': 'sum',
    'data': 'count'
}).rename(columns={'data': 'nr_comenzi'})
print(grouped.sort_values('valoare_totala', ascending=False))
print()

# 6. INSIGHTS AUTOMATE
print("6. INSIGHTS AUTOMATE:")
print(f"   • Orașul cu cele mai mari vânzări: {pivot3.drop('TOTAL')['TOTAL'].idxmax()}")
print(f"   • Produsul best-seller: {df.groupby('produs')['valoare_totala'].sum().idxmax()}")
print(f"   • Valoarea medie pe comandă: {df['valoare_totala'].mean():.2f} lei")
print(f"   • Total vânzări: {df['valoare_totala'].sum():,.2f} lei")
print()

# 7. PIVOT PENTRU TIMELINE
print("7. ANALIZA TIMELINE - Vânzări zilnice:")
df['luna'] = df['data'].dt.to_period('M')
pivot_time = df.pivot_table(
    values='valoare_totala',
    index='luna',
    columns='produs',
    aggfunc='sum',
    fill_value=0
)
print(pivot_time)

print("\n🎉 Gata! Din haos ai creat rapoarte executive profesionale!")
