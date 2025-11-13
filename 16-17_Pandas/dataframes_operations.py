import pandas as pd

# Creăm DataFrame-ul cu comenzi e-commerce
comenzi_data = {
    'produs': ['Laptop', 'Mouse', 'Tastatură', 'Monitor', 'Laptop', 'Mouse'],
    'pret': [2500, 150, 200, 800, 2500, 150],
    'cantitate': [2, 5, 3, 1, 1, 8],
    'data': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19', '2024-01-20'],
    'oras': ['București', 'Cluj', 'București', 'Timișoara', 'Cluj', 'București']
}

df = pd.DataFrame(comenzi_data)
df['data'] = pd.to_datetime(df['data'])  # Convertim în datetime
df['valoare_totala'] = df['pret'] * df['cantitate']  # Calculăm valoarea

print("DataFrame-ul nostru e-commerce:")
print(df)
print("\n" + "="*60 + "\n")

# 1. FILTRARE - Produse scumpe (> 500 lei)
print("1. FILTRARE - Produse cu preț > 500 lei:")
produse_scumpe = df[df['pret'] > 500]
print(produse_scumpe[['produs', 'pret', 'oras']])
print()

# 2. GRUPARE - Vânzări pe orașe
print("2. GRUPARE - Total vânzări pe orașe:")
vanzari_oras = df.groupby('oras')['valoare_totala'].sum().sort_values(ascending=False)
print(vanzari_oras)
print()

# 3. STATISTICI - Descrierea datelor
print("3. STATISTICI - Analiza prețurilor:")
print(df['pret'].describe())
print()

# 4. SORTARE - Top comenzi după valoare
print("4. SORTARE - Top 3 comenzi după valoare:")
top_comenzi = df.nlargest(3, 'valoare_totala')
print(top_comenzi[['produs', 'valoare_totala', 'oras']])
print()

# BONUS: Analiza avansată
print("BONUS - Câte produse diferite pe oraș:")
print(df.groupby('oras')['produs'].nunique())

print("\n🎉 Gata! Ai învățat să analizezi date e-commerce ca un PRO!")