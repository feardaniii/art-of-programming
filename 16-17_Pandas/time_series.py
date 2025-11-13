import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Creăm date reale de vânzări lunare
dates = pd.date_range('2024-01-01', periods=12, freq='M')
vanzari_data = {
    'data': dates,
    'vanzari': [25000, 28000, 32000, 29000, 35000, 38000,
                42000, 39000, 45000, 48000, 51000, 55000],
    'comenzi': [120, 135, 150, 140, 165, 175,
                190, 180, 210, 220, 235, 250],
    'oras': ['București', 'Cluj', 'Timișoara', 'București', 'Cluj', 'București',
             'Timișoara', 'București', 'Cluj', 'București', 'Timișoara', 'Cluj']
}

df = pd.DataFrame(vanzari_data)
df.set_index('data', inplace=True)

print("=== ANALIZA TRENDURILOR - TIME SERIES ===\n")

# 1. RESAMPLE - Agregare pe perioade
print("1. RESAMPLE - Vânzări pe trimestre:")
quarterly = df['vanzari'].resample('Q').sum()
print(quarterly)
print()

# 2. ROLLING WINDOW - Media mobilă (trend smoothing)
print("2. ROLLING WINDOW - Media mobilă pe 3 luni:")
df['vanzari_ma3'] = df['vanzari'].rolling(window=3).mean()
print(df[['vanzari', 'vanzari_ma3']].head(6))
print()

# 3. TREND DETECTION - Creștere procentuală
print("3. TREND DETECTION - Creștere lunară (%):")
df['crestere_pct'] = df['vanzari'].pct_change() * 100
print(df[['vanzari', 'crestere_pct']].head(6))
print()

# 4. STATISTICI DESCRIPTIVE
print("4. STATISTICI COMPLETE:")
print(df['vanzari'].describe())
print()

# 5. IDENTIFICARE PEAK și LOW
print("5. IDENTIFICARE PEAK ȘI LOW:")
peak_month = df['vanzari'].idxmax()
low_month = df['vanzari'].idxmin()
print(f"Luna de vârf: {peak_month.strftime('%B %Y')} - {df['vanzari'].max():,} lei")
print(f"Luna cea mai slabă: {low_month.strftime('%B %Y')} - {df['vanzari'].min():,} lei")
print()

# 6. CUMULATIVE SUM - Vânzări cumulative
print("6. VÂNZĂRI CUMULATIVE:")
df['vanzari_cumulative'] = df['vanzari'].cumsum()
print(f"Total an: {df['vanzari_cumulative'].iloc[-1]:,} lei")
print()

# 7. SEASONAL ANALYSIS - Analiza pe orașe
print("7. ANALIZA PE ORAȘE:")
vanzari_oras = df.groupby('oras')['vanzari'].agg(['sum', 'mean', 'count'])
print(vanzari_oras)
print()

print("✅ Gata! Ai învățat să analizezi trenduri ca un data scientist PRO!")
