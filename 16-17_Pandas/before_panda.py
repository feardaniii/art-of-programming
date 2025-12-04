import pandas as pd

djblue_srl = {
    'id': [1, 2, 3, 4, 5],
    'nume': ['Alex', None, 'Maria', 'Ion', 'Ana'],
    'varsta': [28, 34, None, 45, 29],
    'salariu': [None, 4500, 5200, 3800, 4200],
    'oras': ['București', 'Cluj', '', 'Timișoara', 'București']
}


# Scriere
df = pd.DataFrame(djblue_srl)
df.to_excel("output.xlsx", index=False)

# 📝 Completez numele lipsă cu 'Necunoscut'...
# 📊 Înlocuiesc vârsta lipsă cu media...
# 🏙️ Completez orașul lipsă cu 'București'...




# Citire
# df_csv = pd.read_csv("input.csv")
df_excel = pd.read_excel("input.xlsx")

print(df_excel)
