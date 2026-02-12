import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt 
import os



# =============================================
# PASUL 1: Citirea fișierului
# =============================================
# Calea absolută bazată pe locația scriptului
script_dir = os.path.dirname(os.path.abspath(__file__))
cale_articol = os.path.join(script_dir, "articol_stiri.txt")


with open(cale_articol, "r", encoding="utf-8") as f:
    text_brut = f.read()
 
print("=== TEXT ORIGINAL ===")
print(text_brut[:200], '...')
print(f"\nLungime text brut: {len(text_brut)} caractere")
print(f"Număr cuvinte brute: {len(text_brut.split())}")

# =============================================
# PASUL 2: Curățarea textului
# =============================================
def curata_text(text):
    """Curăță textul: lowercase, elimină caractere speciale."""
    text = text.lower()
    # Eliminăm numere
    text = re.sub(r"\d+", "", text)
    # Păstrăm doar litere românești și spații
    text = re.sub(r"[^a-zăâîșț\s]", "", text)
    # Eliminăm spații multiple
    text = re.sub(r"\s+", " ", text).strip()
    return text
 
text_curat = curata_text(text_brut)
print("\n=== TEXT CURĂȚAT ===")
print(text_curat[:200], '...')


# =============================================
# PASUL 3: Tokenizare
# =============================================
# Tokenizare pe propoziții (pe textul original, înainte de curățare)
propozitii = sent_tokenize(text_brut)
print(f"\nNumăr propoziții: {len(propozitii)}")
print(f"Prima propoziție: {propozitii[0]}")
 
# Tokenizare pe cuvinte (pe textul curățat)
tokens = text_curat.split()
print(f"\nNumăr tokeni: {len(tokens)}")
print(f"Primii 10 tokeni: {tokens[:10]}")


# =============================================
# PASUL 4: Eliminare stopwords
# =============================================
stop_words = set(stopwords.words("romanian"))
 
print(f"\nNumăr stopwords românești în NLTK: {len(stop_words)}")
print(f"Exemple: {list(stop_words)[:10]}")
 
tokens_filtrati = [w for w in tokens if w not in stop_words and len(w) > 1]
 
print(f"\nTokeni înainte de filtrare: {len(tokens)}")
print(f"Tokeni după filtrare:      {len(tokens_filtrati)}")
print(f"Cuvinte eliminate:           {len(tokens) - len(tokens_filtrati)}")


# =============================================
# PASUL 5: Analiza frecvenței
# =============================================
contor = Counter(tokens_filtrati)
 
print("\n=== TOP 15 CUVINTE ===")
for cuvant, nr in contor.most_common(15):
    freq_rel = nr / len(tokens_filtrati)
    print(f"  {cuvant:20} | apariții: {nr:3} | frecv. relativă: {freq_rel:.4f}")


# =============================================
# PASUL 6: Vizualizare
# =============================================
top_15 = contor.most_common(15)
cuvinte = [c for c, _ in top_15]
frecvente = [n for _, n in top_15]
 
plt.figure(figsize=(12, 6))
plt.barh(cuvinte[::-1], frecvente[::-1], color='#2E86C1', edgecolor='white')
plt.xlabel("Număr apariții", fontsize=12)
plt.title("Top 15 cuvinte – Articol: Mascaroanele, dovezile unui „București chipeș“. Fotograf: „Poți să găsești frumosul și unde nu-ți place. Se uită de sus la tine“", fontsize=14)
plt.tight_layout()
plt.show()
