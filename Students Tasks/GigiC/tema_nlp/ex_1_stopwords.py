import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
 
# Descărcăm resursele NLTK 
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

"""
Frecvența absolută ne spune de câte ori apare un cuvânt. Frecvența relativă ne spune ce proporție din text este ocupată de acel cuvânt. 
Se calculează împărțind numărul de apariții la numărul total de cuvinte.
Formula: frecvență_relativă(cuvânt) = număr_apariții(cuvânt) / total_cuvinte
De exemplu, dacă într-un text de 100 de cuvinte, cuvântul „model” apare de 5 ori, frecvența relativă este 5/100 = 0.05 (sau 5%).
"""

def frecventa_relativa(text, limba='romanian', top_n=None, cu_stopwords=False):
    """
    Calculează frecvența relativă a cuvintelor cu preprocesare completă.
    
    Args:
        text (str): Textul de analizat
        limba (str): Limba pentru stopwords ('romanian' sau 'english')
        top_n (int|None): Dacă e specificat, returnează doar primele N cuvinte
        cu_stopwords (bool): Dacă True, păstrează stopwords
    
    Returns:
        dict: Dicționar ordonat {cuvânt: frecvență_relativă}
    """
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Eliminare caractere speciale
    text = re.sub(r"[^a-zăâîșț\s]", "", text)
    
    # 3. Tokenizare
    cuvinte = text.split()

    # 4. Eliminare stopwords (opțional)
    if not cu_stopwords:
        stop_words = set(stopwords.words(limba))
        cuvinte = [w for w in cuvinte if w not in stop_words]


    """
         if not cu_stopwords:
    stop_words = set(stopwords.words(limba))
    # Normalizăm stopwords la formele corecte românești
    stop_words = {
        w.replace("ş", "ș").replace("ţ", "ț") for w in stop_words
    }
    cuvinte = [w for w in cuvinte if w not in stop_words]   
    """
    
    
    # 5. Eliminare cuvinte scurte (sub 2 caractere)
    cuvinte = [w for w in cuvinte if len(w) > 1]
    total = len(cuvinte)
    if total == 0:
        return {}
    
    # 6. Calcul frecvență relativă
    contor = Counter(cuvinte)
    
    # Limităm la top_n dacă e specificat
    items = contor.most_common(top_n) if top_n else contor.most_common()
    
    frecvente = {
        cuvant: round(numar / total, 4)
        for cuvant, numar in items
    }
    
    return frecvente


text = """
Inteligența artificială transformă modul în care trăim și muncim.
Machine learning și deep learning sunt ramuri ale inteligenței artificiale.
Python este limbajul preferat pentru machine learning și analiza datelor.
Companiile mari investesc masiv în inteligență artificială.
"""
 
# Top 5 cuvinte, fără stopwords
print("\n--- Fără stopwords (top 5) ---")
freq = frecventa_relativa(text, top_n=5)
for cuvant, f in freq.items():
    print(f"  {cuvant:20} -> {f:.4f} ({f*100:.1f}%)")
 
# Cu stopwords pentru comparație
print("\n--- Cu stopwords (top 5) ---")
freq_sw = frecventa_relativa(text, top_n=5, cu_stopwords=True)
for cuvant, f in freq_sw.items():
    print(f"  {cuvant:20} -> {f:.4f} ({f*100:.1f}%)")


# Putem vizualiza frecvența relativă sub formă de grafic bar orizontal:

def plot_frecventa(frecvente, titlu='Frecvența relativă a cuvintelor'):
    """Vizualizează frecvența relativă ca bar chart orizontal."""
    cuvinte = list(frecvente.keys())
    valori = list(frecvente.values())
    
    plt.figure(figsize=(10, max(4, len(cuvinte) * 0.4)))
    bars = plt.barh(cuvinte[::-1], valori[::-1], color='#2E86C1')
    
    # Adăugăm procentajul pe fiecare bară
    for bar, val in zip(bars, valori[::-1]):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val*100:.1f}%', va='center', fontsize=9)
    
    plt.xlabel('Frecvență relativă')
    plt.title(titlu)
    plt.tight_layout()
    plt.show()
 
# Utilizare
freq = frecventa_relativa(text, top_n=10)
plot_frecventa(freq, 'Top 10 cuvinte – Articol AI')

# IMPORTANT DE TINUT MINTE
# si apare in amandoua cazurile dintr-un motiv simplu:
#Textul conține: și cu ș (s cu virgulă jos, U+0219) ✅ forma corectă
#NLTK stopwords conține: şi cu ş (s cu sedilă, U+015F) — un caracter diferit!