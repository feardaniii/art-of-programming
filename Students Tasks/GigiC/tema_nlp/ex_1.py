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

def frecventa_relativa_simpla(text):
    """
    Calculează frecvența relativă a cuvintelor dintr-un text.
    Args:
        text (str): Textul de analizat
    Returns:
        dict: Dicționar {cuvânt: frecvență_relativă}
    """
    # Convertim la litere mici
    text = text.lower()
    
    # Eliminăm caracterele speciale (păstrăm doar litere și spații)
    text = re.sub(r"[^a-zăâîșț\s]", "", text)
    
    # Separam în cuvinte
    cuvinte = text.split()
    
    # Numărăm total cuvinte
    total = len(cuvinte)
    
    if total == 0:
        return {}
    
    # Frecvența absolută
    contor = Counter(cuvinte)

    # Frecvența relativă = apariții / total
    frecvente = {
        cuvant: round(numar / total, 4)
        for cuvant, numar in contor.most_common()
    }
    
    return frecvente

# testare
text_test = "NLP este fascinant. NLP este util. Python este popular."
 
rezultat = frecventa_relativa_simpla(text_test)
for cuvant, freq in rezultat.items():
    print(f"  {cuvant:15} -> {freq:.4f} ({freq*100:.1f}%)")
