import re
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from wordcloud import WordCloud

import re
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import os

nltk.download("stopwords", quiet=True)

# --- Preprocesare (aceeași ca în ex_2) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
cale_articol = os.path.join(script_dir, "articol_stiri.txt")

with open(cale_articol, "r", encoding="utf-8") as f:
    text_brut = f.read()

text_curat = text_brut.lower()
text_curat = re.sub(r"\d+", "", text_curat)
text_curat = re.sub(r"[^a-zăâîșț\s]", "", text_curat)
text_curat = re.sub(r"\s+", " ", text_curat).strip()

tokens = text_curat.split()

stop_words = set(stopwords.words("romanian"))
stop_words = {w.replace("ş", "ș").replace("ţ", "ț") for w in stop_words}
tokens_filtrati = [w for w in tokens if w not in stop_words and len(w) > 1]

# --- De aici continuă WordCloud-ul tău ---
text_wc = " ".join(tokens_filtrati)
# ...

# Reunim tokenii filtrați
text_wc = " ".join(tokens_filtrati)
 
# Generăm 3 variante de WordCloud
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
 
# Varianta 1: Clasică
wc1 = WordCloud(width=600, height=400, background_color='white',
    max_words=80, colormap='viridis')
wc1.generate(text_wc)
axes[0].imshow(wc1, interpolation='bilinear')
axes[0].set_title('Varianta 1: viridis', fontsize=13)
axes[0].axis('off')
 
# Varianta 2: Caldă
wc2 = WordCloud(width=600, height=400, background_color='#1B4F72',
    max_words=80, colormap='YlOrRd')
wc2.generate(text_wc)
axes[1].imshow(wc2, interpolation='bilinear')
axes[1].set_title('Varianta 2: fundal închis', fontsize=13)
axes[1].axis('off')
 
# Varianta 3: Elegantă
wc3 = WordCloud(width=600, height=400, background_color='white',
    max_words=60, colormap='coolwarm', prefer_horizontal=0.7)
wc3.generate(text_wc)
axes[2].imshow(wc3, interpolation='bilinear')
axes[2].set_title('Varianta 3: coolwarm', fontsize=13)
axes[2].axis('off')
 
plt.suptitle('WordCloud – 3 Variante de Vizualizare', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
 
# Salvare
wc1.to_file("wordcloud_articol.png")
print("\n✅ WordCloud salvat ca wordcloud_articol.png")
