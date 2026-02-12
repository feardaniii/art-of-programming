import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
cale_csv = os.path.join(script_dir, "spam.csv")
 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
 
nltk.download("stopwords", quiet=True)
 
# ─── 1. Încărcare dataset ───
df = pd.read_csv(cale_csv, encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
# ─── 2. Preprocesare ───
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
 
def preprocess(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)
 
df['clean_text'] = df['text'].apply(preprocess)
 
# ─── 3. Split date ───
X_text = df['clean_text']
y = df['label']
 
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)
 
print(f"Train: {len(X_train_text)} mesaje")
print(f"Test:  {len(X_test_text)} mesaje")
print(f"Distribuție spam în test: {y_test.sum()}/{len(y_test)}")


# ─── Metoda 1: CountVectorizer (Bag of Words) ───
count_vec = CountVectorizer(max_features=3000)
X_train_count = count_vec.fit_transform(X_train_text)
X_test_count = count_vec.transform(X_test_text)
 
print(f"Dimensiune matrice Count: {X_train_count.shape}")
print(f"Exemplu valori: {X_train_count[0].toarray()[0][:10]}")
# Valori întregi: [0, 0, 1, 0, 2, 0, 0, 1, 0, 0]


# ─── Metoda 2: TF-IDF ───
tfidf_vec = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf_vec.fit_transform(X_train_text)
X_test_tfidf = tfidf_vec.transform(X_test_text)
 
print(f"Dimensiune matrice TF-IDF: {X_train_tfidf.shape}")
print(f"Exemplu valori: {X_train_tfidf[0].toarray()[0][:10]}")
# Valori float: [0.0, 0.0, 0.234, 0.0, 0.567, 0.0, 0.0, 0.189, 0.0, 0.0]


# Funcție helper pentru antrenare + evaluare
def evalueaza_model(model, X_train, X_test, y_train, y_test, nume_model):
    """Antrenează modelul și returnează metricile."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
        target_names=['Ham', 'Spam'], output_dict=True)
    
    print('\n' + '='*50)
    print(f' {nume_model}')
    print('='*50)
    print(f' Accuracy:  {acc:.4f}')
    print(f' F1 (spam): {f1:.4f}')
    print(f' Precision: {report["Spam"]["precision"]:.4f}')
    print(f' Recall:    {report["Spam"]["recall"]:.4f}')
    
    return {'accuracy': acc, 'f1': f1,
            'precision': report['Spam']['precision'],
            'recall': report['Spam']['recall'],
            'y_pred': y_pred}

# ─── Naive Bayes cu CountVectorizer ───
rezultat_nb_count = evalueaza_model(
    MultinomialNB(), X_train_count, X_test_count, y_train, y_test,
    "Naive Bayes + CountVectorizer"
)
 
# ─── Naive Bayes cu TF-IDF ───
rezultat_nb_tfidf = evalueaza_model(
    MultinomialNB(), X_train_tfidf, X_test_tfidf, y_train, y_test,
    "Naive Bayes + TF-IDF"
)
 
# ─── Logistic Regression cu CountVectorizer ───
rezultat_lr_count = evalueaza_model(
    LogisticRegression(max_iter=1000), X_train_count, X_test_count,
    y_train, y_test, "Logistic Regression + CountVectorizer"
)
 
# ─── Logistic Regression cu TF-IDF ───
rezultat_lr_tfidf = evalueaza_model(
LogisticRegression(max_iter=1000), X_train_tfidf, X_test_tfidf,
    y_train, y_test, "Logistic Regression + TF-IDF"
)

from sklearn.svm import LinearSVC
 
# ─── SVM cu CountVectorizer ───
rezultat_svm_count = evalueaza_model(
    LinearSVC(max_iter=2000, dual='auto'),
    X_train_count, X_test_count, y_train, y_test,
"LinearSVC + CountVectorizer"
)
 
# ─── SVM cu TF-IDF ───
rezultat_svm_tfidf = evalueaza_model(
    LinearSVC(max_iter=2000, dual='auto'),
    X_train_tfidf, X_test_tfidf, y_train, y_test,
    "LinearSVC + TF-IDF"
)

# Colectăm toate rezultatele
toate_rezultatele = {
    'NB + Count':   rezultat_nb_count,
    'NB + TF-IDF':  rezultat_nb_tfidf,
    'LR + Count':   rezultat_lr_count,
    'LR + TF-IDF':  rezultat_lr_tfidf,
    'SVM + Count':  rezultat_svm_count,
    'SVM + TF-IDF': rezultat_svm_tfidf,
}

# Vizualizăm Confusion Matrix pentru toate 6 combinațiile
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
 
for ax, (nume, rez) in zip(axes.flat, toate_rezultatele.items()):
    cm = confusion_matrix(y_test, rez['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax.set_title(nume, fontsize=11)
    ax.set_xlabel('Predicții')
    ax.set_ylabel('Valori reale')
 
plt.suptitle('Confusion Matrix – Toate modelele', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

 
# Tabel frumos cu pandas
df_rez = pd.DataFrame({
    nume: {m: f'{v[m]:.4f}' for m in ['accuracy','f1','precision','recall']}
    for nume, v in toate_rezultatele.items()
}).T
 
print("\n" + "="*70)
print(" TABEL COMPARATIV FINAL")
print("="*70)
print(df_rez.to_string())

 
# Bar chart comparativ pentru F1-Score (metrica cea mai relevantă)
modele = list(toate_rezultatele.keys())
f1_scores = [toate_rezultatele[m]['f1'] for m in modele]
 
colors = ['#3498DB', '#2980B9',  # NB
          '#E67E22', '#D35400',  # LR
          '#27AE60', '#1E8449']  # SVM
 
plt.figure(figsize=(12, 5))
bars = plt.bar(modele, f1_scores, color=colors, edgecolor='white', linewidth=1.5)
 
# Afișăm valorile
for bar, val in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
 
plt.ylabel('F1-Score (spam)')
plt.title('Comparație F1-Score – Toate modelele')
plt.ylim(0.80, 1.0)
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()



