import pandas as pd
import numpy as np
import re
import string
import os
import matplotlib.pyplot as plt
import seaborn as sns
 
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


# ============================================================
# SETUP COMUN
# ============================================================
 
script_dir = os.path.dirname(os.path.abspath(__file__))
cale_csv = os.path.join(script_dir, "spam.csv")
 
# Încărcare
df = pd.read_csv(cale_csv, encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
 
# Preprocesare
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
 
# Split
X_text = df['clean_text']
y = df['label']
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42)
 
print(f"Dataset: {len(df)} mesaje ({y.sum()} spam, {len(df)-y.sum()} ham)")
print(f"Train: {len(X_train_text)} | Test: {len(X_test_text)}")
 
# Vectorizare
count_vec = CountVectorizer(max_features=3000)
X_train_count = count_vec.fit_transform(X_train_text)
X_test_count = count_vec.transform(X_test_text)
 
tfidf_vec = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf_vec.fit_transform(X_train_text)
X_test_tfidf = tfidf_vec.transform(X_test_text)
 
 
# Funcție de evaluare
def evalueaza_model(model, X_train, X_test, y_train, y_test, nume):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
        target_names=['Ham', 'Spam'], output_dict=True)
    print('\n' + '='*55)
    print(f' {nume}')
    print('='*55)
    print(f" Accuracy:  {acc:.4f}   |  F1 (spam): {f1:.4f}")
    print(f' Precision: {report["Spam"]["precision"]:.4f}  |  ' +
          f'Recall:     {report["Spam"]["recall"]:.4f}')
    return {'accuracy': acc, 'f1': f1,
            'precision': report['Spam']['precision'],
            'recall': report['Spam']['recall'],
            'y_pred': y_pred}


# Antrenăm modelul final pe TOT datasetul
tfidf_final = TfidfVectorizer(max_features=3000)
X_all = tfidf_final.fit_transform(df['clean_text'])
model_final = LogisticRegression(max_iter=1000)
model_final.fit(X_all, df['label'])
 
 
def predic_spam(mesaj):
    """Prezice spam/ham cu probabilitate."""
    mesaj_curat = preprocess(mesaj)
    mesaj_vec = tfidf_final.transform([mesaj_curat])
    pred = model_final.predict(mesaj_vec)[0]
    proba = model_final.predict_proba(mesaj_vec)[0]
    eticheta = "SPAM 🚨" if pred == 1 else "HAM ✅"
    return eticheta, max(proba)*100, proba
 
 
# ─── Test batch ───
mesaje_test = [
    "FREE entry to win FA Cup tickets! Text NOW!",
    "Congratulations! You won a $1000 gift card. Claim now!",
    "WINNER! Call 09061743810 to claim your prize.",
    "Hey, are you coming to the party tonight?",
    "Can you pick up some milk on your way home?",
    "The meeting has been moved to 3pm tomorrow.",
    "Special offer just for you! 50% off today only!",
    "Your order has been shipped. Track delivery here.",
]
 
print("\n" + "═"*60)
print("  EXERCIȚIUL III – SISTEM INTERACTIV SPAM/HAM")
print("═"*60)
print("\n📋 Test batch:")
 
for i, msg in enumerate(mesaje_test, 1):
    et, conf, pr = predic_spam(msg)
    preview = msg[:50] + '...' if len(msg) > 50 else msg
    print(f"  [{i}] {preview}")
    print(f"      → {et}  (confidență: {conf:.1f}%)\n")
 
 
# ─── Loop interactiv ───
print("\n" + "-"*60)
print("  Mod interactiv – scrie un mesaj SMS (sau \"exit\")")
print("-"*60)
 
while True:
    mesaj = input("\n📩 Mesaj: ").strip()
    if mesaj.lower() == "exit":
        print("\n👋 La revedere!")
        break
    if not mesaj:
        continue
    et, conf, pr = predic_spam(mesaj)
    print(f"  Rezultat:   {et}")
    print(f"  Confidență:  {conf:.1f}%")
    print(f"  P(ham)={pr[0]:.4f}  P(spam)={pr[1]:.4f}")
