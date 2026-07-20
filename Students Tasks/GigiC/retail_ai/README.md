# 🛒 Carrefour Cluj-Vivo — Magazin Retail Asistat de AI

Un sistem de gestiune a inventarului unui magazin retail, integrat cu un agent AI (Google Gemini) care înțelege limbaj natural și poate executa operații autonome pe baza de date.

---

## 📋 Descriere

Aplicație Python care combină:
- **Bază de date SQLite** cu inventarul unui magazin retail
- **Agent AI (Google Gemini)** care răspunde la întrebări și execută operații
- **Memoria sesiunii** — conversațiile se salvează în JSON și se reîncarcă
- **Meniu interactiv** în terminal cu două moduri: normal și avansat

### Modul Avansat (AI Autonom)
Agentul poate interpreta cereri în limbaj natural și executa automat comenzi SQL:

> *"Am vândut 15 TV-uri astăzi, actualizează stocul"*

Agentul generează și execută automat:
```sql
UPDATE produse SET stoc_curent = stoc_curent - 15 WHERE id = 1;
INSERT INTO vanzari (produs_id, cantitate, data_vanzarii, total_valoare) VALUES (1, 15, DATE('now'), 49499.85);
```

---

## 🚀 Instalare

### 1. Clonează sau descarcă proiectul
```bash
git clone <url-repo>
cd retail_ai
```

### 2. Creează și activează un virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalează dependențele
```bash
pip install -r requirements.txt
```

### 4. Configurează cheia API Gemini
Creează un fișier `.env` în folderul proiectului:
```
GEMINI_API_KEY=cheia_ta_de_la_google_ai_studio
```

> **Obții cheia gratuit pe:** https://aistudio.google.com → Get API Key

### 5. Rulează aplicația
```bash
python main.py
```

---

## 📁 Structura proiectului

```
retail_ai/
│
├── main.py              # Aplicația principală + meniu + integrare Gemini
├── database.py          # Baza de date SQLite — creare, populare, interogare
├── requirements.txt     # Dependențele Python
├── .env                 # Cheia API (NU se include în git!)
├── .gitignore           # Fișiere ignorate de git
│
├── carrefour_cluj.db    # Generat automat la prima rulare
└── session_*.json       # Sesiuni salvate automat
```

---

## 🎮 Funcționalități

| Opțiune | Descriere |
|---------|-----------|
| 1 | Afișează inventarul complet cu status stoc |
| 2 | Caută produse după nume sau categorie |
| 3 | Statistici vânzări lunare (RON și USD) |
| 4 | Produse cu stoc mic sau epuizat |
| 5 | Actualizează stoc manual |
| 6 | Chat cu AI — întrebări despre magazin |
| 7 | Chat AI Avansat — execută SQL autonom |
| 8 | Încarcă o sesiune anterioară |
| 9 | Vezi istoricul conversației curente |

---

## 💬 Exemple de utilizare

### Mod normal (opțiunea 6):
```
Tu: Ce frigidere avem în stoc?
AI: Avem în stoc Frigider Samsung No Frost: 30 bucăți, la prețul de 2199.99 RON.
```

### Mod avansat (opțiunea 7):
```
Tu: Am vândut 10 frigidere azi, actualizează stocul
AI: Am actualizat stocul pentru Frigider Samsung No Frost: 30 → 20 bucăți.
    Am înregistrat și vânzarea în tabelul vanzari.
```

### Cerere complexă (provocarea avansată):
```
Tu: Am vândut 15 TV-uri, 50 frigidere, 35 mașini de spălat și am epuizat
    toate cuptoarele cu microunde. Actualizează stocul și dă-mi statisticile lunare.
```

---

## 🗄️ Schema bazei de date

```
categorii          produse              vanzari
─────────          ───────              ───────
id                 id                   id
nume               nume                 produs_id
descriere          categorie_id         cantitate
                   furnizor_id          data_vanzarii
                   pret_unitar          total_valoare
                   stoc_curent
                   stoc_minim           furnizori
                   unitate_masura       ────────
                                        id
                                        nume
                                        contact
                                        tara
```

---

## 🔧 Tehnologii folosite

- **Python 3.11+**
- **SQLite3** — bază de date locală (inclusă în Python)
- **Google Generative AI** (`google-generativeai`) — Gemini 2.5 Flash
- **python-dotenv** — gestionarea variabilelor de mediu

---

## ⚠️ Securitate

- Cheia API se stochează **exclusiv** în `.env`
- `.env` este inclus în `.gitignore` — nu ajunge niciodată pe GitHub
- Pentru producție: validați SQL-ul generat de AI înainte de execuție

---


