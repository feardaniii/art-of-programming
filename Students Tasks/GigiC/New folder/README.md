# 🛒 AI Retail Agent

Un agent AI simplu pentru gestiunea unui magazin (produse, stocuri, vânzări), construit cu Python, SQLite și Google Gemini.

---

## Funcționalități

- Afișare produse din baza de date
- Vânzare produse (update stock)
- Ștergere produse
- AI care interpretează comenzi în limbaj natural
- memorie simplă a conversațiilor
- SQLite database local

---

## Arhitectură

Proiectul este împărțit în 4 componente:

- `main.py` → interfața CLI (chat loop)
- `chat.py` → AI (Gemini) care decide acțiuni
- `agent.py` → execută acțiunile pe DB
- `database.py` → SQLite logic
- `memory.py` → salvare istoric conversații

---

## ⚙️ Instalare

### 1. Clonează proiectul

```bash
git clone <repo-url>
cd ai_retail_app
2. Creează mediu virtual
python -m venv venv
3. Activează venv

Windows:

venv\Scripts\activate
4. Instalează dependințe
pip install google-genai python-dotenv
5. Configurează API Key

Creează fișier .env:

GOOGLE_API_KEY=your_api_key_here
 Rulare
python main.py
 Exemple de comenzi

În interiorul aplicației:

show produse
vinde 2 TV Samsung
șterge produs 1
🗄️ Baza de date

SQLite local (store.db) cu tabel:

id
nume
categorie
preț
stoc
AI Flow
User scrie comandă

Gemini returnează JSON:

{"action": "show_products"}
Agent execută acțiunea
Output este formatat pentru utilizator
 Note
Necesită cheie Google Gemini API
Funcționează offline doar DB, AI necesită internet
Model recomandat: gemini-2.5-flash-lite
```
