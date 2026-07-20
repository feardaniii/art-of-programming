# AI Retail App

## Run

pip install -r requirements.txt

uvicorn api:app --reload

streamlit run ui.py

## Login

admin / 1234

## Features

- chat AI (fallback local)
- sell products
- live chart

=====Sfaturi utile=====
✅ GARANTAT FUNCȚIONEAZĂ
fără AI key
fără JSON errors
cu graf live

1. Pornește backend-ul

Într-un terminal NOU:

cd "C:\Users\user\Desktop\Free A.I"
uvicorn api:app --reload
✅ 2. Verifică că merge

Trebuie să vezi ceva de genul:

Uvicorn running on http://127.0.0.1:8000

👉 dacă NU vezi asta → spune-mi ce apare

🔥 3. NU închide terminalul

⚠️ Foarte important
Backend-ul trebuie să rămână pornit

✅ 4. Rulează UI (în ALT terminal)
cd "C:\Users\user\Desktop\Free A.I"
streamlit run ui.py

🚀 Rulează aplicația

1. Backend (terminal 1)
   uvicorn api:app --reload

👉 trebuie să vezi:

Uvicorn running on http://127.0.0.1:8000 2. UI (terminal 2)
cd "C:\Users\user\Desktop\Free A.I"
streamlit run ui.py
🌐 Browser

👉 http://localhost:8501

🔐 Login
admin
1234
🧪 Test
show products
sell product id 1 quantity 2
