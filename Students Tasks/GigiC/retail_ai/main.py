"""
=============================================================
MAIN.PY — Aplicația principală: Magazin Retail Asistat de AI
=============================================================
Ce face acest fișier:
- Meniul principal (bucla interactivă)
- Integrarea cu Gemini LLM prin API
- Memoria sesiunii (salvare/încărcare conversație în JSON)
- Interogarea bazei de date SQLite
- Modul avansat: agentul AI execută operații autonome pe DB

Fluxul aplicației:
    Utilizator → Meniu → [Căutare DB] sau [Chat AI]
                              ↓               ↓
                         SQLite DB      Gemini API
                                            ↓
                                    Memorie sesiune (JSON)
=============================================================
"""

import os
import json
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# Importăm funcțiile din database.py
from database import (
    initialize_database,
    get_all_products,
    search_products,
    update_stock,
    get_monthly_stats,
    get_low_stock_products,
    execute_sql,
    get_connection,
)

# ── ÎNCĂRCAREA CHEII API ──────────────────────────────────
# load_dotenv() citește fișierul .env și încarcă variabilele
# de mediu în Python. Fără această linie, os.getenv() 
# returnează None.
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ EROARE: GEMINI_API_KEY nu a fost găsită în fișierul .env!")
    print("   Creează fișierul .env cu: GEMINI_API_KEY=cheia_ta")
    exit(1)

# Configurăm biblioteca Gemini cu cheia noastră
genai.configure(api_key=GEMINI_API_KEY)

# ── CONFIGURARE MODEL ─────────────────────────────────────
# gemini-2.0-flash = modelul rapid și gratuit de la Google
MODEL_NAME = "gemini-2.5-flash"
# ── MEMORIA SESIUNII ──────────────────────────────────────
# Conversația se salvează într-un fișier JSON
# La fiecare nouă sesiune, se creează un fișier nou cu timestamp
SESSION_FILE = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Istoricul conversației din sesiunea curentă
# Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
conversation_history = []


# =============================================================
# FUNCȚII PENTRU MEMORIA SESIUNII
# =============================================================

def save_session():
    """
    Salvează conversația curentă în fișier JSON.
    
    De ce JSON? E ușor de citit și de încărcat înapoi în Python.
    Fișierul arată astfel:
    [
        {"role": "user", "content": "Ce stoc avem?", "timestamp": "2026-04-26 14:00"},
        {"role": "assistant", "content": "Avem 45 TV-uri...", "timestamp": "2026-04-26 14:00"}
    ]
    """
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=2)


def load_previous_session(filename):
    """
    Încarcă o sesiune anterioară din fișier JSON.
    Aceasta devine 'pre-context' pentru LLM - el știe ce s-a discutat înainte.
    """
    global conversation_history
    try:
        with open(filename, "r", encoding="utf-8") as f:
            conversation_history = json.load(f)
        print(f"✅ Sesiune încărcată: {len(conversation_history)} mesaje din {filename}")
        return True
    except FileNotFoundError:
        print(f"❌ Fișierul {filename} nu a fost găsit!")
        return False


def add_to_history(role, content):
    """
    Adaugă un mesaj în istoricul conversației și salvează.
    role = "user" sau "assistant"
    """
    conversation_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_session()


def get_conversation_as_text():
    """
    Convertește istoricul conversației într-un text continuu.
    Acesta devine 'pre-context' - îl trimitem la Gemini înainte de întrebarea nouă.
    
    Gândește-te la asta ca la un rezumat al conversației anterioare
    pe care îl dai unui asistent nou ca să știe contextul.
    """
    if not conversation_history:
        return ""
    
    text = "=== ISTORICUL CONVERSAȚIEI ANTERIOARE ===\n"
    for msg in conversation_history:
        role_label = "👤 Utilizator" if msg["role"] == "user" else "🤖 Asistent"
        text += f"\n[{msg['timestamp']}] {role_label}:\n{msg['content']}\n"
    text += "\n=== SFÂRȘIT ISTORIC ===\n"
    return text


# =============================================================
# FUNCȚII PENTRU BAZA DE DATE (AFIȘARE)
# =============================================================

def display_all_products():
    """Afișează inventarul complet în terminal."""
    products = get_all_products()
    if not products:
        print("❌ Nu există produse în baza de date!")
        return
    
    print("\n" + "="*85)
    print(f"{'📦 INVENTAR COMPLET — CARREFOUR CLUJ-VIVO':^85}")
    print("="*85)
    print(f"{'ID':<4} {'Produs':<38} {'Stoc':>6} {'U.M.':<8} {'Preț RON':>10}  Status")
    print("-"*85)
    
    current_category = None
    for p in products:
        # Afișăm header-ul categoriei când se schimbă
        if p["categorie"] != current_category:
            current_category = p["categorie"]
            print(f"\n  🏷️  {current_category.upper()}")
        
        print(f"[{p['id']:2d}] {p['nume']:<38} {p['stoc_curent']:>6} {p['unitate_masura']:<8} "
              f"{p['pret_unitar']:>10.2f}  {p['status_stoc']}")
    
    print("="*85)


def display_search_results(query):
    """Caută și afișează produse."""
    results = search_products(query)
    if not results:
        print(f"❌ Nu am găsit produse pentru '{query}'")
        return
    
    print(f"\n🔍 Rezultate pentru '{query}':")
    print("-"*70)
    for p in results:
        print(f"[{p['id']:2d}] {p['nume']:<35} Stoc: {p['stoc_curent']:4d} {p['unitate_masura']:<8} "
              f"Preț: {p['pret_unitar']:.2f} RON")


def display_monthly_stats():
    """Afișează statisticile lunare."""
    stats = get_monthly_stats()
    if not stats:
        print("❌ Nu există date de vânzări pentru luna curentă.")
        # Încearcă luna anterioară
        stats = get_monthly_stats(luna=4, an=2026)
    
    print("\n" + "="*75)
    print(f"{'📊 STATISTICI VÂNZĂRI — APRILIE 2026':^75}")
    print("="*75)
    print(f"{'Produs':<35} {'Buc':>6} {'Total RON':>12} {'Total USD':>12}")
    print("-"*75)
    
    total_ron = 0
    total_usd = 0
    for s in stats:
        print(f"{s['produs']:<35} {s['total_bucati']:>6} {s['total_valoare_ron']:>12.2f} "
              f"{s['total_valoare_usd']:>12.2f}")
        total_ron += s['total_valoare_ron']
        total_usd += s['total_valoare_usd']
    
    print("-"*75)
    print(f"{'TOTAL':.<35} {'':>6} {total_ron:>12.2f} {total_usd:>12.2f}")
    print("="*75)


def display_low_stock():
    """Afișează produsele cu stoc critic."""
    products = get_low_stock_products()
    if not products:
        print("✅ Toate produsele au stoc suficient!")
        return
    
    print("\n⚠️  PRODUSE CU STOC MIC SAU EPUIZAT:")
    print("-"*70)
    for p in products:
        status = "🔴 EPUIZAT" if p['stoc_curent'] == 0 else "🟡 STOC MIC"
        print(f"{status} {p['nume']}")
        print(f"        Stoc: {p['stoc_curent']}/{p['stoc_minim']} buc | "
              f"Furnizor: {p['furnizor']} ({p['contact']})")


# =============================================================
# FUNCȚII PENTRU AGENTUL AI (GEMINI)
# =============================================================

def get_database_context():
    """
    Pregătește un rezumat al bazei de date pentru LLM.
    
    LLM-ul nu are acces direct la baza de date — trebuie să îi 
    explicăm ce conține, ca el să poată genera SQL corect.
    
    E ca și cum ai explica unui consultant extern structura 
    companiei tale înainte să îl lași să ia decizii.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Obținem schema tabelelor
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    schema_text = "=== STRUCTURA BAZEI DE DATE (SQLite) ===\n\n"
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        schema_text += f"Tabel: {table_name}\n"
        for col in columns:
            schema_text += f"  - {col[1]} ({col[2]})\n"
        schema_text += "\n"
    
    # Adăugăm datele actuale
    schema_text += "=== DATE ACTUALE ===\n\n"
    products = get_all_products()
    schema_text += "PRODUSE:\n"
    for p in products:
        schema_text += (f"  ID={p['id']}: {p['nume']} | "
                       f"Stoc={p['stoc_curent']} | "
                       f"Preț={p['pret_unitar']} RON | "
                       f"Status={p['status_stoc']}\n")
    
    conn.close()
    return schema_text


def chat_with_ai(user_message, advanced_mode=False):
    """
    Trimite un mesaj la Gemini și primește răspunsul.
    
    Parametri:
    - user_message: ce a scris utilizatorul
    - advanced_mode: dacă True, agentul poate executa SQL autonom
    
    Cum funcționează memoria sesiunii:
    1. Luăm tot istoricul conversației anterioare (pre-context)
    2. Adăugăm contextul bazei de date
    3. Adăugăm mesajul nou al utilizatorului
    4. Trimitem totul la Gemini
    5. Salvăm răspunsul în istoric
    """
    
    # ── CONSTRUIM PROMPT-UL COMPLET ───────────────────────
    
    # 1. System prompt - spunem LLM-ului cine este și ce poate face
    if advanced_mode:
        system_prompt = """Ești un asistent AI pentru magazinul Carrefour Cluj-Vivo.
        
Ai acces COMPLET la baza de date SQLite a magazinului și poți:
1. Interpreta cereri în limbaj natural
2. Genera și executa comenzi SQL (SELECT, UPDATE, DELETE, INSERT)
3. Actualiza stocuri după vânzări
4. Genera statistici de vânzări
5. Planifica achiziții

Când primești o cerere:
- Analizează ce operații SQL sunt necesare
- Scrie comenzile SQL între tag-urile [SQL] și [/SQL]
- Explică ce faci în română
- Oferă rezultatele și recomandările

Format răspuns:
[SQL]
UPDATE produse SET stoc_curent = stoc_curent - X WHERE id = Y;
[/SQL]
Explicație: Am actualizat stocul pentru produsul Y.

Rata de schimb: 1 USD = 4.97 RON
"""
    else:
        system_prompt = """Ești un asistent AI prietenos pentru magazinul Carrefour Cluj-Vivo.
Răspunzi în română și ajuți cu întrebări despre inventar, prețuri și stocuri.
Folosește informațiile din baza de date furnizată pentru răspunsuri precise.
Fii concis și util.
"""
    
    # 2. Pre-context: istoricul conversației anterioare
    conversation_context = get_conversation_as_text()
    
    # 3. Contextul bazei de date
    db_context = get_database_context()
    
    # 4. Construim mesajul final
    full_prompt = f"""{system_prompt}

{db_context}

{conversation_context}

👤 Utilizator: {user_message}

🤖 Asistent:"""

    # ── APELĂM GEMINI API ─────────────────────────────────
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(full_prompt)
        ai_response = response.text
        
        # ── MODUL AVANSAT: EXECUTĂM SQL-UL DIN RĂSPUNS ───
        if advanced_mode and "[SQL]" in ai_response:
            ai_response = execute_ai_sql(ai_response)
        
        # ── SALVĂM ÎN MEMORIE ─────────────────────────────
        add_to_history("user", user_message)
        add_to_history("assistant", ai_response)
        
        return ai_response
        
    except Exception as e:
        return f"❌ Eroare la conectarea cu Gemini: {str(e)}"


def execute_ai_sql(ai_response):
    """
    Extrage și execută comenzile SQL din răspunsul AI.
    
    Acesta e inima modului avansat — agentul generează SQL
    și noi îl executăm autonom pe baza de date.
    
    ATENȚIE: În producție, ar trebui validat SQL-ul înainte
    de execuție (securitate). Aici e simplificat pentru temă.
    """
    import re
    
    # Extragem tot ce e între [SQL] și [/SQL]
    sql_pattern = re.compile(r'\[SQL\](.*?)\[/SQL\]', re.DOTALL)
    sql_matches = sql_pattern.findall(ai_response)
    
    if not sql_matches:
        return ai_response
    
    execution_results = "\n\n📊 REZULTATE EXECUȚIE SQL:\n" + "="*50
    
    for sql_query in sql_matches:
        sql_query = sql_query.strip()
        if not sql_query:
            continue
            
        print(f"\n⚙️  Execut SQL: {sql_query[:60]}...")
        success, result = execute_sql(sql_query)
        
        if success:
            if isinstance(result, list):
                # E un SELECT - afișăm rezultatele
                execution_results += f"\n✅ Query executat cu succes:\n"
                for row in result:
                    execution_results += f"   {dict(row)}\n"
            else:
                execution_results += f"\n✅ {result}\n"
        else:
            execution_results += f"\n❌ Eroare SQL: {result}\n"
    
    return ai_response + execution_results


# =============================================================
# MENIUL PRINCIPAL (BUCLA INTERACTIVĂ)
# =============================================================

def show_menu():
    """Afișează meniul principal."""
    print("\n" + "="*55)
    print(f"{'🛒 CARREFOUR CLUJ-VIVO — SISTEM AI':^55}")
    print("="*55)
    print("  📦 BAZĂ DE DATE:")
    print("     1. Vezi inventarul complet")
    print("     2. Caută produse")
    print("     3. Statistici vânzări lunare")
    print("     4. Produse cu stoc mic/epuizat")
    print("     5. Actualizează stoc manual")
    print()
    print("  🤖 ASISTENT AI:")
    print("     6. Chat cu AI (mod normal)")
    print("     7. Chat cu AI (mod AVANSAT - execută SQL autonom)")
    print()
    print("  ⚙️  SESIUNE:")
    print("     8. Încarcă sesiune anterioară")
    print("     9. Vezi istoricul conversației curente")
    print()
    print("     0. Ieșire")
    print("="*55)


def main():
    """
    Funcția principală — bucla interactivă a aplicației.
    
    O buclă while True rulează la infinit până utilizatorul
    alege să iasă (opțiunea 0).
    """
    print("\n🚀 Pornire sistem Carrefour AI...")
    
    # Inițializăm baza de date (creează tabelele și datele dacă nu există)
    initialize_database()
    
    print(f"💾 Sesiunea curentă se salvează în: {SESSION_FILE}")
    
    # ── BUCLA PRINCIPALĂ ──────────────────────────────────
    while True:
        show_menu()
        
        choice = input("\n  Alege opțiunea: ").strip()
        
        # ── OPȚIUNEA 1: Inventar complet ──────────────────
        if choice == "1":
            display_all_products()
        
        # ── OPȚIUNEA 2: Căutare produse ───────────────────
        elif choice == "2":
            query = input("\n  🔍 Caută (nume sau categorie): ").strip()
            if query:
                display_search_results(query)
        
        # ── OPȚIUNEA 3: Statistici lunare ─────────────────
        elif choice == "3":
            display_monthly_stats()
        
        # ── OPȚIUNEA 4: Stoc mic ──────────────────────────
        elif choice == "4":
            display_low_stock()
        
        # ── OPȚIUNEA 5: Actualizare stoc manual ───────────
        elif choice == "5":
            display_all_products()
            try:
                produs_id = int(input("\n  ID produs: "))
                cantitate = int(input("  Cantitate vândută: "))
                success, msg = update_stock(produs_id, cantitate)
                print(f"\n  {msg}")
            except ValueError:
                print("  ❌ Introdu numere valide!")
        
        # ── OPȚIUNEA 6: Chat AI normal ────────────────────
        elif choice == "6":
            print("\n  🤖 MOD CHAT AI (scrie 'exit' pentru a ieși)")
            print("  " + "-"*50)
            
            while True:
                user_input = input("\n  Tu: ").strip()
                if user_input.lower() == "exit":
                    break
                if not user_input:
                    continue
                
                print("\n  🤖 AI gândește...")
                response = chat_with_ai(user_input, advanced_mode=False)
                print(f"\n  🤖 Asistent:\n  {response}")
        
        # ── OPȚIUNEA 7: Chat AI avansat ───────────────────
        elif choice == "7":
            print("\n  🤖 MOD AVANSAT — Agentul poate modifica baza de date!")
            print("  ⚠️  Orice comandă va fi executată autonom pe DB.")
            print("  Exemplu: 'Am vândut 15 TV-uri astăzi, actualizează stocul'")
            print("  " + "-"*50)
            
            while True:
                user_input = input("\n  Tu: ").strip()
                if user_input.lower() == "exit":
                    break
                if not user_input:
                    continue
                
                print("\n  ⚙️  Agentul procesează cererea...")
                response = chat_with_ai(user_input, advanced_mode=True)
                print(f"\n  🤖 Asistent:\n{response}")
        
        # ── OPȚIUNEA 8: Încarcă sesiune anterioară ────────
        elif choice == "8":
            # Listăm fișierele de sesiune disponibile
            session_files = [f for f in os.listdir(".") if f.startswith("session_") and f.endswith(".json")]
            
            if not session_files:
                print("  ❌ Nu există sesiuni salvate!")
            else:
                print("\n  📂 Sesiuni disponibile:")
                for i, f in enumerate(sorted(session_files), 1):
                    print(f"     {i}. {f}")
                
                try:
                    idx = int(input("  Alege numărul sesiunii: ")) - 1
                    if 0 <= idx < len(session_files):
                        load_previous_session(sorted(session_files)[idx])
                    else:
                        print("  ❌ Număr invalid!")
                except ValueError:
                    print("  ❌ Introdu un număr valid!")
        
        # ── OPȚIUNEA 9: Vezi istoricul curent ─────────────
        elif choice == "9":
            if not conversation_history:
                print("\n  ℹ️  Nu există conversații în sesiunea curentă.")
            else:
                print(f"\n  📜 ISTORICUL SESIUNII CURENTE ({len(conversation_history)} mesaje):")
                print("  " + "-"*50)
                for msg in conversation_history:
                    role = "👤 Tu" if msg["role"] == "user" else "🤖 AI"
                    print(f"\n  [{msg['timestamp']}] {role}:")
                    # Afișăm primele 200 de caractere
                    content_preview = msg["content"][:200]
                    if len(msg["content"]) > 200:
                        content_preview += "..."
                    print(f"  {content_preview}")
        
        # ── OPȚIUNEA 0: Ieșire ────────────────────────────
        elif choice == "0":
            print(f"\n  👋 La revedere! Sesiunea salvată în: {SESSION_FILE}")
            print(f"  📊 Total mesaje în sesiune: {len(conversation_history)}")
            break
        
        else:
            print("  ❌ Opțiune invalidă! Alege între 0-9.")
        
        # Pauză scurtă între operații
        input("\n  Apasă Enter pentru a continua...")


# ── PUNCT DE INTRARE ──────────────────────────────────────
# Acest bloc rulează doar când fișierul e executat direct
# (nu când e importat de alt modul)
if __name__ == "__main__":
    main()