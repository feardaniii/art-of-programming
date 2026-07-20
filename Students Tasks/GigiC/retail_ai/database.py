"""
=============================================================
DATABASE.PY — Baza de date SQLite pentru magazin retail
=============================================================
Ce face acest fișier:
- Creează baza de date SQLite local (un fișier .db pe calculator)
- Definește tabelele: categorii, produse, vânzări, furnizori
- Populează cu date de exemplu (Carrefour Cluj-Vivo)
- Oferă funcții de interogare și actualizare

SQLite = o bază de date care trăiește într-un singur fișier .db
Nu necesită server, nu necesită instalare separată - vine cu Python!
=============================================================
"""

import sqlite3
import os
from datetime import datetime, date

# Numele fișierului bazei de date
# Va fi creat automat în același folder cu scriptul
DB_NAME = "carrefour_cluj.db"


def get_connection():
    """
    Creează și returnează o conexiune la baza de date.
    
    Gândește-te la conexiune ca la o "linie telefonică" 
    între Python și baza de date SQLite.
    """
    conn = sqlite3.connect(DB_NAME)
    # Această linie face ca rezultatele să fie returnate
    # ca dicționare (ex: {"nume": "TV Samsung"})
    # în loc de tupluri simple (ex: ("TV Samsung",))
    conn.row_factory = sqlite3.Row
    return conn


def create_tables():
    """
    Creează structura bazei de date (tabelele).
    
    Structura bazei de date pentru Carrefour Cluj-Vivo:
    
    categorii          produse              vanzari
    ---------          -------              -------
    id                 id                   id
    nume               nume                 produs_id ──→ produse.id
    descriere          categorie_id ──→     cantitate
                       categorii.id         data_vanzarii
                       pret_unitar          total_valoare
                       stoc_curent
                       stoc_minim           furnizori
                       furnizor_id ──→      --------
                       furnizori.id         id
                                            nume
                                            contact
                                            tara
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── TABEL: categorii ──────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categorii (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            nume        TEXT NOT NULL UNIQUE,
            descriere   TEXT
        )
    """)

    # ── TABEL: furnizori ──────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS furnizori (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            nume        TEXT NOT NULL,
            contact     TEXT,
            tara        TEXT
        )
    """)

    # ── TABEL: produse ────────────────────────────────────
    # Acesta este tabelul principal - inventarul magazinului
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS produse (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            nume            TEXT NOT NULL,
            categorie_id    INTEGER REFERENCES categorii(id),
            furnizor_id     INTEGER REFERENCES furnizori(id),
            pret_unitar     REAL NOT NULL,
            stoc_curent     INTEGER DEFAULT 0,
            stoc_minim      INTEGER DEFAULT 10,
            unitate_masura  TEXT DEFAULT 'buc'
        )
    """)

    # ── TABEL: vanzari ────────────────────────────────────
    # Istoricul vânzărilor - pentru statistici lunare
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vanzari (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            produs_id       INTEGER REFERENCES produse(id),
            cantitate       INTEGER NOT NULL,
            data_vanzarii   DATE DEFAULT CURRENT_DATE,
            total_valoare   REAL
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Tabelele au fost create cu succes!")


def populate_data():
    """
    Populează baza de date cu date de exemplu.
    Simulează inventarul unui Carrefour din Cluj-Vivo.
    
    Această funcție rulează O SINGURĂ DATĂ - dacă datele există deja, nu le mai adaugă.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Verifică dacă datele există deja
    cursor.execute("SELECT COUNT(*) FROM categorii")
    if cursor.fetchone()[0] > 0:
        print("ℹ️  Datele există deja în baza de date.")
        conn.close()
        return

    # ── CATEGORII ─────────────────────────────────────────
    categorii = [
        ("Electronice", "TV-uri, frigidere, mașini de spălat, electrocasnice"),
        ("Alimentare", "Produse alimentare, băuturi, conserve"),
        ("Îmbrăcăminte", "Haine, încălțăminte, accesorii"),
        ("Curățenie", "Detergenți, produse de curățenie"),
        ("Sport", "Echipamente sportive, articole fitness"),
    ]
    cursor.executemany(
        "INSERT INTO categorii (nume, descriere) VALUES (?, ?)", 
        categorii
    )

    # ── FURNIZORI ─────────────────────────────────────────
    furnizori = [
        ("Samsung Electronics", "samsung@partner.ro", "Coreea de Sud"),
        ("LG Romania", "lg@romania.ro", "Romania"),
        ("Nestlé Romania", "nestle@ro.nestle.com", "Romania"),
        ("Adidas Romania", "adidas@romania.ro", "Germania"),
        ("Procter & Gamble", "pg@romania.ro", "SUA"),
    ]
    cursor.executemany(
        "INSERT INTO furnizori (nume, contact, tara) VALUES (?, ?, ?)",
        furnizori
    )

    # ── PRODUSE (INVENTAR) ────────────────────────────────
    # Format: (nume, categorie_id, furnizor_id, pret, stoc_curent, stoc_minim, unitate)
    produse = [
        # Electronice (categorie_id=1)
        ("TV Samsung 55\" 4K QLED",      1, 1, 3299.99,  45,  10, "buc"),
        ("Frigider Samsung No Frost",     1, 1, 2199.99,  30,   5, "buc"),
        ("Mașină de spălat LG 8kg",      1, 2, 1799.99,  25,   5, "buc"),
        ("Cuptor cu microunde Samsung",   1, 1,  599.99,   0,  10, "buc"),  # EPUIZAT!
        ("Aspirator LG CordZero",        1, 2,  899.99,  15,   5, "buc"),
        
        # Alimentare (categorie_id=2)
        ("Lapte Zuzu 1L",                2, 3,    6.99, 500, 100, "buc"),
        ("Iaurt Danone 400g",            2, 3,    4.49, 350,  80, "buc"),
        ("Apă Dorna 2L",                 2, 3,    3.29, 800, 200, "buc"),
        ("Pâine albă feliată",           2, 3,    5.99, 120,  50, "buc"),
        ("Cafea Nescafé Gold 200g",      2, 3,   29.99, 200,  50, "buc"),
        
        # Îmbrăcăminte (categorie_id=3)
        ("Tricou Adidas Sport M",        3, 4,   89.99,  80,  20, "buc"),
        ("Pantofi sport Adidas Run",     3, 4,  299.99,  40,  10, "perechi"),
        
        # Curățenie (categorie_id=4)
        ("Detergent Ariel 3kg",          4, 5,   54.99, 150,  30, "buc"),
        ("Balsam Lenor 1.5L",            4, 5,   29.99, 120,  30, "buc"),
        
        # Sport (categorie_id=5)
        ("Minge fotbal Adidas",          5, 4,  149.99,  30,   5, "buc"),
        ("Saltea yoga 6mm",              5, 4,   79.99,  20,   5, "buc"),
    ]
    cursor.executemany("""
        INSERT INTO produse 
            (nume, categorie_id, furnizor_id, pret_unitar, stoc_curent, stoc_minim, unitate_masura)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, produse)

    # ── VÂNZĂRI (ultimele 30 zile) ────────────────────────
    # Simulăm câteva vânzări pentru statistici
    vanzari = [
        (1,  15, "2026-04-01",  49499.85),  # 15 TV-uri
        (2,  50, "2026-04-01", 109999.50),  # 50 frigidere
        (3,  35, "2026-04-01",  62999.65),  # 35 mașini de spălat
        (4,  20, "2026-03-25",  11999.80),  # 20 cuptoare (înainte să se epuizeze)
        (6, 200, "2026-04-10",   1398.00),  # lapte
        (7, 150, "2026-04-10",    673.50),  # iaurt
        (13, 80, "2026-04-15",   4399.20),  # detergent
    ]
    cursor.executemany("""
        INSERT INTO vanzari (produs_id, cantitate, data_vanzarii, total_valoare)
        VALUES (?, ?, ?, ?)
    """, vanzari)

    conn.commit()
    conn.close()
    print("✅ Datele de exemplu au fost adăugate cu succes!")


def get_all_products():
    """
    Returnează toate produsele cu detalii complete.
    Folosit pentru a arăta LLM-ului structura bazei de date.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            p.id,
            p.nume,
            c.nume AS categorie,
            f.nume AS furnizor,
            p.pret_unitar,
            p.stoc_curent,
            p.stoc_minim,
            p.unitate_masura,
            CASE 
                WHEN p.stoc_curent = 0 THEN '🔴 EPUIZAT'
                WHEN p.stoc_curent <= p.stoc_minim THEN '🟡 STOC MIC'
                ELSE '🟢 OK'
            END AS status_stoc
        FROM produse p
        JOIN categorii c ON p.categorie_id = c.id
        JOIN furnizori f ON p.furnizor_id = f.id
        ORDER BY c.nume, p.nume
    """)
    products = cursor.fetchall()
    conn.close()
    return products


def search_products(query):
    """
    Caută produse după nume sau categorie.
    query = textul introdus de utilizator
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            p.id, p.nume, c.nume AS categorie,
            p.pret_unitar, p.stoc_curent, p.unitate_masura
        FROM produse p
        JOIN categorii c ON p.categorie_id = c.id
        WHERE p.nume LIKE ? OR c.nume LIKE ?
        ORDER BY p.nume
    """, (f"%{query}%", f"%{query}%"))
    results = cursor.fetchall()
    conn.close()
    return results


def update_stock(produs_id, cantitate_vanduta):
    """
    Actualizează stocul după o vânzare.
    cantitate_vanduta = câte bucăți s-au vândut (număr pozitiv)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Verificăm stocul curent
    cursor.execute("SELECT stoc_curent, nume FROM produse WHERE id = ?", (produs_id,))
    produs = cursor.fetchone()
    
    if not produs:
        conn.close()
        return False, "Produsul nu există!"
    
    nou_stoc = produs["stoc_curent"] - cantitate_vanduta
    if nou_stoc < 0:
        conn.close()
        return False, f"Stoc insuficient! Disponibil: {produs['stoc_curent']} buc"
    
    cursor.execute(
        "UPDATE produse SET stoc_curent = ? WHERE id = ?",
        (nou_stoc, produs_id)
    )
    conn.commit()
    conn.close()
    return True, f"✅ Stoc actualizat pentru {produs['nume']}: {produs['stoc_curent']} → {nou_stoc}"


def get_monthly_stats(luna=None, an=None):
    """
    Returnează statisticile de vânzări pentru o lună.
    Dacă nu se specifică luna, returnează luna curentă.
    """
    if not luna:
        luna = datetime.now().month
    if not an:
        an = datetime.now().year
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            p.nume AS produs,
            c.nume AS categorie,
            SUM(v.cantitate) AS total_bucati,
            SUM(v.total_valoare) AS total_valoare_ron,
            ROUND(SUM(v.total_valoare) / 4.97, 2) AS total_valoare_usd
        FROM vanzari v
        JOIN produse p ON v.produs_id = p.id
        JOIN categorii c ON p.categorie_id = c.id
        WHERE strftime('%m', v.data_vanzarii) = ?
          AND strftime('%Y', v.data_vanzarii) = ?
        GROUP BY p.id
        ORDER BY total_valoare_ron DESC
    """, (f"{luna:02d}", str(an)))
    stats = cursor.fetchall()
    conn.close()
    return stats


def get_low_stock_products():
    """
    Returnează produsele cu stoc mic sau epuizat.
    Util pentru planificarea achizițiilor.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            p.id, p.nume, c.nume AS categorie,
            f.nume AS furnizor, f.contact,
            p.stoc_curent, p.stoc_minim,
            p.pret_unitar
        FROM produse p
        JOIN categorii c ON p.categorie_id = c.id
        JOIN furnizori f ON p.furnizor_id = f.id
        WHERE p.stoc_curent <= p.stoc_minim
        ORDER BY p.stoc_curent ASC
    """)
    results = cursor.fetchall()
    conn.close()
    return results


def execute_sql(query):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Împărțim în statements separate
        statements = [s.strip() for s in query.split(';') if s.strip()]
        results = []
        for statement in statements:
            cursor.execute(statement)
            if statement.upper().startswith("SELECT"):
                results.extend(cursor.fetchall())
            
        conn.commit()
        if results:
            conn.close()
            return True, results
        else:
            affected = cursor.rowcount
            conn.close()
            return True, f"{affected} rânduri afectate"
    except Exception as e:
        conn.close()
        return False, str(e)


def initialize_database():
    """
    Funcția principală - inițializează toată baza de date.
    Se apelează o singură dată la pornirea aplicației.
    """
    print("🔧 Inițializare bază de date...")
    create_tables()
    populate_data()
    print(f"✅ Baza de date '{DB_NAME}' este gata!\n")


# ── TESTARE DIRECTĂ ───────────────────────────────────────
# Dacă rulezi acest fișier direct (python database.py),
# se inițializează baza de date și afișează produsele
if __name__ == "__main__":
    initialize_database()
    
    print("\n📦 INVENTAR COMPLET:")
    print("-" * 80)
    products = get_all_products()
    for p in products:
        print(f"[{p['id']:2d}] {p['nume']:<40} "
              f"Stoc: {p['stoc_curent']:4d} {p['unitate_masura']:<8} "
              f"Preț: {p['pret_unitar']:8.2f} RON  {p['status_stoc']}")
    
    print("\n📊 PRODUSE CU STOC MIC/EPUIZAT:")
    print("-" * 80)
    low = get_low_stock_products()
    for p in low:
        print(f"⚠️  {p['nume']}: {p['stoc_curent']}/{p['stoc_minim']} buc "
              f"(Furnizor: {p['furnizor']} - {p['contact']})")