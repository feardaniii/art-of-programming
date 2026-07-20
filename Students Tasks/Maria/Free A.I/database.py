import sqlite3

DB = "store.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY,
        name TEXT,
        price REAL,
        stock INTEGER
    )
    """)

    c.execute("DELETE FROM products")

    c.execute("INSERT INTO products VALUES (1,'Laptop',3000,5)")
    c.execute("INSERT INTO products VALUES (2,'Telefon',1500,10)")

    conn.commit()
    conn.close()


def get_products():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("SELECT * FROM products")
    rows = c.fetchall()

    conn.close()

    return [
        {"id": r[0], "name": r[1], "price": r[2], "stock": r[3]}
        for r in rows
    ]


def sell_product(pid, qty):
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("SELECT stock FROM products WHERE id=?", (pid,))
    row = c.fetchone()

    if not row:
        return "Produs inexistent"

    if qty <= 0:
        return "Cantitate invalida"

    stock = row[0]

    if stock < qty:
        return "Stoc insuficient"

    new_stock = stock - qty

    c.execute("UPDATE products SET stock=? WHERE id=?", (new_stock, pid))

    conn.commit()
    conn.close()

    return f"Vandut {qty}"