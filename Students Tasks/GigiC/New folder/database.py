import sqlite3

DB_NAME = "store.db"


def connect():
    return sqlite3.connect(DB_NAME)


# 🔹 INIT DB (rulează o singură dată la start)
def init_db():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        category TEXT,
        price REAL,
        stock INTEGER
    )
    """)

    # seed data (doar dacă e goală)
    cursor.execute("SELECT COUNT(*) FROM products")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
        INSERT INTO products (name, category, price, stock)
        VALUES (?, ?, ?, ?)
        """, [
            ("TV Samsung", "electronics", 2500.0, 10),
            ("Frigider LG", "electronics", 1800.0, 5),
            ("Masina spalat", "home", 1500.0, 7)
        ])

    conn.commit()
    conn.close()


# 🔹 GET PRODUCTS (FORMAT JSON pentru AI)
def get_products():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM products")
    rows = cursor.fetchall()

    conn.close()

    return [
        {
            "id": r[0],
            "name": r[1],
            "category": r[2],
            "price": r[3],
            "stock": r[4]
        }
        for r in rows
    ]


# 🔹 DELETE PRODUCT
def delete_product(product_id):
    try:
        product_id = int(product_id)
    except:
        return "Invalid product id"

    conn = connect()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM products WHERE id=?", (product_id,))
    conn.commit()
    conn.close()

    return f"Produsul {product_id} a fost șters."


# 🔹 SELL PRODUCT (UPDATE STOCK)
def sell_product(product_id, quantity):
    try:
        product_id = int(product_id)
        quantity = int(quantity)
    except:
        return "Invalid input"

    conn = connect()
    cursor = conn.cursor()

    cursor.execute("SELECT stock FROM products WHERE id=?", (product_id,))
    result = cursor.fetchone()

    if not result:
        conn.close()
        return "Produs inexistent."

    stock = result[0]

    if stock < quantity:
        conn.close()
        return "Stoc insuficient."

    new_stock = stock - quantity

    cursor.execute(
        "UPDATE products SET stock=? WHERE id=?",
        (new_stock, product_id)
    )

    conn.commit()
    conn.close()

    return f"Stoc actualizat. Au ramas {new_stock} bucati."