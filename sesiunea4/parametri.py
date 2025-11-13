def comanda_cafea(tip_cafea, nivel_zahar):
    """
    Funcție cu parametri pentru comenzi personalizate
    Parametri:
    - tip_cafea (string): Tipul de cafea dorit
    - nivel_zahar (string): Preferința pentru zahăr
    """
    comanda = f"☕ Comandă: {tip_cafea} {nivel_zahar}"
    comanda += ". Comanda ta este pregătită!"
    return comanda

# Apelare cu parametri diferiți
rezultat = comanda_cafea("Espresso", "fără zahăr")
print(rezultat)

# Același cod, rezultate diferite:
# comanda_cafea("Espresso", "fără zahăr")
# comanda_cafea("Cappuccino", "cu zahăr")