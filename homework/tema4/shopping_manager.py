# II. Smart Shopping & Budget Manager

# Dicționar produse: preț.
# Operații: adăugare, ștergere, afișare total, sortare, filtrare produse > 50 lei.
# Algoritm „buget”: utilizatorul setează un buget → dacă depășește, programul sugerează ce produse să scoată sau alternative mai ieftine.


produse = {
    "paine": 5,
    "carne": 25,
    "apa": 4,
    "faina": 6,
    "carne vita": 65,
    "ulei masline": 55,
    "detergent rufe": 70,
    "telefon": 1200
}
buget = 0
cheltuieli = 0

while True:
    print("\n=== MENIU ===")
    print("1. Adaugă produs")
    print("2. Șterge produs")
    print("3. Afișează lista și totalul")
    print("4. Sortează produsele după preț")
    print("5. Filtrează produse > 50 lei")
    print("6. Setează buget și verifică")
    print("7. Ieșire")

    alegere = input("\nAlege opțiunea: ")

    if alegere == "1":
        nume = input("Nume produs: ")
        pret = float(input("Preț produs: "))
        produse[nume] = pret
        print("Produs adăugat!")

    elif alegere == "2":
        nume = input("Nume produs de șters: ")
        if nume in produse:
            produse.pop(nume)
            print("Produs șters!")
        else:
            print("Produsul nu există în listă.")

    elif alegere == "3":
        print("\n=== Lista cumpărături ===")
        total = 0
        for p in produse:
            print(p, "-", produse[p], "lei")
            total += produse[p]
        print("Total:", total, "lei")

    elif alegere == "4":
        print("\n=== Produse sortate după preț ===")
        # sortăm după valoarea prețului
        for p in sorted(produse, key=produse.get):
            print(p, "-", produse[p], "lei")

    elif alegere == "5":
        print("\n=== Produse cu preț > 50 lei ===")
        for p in produse:
            if produse[p] > 50:
                print(p, "-", produse[p], "lei")

    elif alegere == "6":
        buget = float(input("Setează bugetul (lei): "))
        total = sum(produse.values())
        print("Total cumpărături:", total, "lei")
        if total <= buget:
            print("Ești în buget! Bravo!")
        else:
            print("Ai depășit bugetul cu", total - buget, "lei!")
            print("Sugestii: încearcă să elimini produse mai scumpe.")
            # sortăm produsele descrescător după preț pentru sugestii
            for p in sorted(produse, key=produse.get, reverse=True):
                print("Poți scoate:", p, "-", produse[p], "lei")

    elif alegere == "7":
        print("La revedere!")
        break

    else:
        print("Opțiune invalidă, încearcă din nou.")