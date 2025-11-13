# Lista de prieteni
prieteni = [
    {"nume": "Ana", "telefon": "1234567890", "varsta": 23, "ultim_contact": "2025-09-10", "hobbyuri": ["citit", "yoga"]},
    {"nume": "Mihai", "telefon": "1111111111", "varsta": 47, "ultim_contact": "2025-08-20", "hobbyuri": ["fotbal", "calatorii"]},
    {"nume": "Elena", "telefon": "2222222222", "varsta": 17, "ultim_contact": "2025-08-15", "hobbyuri": ["gatit", "pictura"]},
    {"nume": "Andrei", "telefon": "3333333333", "varsta": 32, "ultim_contact": "2025-09-03", "hobbyuri": ["gaming", "fotbal"]},
    {"nume": "Ioana", "telefon": "4444444444", "varsta": 65, "ultim_contact": "2025-07-05", "hobbyuri": ["citit", "calatorii"]}
]

while True:
    print("\n=== Meniu ===")
    print("1. Afiseaza lista prieteni")
    print("2. Adauga un prieten nou")
    print("3. Schimba numarul de telefon pentru un prieten")
    print("4. Afiseaza hobbyurile unice")
    print("5. Iesire")

    alegere = input("Alege o optiune (1-5): ")

    if alegere == "1":
        print("\nLista prieteni:")
        for p in prieteni:
            print(p)

    elif alegere == "2":
        nume = input("Nume: ")
        telefon = input("Telefon: ")
        varsta = int(input("Varsta: "))
        ultim_contact = input("Ultimul contact (YYYY-MM-DD): ")
        hobbyuri_input = input("Hobbyuri (separate prin virgula): ")
        hobbyuri = [h.strip() for h in hobbyuri_input.split(",")]
        prieteni.append({"nume": nume, "telefon": telefon, "varsta": varsta, "ultim_contact": ultim_contact, "hobbyuri": hobbyuri})
        print(f"{nume} a fost adaugat cu succes!")

    elif alegere == "3":
        nume = input("Introdu numele prietenului pentru care vrei sa schimbi numarul: ")
        gasit = False
        for p in prieteni:
            if p["nume"] == nume:
                telefon_nou = input("Introdu numarul nou: ")
                p["telefon"] = telefon_nou
                print(f"Numarul lui {nume} a fost actualizat.")
                gasit = True
        if not gasit:
            print(f"Nu s-a gasit prieten cu numele {nume}.")

    elif alegere == "4":
        hobbyuri_unice = set()
        for p in prieteni:
            for h in p["hobbyuri"]:
                hobbyuri_unice.add(h)
        print("\nHobbyuri unice:")
        for h in hobbyuri_unice:
            print(h)

    elif alegere == "5":
        print("La revedere!")
        break

    else:
        print("Optiune invalida, incearca din nou.")