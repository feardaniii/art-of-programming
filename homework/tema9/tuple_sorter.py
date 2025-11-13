persoane = [
    ("Ion", 23),
    ("Mihai", 30),
    ("Ionut", 25),
    ("Ioana", 39)
]

while True:
    print("\nMENIU OPTIUNI")
    print("1. Afiseaza lista cu nume")
    print("2. Sorteaza dupa varsta crescator")
    print("3. Sorteaza dupa varsta descrescator")
    print("4. Iesire")

    alegere = input("\n Alege o optiune (1-4): ").strip().lower()

    if alegere in ["1", "afiseaza lista"]:
        print("\nLista persoanelor: ")
        for nume, varsta in persoane:
            print(f"- {nume} ({varsta} ani)")

    elif alegere in ["2", "crescator", "sortare crescatoare"]:
        persoane_sortate = sorted(persoane, key=lambda x: x[1])
        print("\nLista sortata crescator dupa varsta:")
        for nume, varsta in persoane_sortate:
            print(f"- {nume} ({varsta} ani)")

    elif alegere in ["3", "descrescator", "sortare descrescatoare"]:
        persoane_sortate = sorted(persoane, key=lambda x: x[1], reverse=True)
        print("\nLista sortata descrescator dupa varsta:")
        for nume, varsta in persoane_sortate:
            print(f"- {nume} ({varsta} ani)")

    elif alegere in ["4", "exit", "iesire"]:
        print("\nProgramul s-a incheiat. La revedere!")
        break

    else:
        print("\nOptiune invalida. Incearca din nou!")

print("\n")