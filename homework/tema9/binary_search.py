def binary_search(lista, valoare, return_steps=False):
    stanga = 0
    dreapta = len(lista) - 1
    pasi = []

    while stanga <= dreapta:
        mijloc = (stanga + dreapta) // 2
        element = lista[mijloc]
        if return_steps:
            pasi.append((stanga, mijloc, dreapta, element))

        if element == valoare:
            return True, mijloc, pasi
        elif valoare < element:
            dreapta = mijloc - 1
        else:
            stanga = mijloc + 1

    return False, -1, pasi


def insert_sorted(lista, valoare):
    if not lista:
        lista.append(valoare)
        return
    stanga, dreapta = 0, len(lista)
    while stanga < dreapta:
        mijloc = (stanga + dreapta) // 2
        if lista[mijloc] < valoare:
            stanga = mijloc + 1
        else:
            dreapta = mijloc
    lista.insert(stanga, valoare)


scoruri = [10, 20, 35, 50, 75, 100]

ALIASE_1 = {"1", "afiseaza", "afiseaza lista", "lista"}
ALIASE_2 = {"2", "cauta", "cauta scor", "search"}
ALIASE_3 = {"3", "adauga", "adauga scor", "add"}
ALIASE_4 = {"4", "iesire", "exit", "quit"}

while True:
    print("\nMENIU")
    print("1. Afiseaza lista de scoruri")
    print("2. Cauta un scor (binary search)")
    print("3. Adauga un scor (lista ramane sortata)")
    print("4. Iesire")

    alegere = input("\nAlege optiunea (1-4): ").strip().lower()

    if alegere in ALIASE_1:
        if not scoruri:
            print("\nLista este goala.")
        else:
            print("\nScoruri:", ", ".join(str(x) for x in scoruri))

    elif alegere in ALIASE_2:
        val = input("Introdu scorul cautat: ").strip()
        try:
            val = int(val)
        except ValueError:
            print("Te rog introdu un numar intreg valid.")
            continue

        vsteps = input("Vrei sa vezi pasii? (da/nu): ").strip().lower()
        show_steps = vsteps in {"da", "d", "y", "yes"}

        gasit, idx, pasi = binary_search(scoruri, val, return_steps=show_steps)
        if gasit:
            print(f"Scorul {val} a fost gasit la indexul {idx}.")
        else:
            print(f"Scorul {val} nu exista in lista.")

        if show_steps:
            print("\nPasi (stanga, mijloc, dreapta, element_la_mijloc):")
            for st, mj, dr, el in pasi:
                print(f"({st}, {mj}, {dr}, {el})")

    elif alegere in ALIASE_3:
        val = input("Introdu scorul de adaugat: ").strip()
        try:
            val = int(val)
        except ValueError:
            print("Te rog introdu un numar intreg valid.")
            continue
        insert_sorted(scoruri, val)
        print("Scor adaugat. Lista ramane sortata.")
        print("Scoruri:", ", ".join(str(x) for x in scoruri))

    elif alegere in ALIASE_4:
        print("\nProgramul s-a incheiat. La revedere!")
        break

    else:
        print("Optiune invalida. Incearca din nou.")
