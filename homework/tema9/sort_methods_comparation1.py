def bubble_sort(lista):
    n = len(lista)
    for i in range(n):
        for j in range(0, n - i - 1):
            if lista[j][1] > lista[j + 1][1]:
                lista[j], lista[j + 1] = lista[j + 1], lista[j]
    return lista


def quick_sort(lista):
    if len(lista) <= 1:
        return lista
    pivot = lista[len(lista) // 2][1]
    stanga = [x for x in lista if x[1] < pivot]
    mijloc = [x for x in lista if x[1] == pivot]
    dreapta = [x for x in lista if x[1] > pivot]
    return quick_sort(stanga) + mijloc + quick_sort(dreapta)


def builtin_sort(lista):
    return sorted(lista, key=lambda x: x[1])


produse = {
    "paine": 5,
    "lapte": 7,
    "oua": 12,
    "carne": 30,
    "ciocolata": 10
}

while True:
    print("\n=== MENIU ===")
    print("1. Bubble Sort")
    print("2. Quick Sort")
    print("3. Built-in Sort (TimSort)")
    print("4. Iesire")

    alegere = input("\nAlege o optiune (1-4): ").strip().lower()

    if alegere == "1":
        lista = list(produse.items())
        rezultat = bubble_sort(lista)
        print("\nRezultat Bubble Sort:")
        for nume, pret in rezultat:
            print(f"- {nume}: {pret} lei")

    elif alegere == "2":
        lista = list(produse.items())
        rezultat = quick_sort(lista)
        print("\nRezultat Quick Sort:")
        for nume, pret in rezultat:
            print(f"- {nume}: {pret} lei")

    elif alegere == "3":
        lista = list(produse.items())
        rezultat = builtin_sort(lista)
        print("\nRezultat Built-in Sort (TimSort):")
        for nume, pret in rezultat:
            print(f"- {nume}: {pret} lei")

    elif alegere == "4":
        print("\nProgramul s-a incheiat. La revedere!")
        break

    else:
        print("\nOptiune invalida. Incearca din nou.")
