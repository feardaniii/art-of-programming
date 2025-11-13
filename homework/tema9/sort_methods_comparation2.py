import random
import time

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

produse = {f"Produs_{i}": random.randint(1, 10000) for i in range(10_000)} #for i in range(1_000_000)}
lista_produse = list(produse.items())

start = time.time()
bubble_sort(lista_produse.copy())
end = time.time()
print(f"BubbleSort: {end - start:.2f} secunde")

start = time.time()
quick_sort(lista_produse.copy())
end = time.time()
print(f"QuickSort: {end - start:.2f} secunde")

start = time.time()
builtin_sort(lista_produse.copy())
end = time.time()
print(f"TimSort (built-in): {end - start:.2f} secunde")