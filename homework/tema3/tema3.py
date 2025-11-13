# 📈 Tema pentru acasă avansati

# - Creează o buclă care cere un PIN de 4 cifre până e corect. - La 3 încercări greșite - timeout 15-> apoi 60 (a doua greșeală) -> apoi lockout complet.
# - Desenează diferite forme cu “*” (user inputs option: circle, triangle, hexagon, etc)
# - Scrie o buclă care adună toate numerele pare între 1 și 10.000 (O(n) cât mai mic cu explicații)




# Exercitiu 1

import time

pin_corect = "1234"
incercari = 0
max_incercari = 5

while incercari < max_incercari:
    pin = input("\nIntrodu PIN-ul: ")
    incercari += 1

    if pin == pin_corect:
        print(f"\nPIN corect! Acces permis! \n\nIncercari: {incercari}\n")
        break

    if incercari == 3:
        print("Incearca din nou peste 15 secunde!")
        time.sleep(15)

    if incercari == 4:
        print("Incearca din nou peste 30 secunde!")
        time.sleep(30)

    if incercari == 5:
        print("Blocare definitiva. Acces refuzat\n")
    
    else:
        print("PIN gresit. Incearca din nou!")




# Exercitiu 2

# Triunghi
time.sleep(3)
print("\n\n")
linii_triunghi = 6
for i in range(1, linii_triunghi + 1):
    print("*" * i)


# Hexagon
print("\n\n")
n = 4

for i in range(1, n + 1):
    spaces = n - i
    stars = n + i
    print(" " * spaces + "*" * stars)

for _ in range(n):
    print("*" * (2 * n))

for i in range(n, 0, -1):
    spaces = n - i
    stars = n + i
    print(" " * spaces + "*" * stars)
print("\n\n")




# Exercitiu 3 - Mod 1

time.sleep(3)

print("timp_start: ",time.ctime())

total = 0
timp_start = time.time()

for x in range(2, 10001, 2):
    total += x       # Aduna x (toate numerele din range) la total (care este setat ca fiind 0)
timp_end = time.time()
print("\n",total)
print("timp_pas1: ",time.ctime())
print("timp metoda 1: ",timp_end - timp_start," secunde")

# Exercitiu 3 - Mod 2

timp_start = time.time()
print("\n",sum(range(2, 10001, 2)))
timp_end = time.time()
print("timp_pas2: ",time.ctime())
print("timp metoda 2: ",timp_end - timp_start," secunde")