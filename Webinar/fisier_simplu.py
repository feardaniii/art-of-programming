# import time
#
#
# print("Reduceri de vara 50%")
# time.sleep(5)
# print("Pana in 15 August")



















# 5000 de lei
cont_maria = 5000
pret_frigider_fara_tva = 2500
tva = 0.19  # 19%

# Calculăm prețul cu TVA
pret_frigider = pret_frigider_fara_tva * (1 + tva)
print(f"Preț fără TVA: {pret_frigider_fara_tva} lei")
print(f"Preț cu TVA: {pret_frigider} lei")

# Simulam actiunea
# Maria Cumpara Frigider
cont_maria = cont_maria - pret_frigider
print(f"Cont Maria după cumpărare: {cont_maria} lei")