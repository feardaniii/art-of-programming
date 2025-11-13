# user = input("Utilizator: ")
# parola = input("Parolă: ")
# cod = input("Cod de acces: ")
# if user == "admin" and parola == "1234" and cod == "123":
#     print("Acces complet")
# else:
#     print("Eroare")



user = input("\nUtilizator: ")
parola = input("\nParolă: ")
cod = input("\nCod de acces: ")

print("\nHai sa ne logam acum!")

verificare_user = input("\nIntrodu username-ul tau: ")
verificare_parola = input("\nIntrodu parola ta: ")
verificare_cod = input("\nIntrodu codul tau: ")

if user == verificare_user and parola == verificare_parola and cod == verificare_cod:
    print("\nAcces complet")

else:
    print("\nAi introdus ceva gresit!", "\n\n")