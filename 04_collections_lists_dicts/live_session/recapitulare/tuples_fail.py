# Crearea unui tuplu (coordonate fixe)
casa = (44.4268, 26.1025)
scoala = (44.4378, 26.0967)

# Accesarea elementelor
latitudine = casa[0]
longitudine = casa[1]

# Tuplurile nu se pot modifica!
casa[0] = 45.0  # Eroare!

# Parcurgerea unui tuplu
for coordonata in casa:
    print(coordonata)

# Unpacking (destructurare)
lat, lon = casa
print(f"Lat: {lat}, Long: {lon}")