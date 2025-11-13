class Persoana:
    """
    Clasa de bază (parent class)
    """
    def __init__(self, nume):
        self.nume = nume
    
    def vorbeste(self):
        """Metodă comună pentru toate persoanele"""
        return f"🗣️ {self.nume} vorbește: 'Salut!'"
    
    def prezinta(self):
        return f"👤 Sunt {self.nume}"

class Profesor(Persoana):
    """
    Clasa derivată (child class) - MOȘTENEȘTE din Persoana
    """
    def __init__(self, nume, specializare):
        super().__init__(nume)  # Apelează constructorul părinte
        self.specializare = specializare
    
    def preda(self):
        """Metodă NOUĂ, specifică doar profesorilor"""
        return f"📚 {self.nume} predă: 'Astăzi învățăm despre {self.specializare}!'"
    
    def prezinta(self):
        """Metodă SUPRASCRISĂ (override)"""
        return f"👨‍🏫 Sunt Prof. {self.nume}, specializat în {self.specializare}"

# Utilizare:
persoana = Persoana("Ana Popescu")
profesor = Profesor("Ion Marinescu", "Matematică")

# Persoana poate doar să vorbească:
print(persoana.vorbeste())    # ✅ Funcționează
# print(persoana.preda())     # ❌ AttributeError!

# Profesorul poate și să vorbească (moștenit) și să predea:
print(profesor.vorbeste())    # ✅ Moștenit din Persoana
print(profesor.preda())       # ✅ Metodă nouă din Profesor

# Polimorfism - aceeași interfață, comportamente diferite:
print(persoana.prezinta())    # "👤 Sunt Ana Popescu"
print(profesor.prezinta())    # "👨‍🏫 Sunt Prof. Ion..."