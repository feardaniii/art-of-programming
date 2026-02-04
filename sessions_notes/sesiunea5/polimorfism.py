class Animal:
    """
    Clasa de bază pentru toate animalele
    """
    def __init__(self, nume):
        self.nume = nume
    
    def vorbeste(self):
        """Metodă abstractă - va fi suprascrisă"""
        pass
    
    def mananca(self):
        return f"{self.nume} mănâncă"

class Caine(Animal):
    def vorbeste(self):
        return f"{self.nume}: Ham ham! 🐕"

class Pisica(Animal):
    def vorbeste(self):
        return f"{self.nume}: Miau! 🐱"

class Papagal(Animal):
    def vorbeste(self):
        return f"{self.nume}: Polly wants a cracker! 🦜"

class Peste(Animal):
    def vorbeste(self):
        return f"{self.nume}: Blub blub... (nu face sunet) 🐟"

# Crearea unei liste cu animale diferite:
animale = [
    Caine("Rex"),
    Pisica("Mimi"),
    Papagal("Tweety"),
    Peste("Goldy")
]

# POLIMORFISM în acțiune:
def concert_animal(lista_animale):
    """
    O singură funcție pentru toate animalele!
    """
    for animal in lista_animale:
        # Aceeași metodă, comportamente diferite:
        print(animal.vorbeste())

# Apelare:
concert_animal(animale)

# Rezultat:
# Rex: Ham ham! 🐕
# Mimi: Miau! 🐱  
# Tweety: Polly wants a cracker! 🦜
# Goldy: Blub blub... (nu face sunet) 🐟