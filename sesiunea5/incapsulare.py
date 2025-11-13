class ContBancar:
    """
    Clasă cu incapsulare - sold privat, acces controlat
    """
    def __init__(self, nume, sold_initial=0):
        self.nume = nume
        self.__sold = sold_initial      # Atribut PRIVAT
        self.__este_blocat = False      # Atribut PRIVAT
    
    def depune(self, suma):
        """Metodă publică pentru depunere"""
        if self.__este_blocat:
            return "❌ Contul este blocat!"
        
        if suma > 0:
            self.__sold += suma
            return f"✅ Depozit: +{suma} RON"
        return "❌ Suma trebuie să fie pozitivă!"
    
    def retrage(self, suma):
        """Metodă publică pentru retragere"""
        if self.__este_blocat:
            return "❌ Contul este blocat!"
        
        if suma > self.__sold:
            return f"❌ Fonduri insuficiente! Sold: {self.__sold}"
        
        if suma > 0:
            self.__sold -= suma
            return f"✅ Retragere: -{suma} RON"
        return "❌ Suma trebuie să fie pozitivă!"
    
    def consulta_sold(self):
        """Metodă publică pentru consultare sold"""
        return self.__sold
    
    def blocheaza_cont(self):
        """Metodă pentru blocarea contului"""
        self.__este_blocat = True
    
    def deblocheaza_cont(self):
        """Metodă pentru deblocarea contului"""
        self.__este_blocat = False

# Utilizare:
cont = ContBancar("Ana Popescu", 1500)

# Accesul direct la __sold NU funcționează:
# print(cont.__sold)  # AttributeError!

# Accesul controlat prin metode:
print(cont.consulta_sold())    # ✅ 1500
print(cont.depune(500))        # ✅ Depozit: +500 RON  
print(cont.retrage(200))       # ✅ Retragere: -200 RON