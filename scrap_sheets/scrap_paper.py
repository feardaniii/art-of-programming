def greet():
    print("Hi there!")
greet()

def multiply(a, b):
    return a * b
result = multiply(20, 20)
print(result)

def cart_total(prices, cart):
    
    # prices: dict mapping item -> unit price
    # cart: dict mapping item -> quantity
    # returns total cost
    
    total = 0.0
    for item, qty in cart.items():
        unit = prices.get(item, 0)      # price 0 if item missing
        total += unit * qty
    return total

# Example:
prices = {"apple": 0.5, "milk": 1.2, "bread": 1.0}
cart = {"apple": 4, "bread": 2}
print(cart_total(prices, cart))  


# A sample dictionary
prices = {"apple": 0.5, "milk": 1.2, "bread": 1.0}

print("\nWelcome to the dictionary playground!")
print("We have this dictionary:", prices)

while True:
    print("\nOptions:")
    print("1 - Show all items with .items()")
    print("2 - Get a value with [] (may crash if missing)")
    print("3 - Get a value with .get() (safe way)")
    print("4 - Quit")

    choice = input("Choose an option (1-4): ")

    if choice == "1":
        print("\nLooping with .items():")
        for key, value in prices.items():
            print(f"{key} -> {value}")

    elif choice == "2":
        key = input("Enter a key to look up (e.g., apple): ")
        try:
            print(f"Value: {prices[key]}")
        except KeyError:
            print("That key does not exist! (KeyError)")

    elif choice == "3":
        key = input("Enter a key to look up (e.g., apple): ")
        default = input("Enter a default value (or leave blank): ")
        if default == "":
            value = prices.get(key)
        else:
            value = prices.get(key, default)
        print(f"Value: {value}")

    elif choice == "4":
        print("Goodbye!")
        break

    else:
        print("Invalid choice. Please enter 1-4.")