from database import get_products, sell_product

def handle(action):
    if action["action"] == "show_products":
        return get_products()

    if action["action"] == "sell_product":
        return sell_product(action.get("id"), action.get("quantity"))

    return "Unknown action"