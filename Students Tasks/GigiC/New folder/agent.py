from database import get_products, delete_product, sell_product

def handle_ai_response(data):

    action = data.get("action")

    if action == "show_products":
        return get_products()

    elif action == "delete_product":
        return delete_product(data["args"]["id"])

    elif action == "sell_product":
        args = data["args"]
        return sell_product(args["id"], args["quantity"])

    return "Unknown action"