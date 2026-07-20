TOOLS = [
    {
        "function_declarations": [
            {"name": "show_products", "parameters": {}},
            {
                "name": "sell_product",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "quantity": {"type": "integer"}
                    }
                }
            }
        ]
    }
]