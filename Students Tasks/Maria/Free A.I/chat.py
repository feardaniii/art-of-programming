from config import GOOGLE_API_KEY
import re

def ask_llm(text, memory):
    """
    Fallback simplu (fără AI real dacă nu ai API key)
    """

    if "show" in text:
        return {"action": "show_products"}

    match = re.search(r"(\d+).*?(\d+)", text)

    if "sell" in text and match:
        return {
            "action": "sell_product",
            "id": int(match.group(1)),
            "quantity": int(match.group(2))
        }

    return {"action": "show_products"}

#N.B. Asta garantează că aplicația merge chiar fără AI