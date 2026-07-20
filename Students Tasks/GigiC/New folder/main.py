from chat import ask_llm
from agent import handle_ai_response
from memory import save_memory, load_memory
from database import init_db

init_db()

def format_output(result):

    if isinstance(result, list):
        if not result:
            return "Nu există produse."

        text = "\nProduse disponibile:\n"
        for p in result:
            text += f"- {p['name']} ({p['stock']} buc) - {p['price']} lei\n"
        return text

    return str(result)


def main():
    print("AI Retail Agent (PRO)")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        memory = load_memory()

        ai_decision = ask_llm(user_input, memory)

        result = handle_ai_response(ai_decision)

        output = format_output(result)

        print("AI:", output)

        save_memory(user_input, output)


if __name__ == "__main__":
    main()