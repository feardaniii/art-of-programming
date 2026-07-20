import json

FILE = "memory.json"


def load_memory():
    try:
        with open(FILE, "r") as f:
            data = json.load(f)
            return data[-10:]  # LIMIT CONTEXT (FOARTE IMPORTANT)
    except:
        return []


def save_memory(user_input, ai_output):
    memory = load_memory()

    memory.append({
        "user": user_input,
        "ai": str(ai_output)
    })

    with open(FILE, "w") as f:
        json.dump(memory, f, indent=2)