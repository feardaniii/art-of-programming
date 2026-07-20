MEMORY = {}

def get_memory(token):
    return MEMORY.get(token, [])

def save_memory(token, mem):
    MEMORY[token] = mem[-10:]