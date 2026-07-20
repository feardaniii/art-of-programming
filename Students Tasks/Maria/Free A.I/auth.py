import uuid

USERS = {"admin": "1234"}
SESSIONS = {}

def login(username, password):
    if USERS.get(username) != password:
        return None

    token = str(uuid.uuid4())
    SESSIONS[token] = username
    return token

def verify(token):
    return token in SESSIONS