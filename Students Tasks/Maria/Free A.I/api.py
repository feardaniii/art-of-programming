from fastapi import FastAPI, Header

from auth import login, verify
from database import init_db, get_products
from chat import ask_llm
from agent import handle
from memory import get_memory, save_memory

app = FastAPI()


@app.on_event("startup")
def start():
    init_db()


@app.post("/login")
def login_api(data: dict):
    token = login(data.get("username"), data.get("password"))

    if not token:
        return {"error": "Invalid"}

    return {"token": token}


@app.post("/chat")
def chat_api(data: dict, authorization: str = Header(None)):

    if not authorization or not verify(authorization):
        return {"error": "Unauthorized"}

    msg = data.get("message")

    memory = get_memory(authorization)

    action = ask_llm(msg, memory)
    result = handle(action)

    memory.append({"role": "user", "content": msg})
    memory.append({"role": "assistant", "content": str(action)})

    save_memory(authorization, memory)

    return {"result": result}


@app.get("/products")
def products_api(authorization: str = Header(None)):

    if not authorization or not verify(authorization):
        return {"error": "Unauthorized"}

    return get_products()