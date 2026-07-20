from google import genai
import os, json, re
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return {"action": "error", "args": {}, "message": text}


def ask_llm(user_input, memory):

    prompt = f"""
You are a retail AI agent.

Return ONLY JSON:

Allowed actions:
- show_products
- sell_product (id, quantity)
- delete_product (id)

Memory:
{memory}

User:
{user_input}

Return format:
{{"action": "...", "args": {{}}}}
"""

    response = client.models.generate_content(
        model="models/gemini-2.5-flash-lite",
        contents=prompt
    )

    return extract_json(response.text)