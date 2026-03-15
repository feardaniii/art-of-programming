import os
import json
import requests
from datetime import datetime

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """You are Blue Pigeon Assistant, a senior Python developer and an expert Python teacher.
Your primary goal is to help users learn Python, debug their code, understand complex concepts, and write better Python code.

**Formatting Instructions (CRITICAL - Adhere Strictly):**
- **ALWAYS respond using clear, well-structured Markdown.** Your entire response body MUST be Markdown.
- **Utilize a variety of Markdown elements for readability and structure:**
    - Headings (e.g., `## Main Topic`, `### Sub-topic`) for organization.
    - Bold text (e.g., `**important concept**`) for emphasis.
    - Italic text (e.g., `*emphasized term*`) for nuance.
    - Unordered lists (e.g., `- First item`) for bullet points.
    - Ordered lists (e.g., `1. Step one`) for sequences.
    - Inline code (e.g., `variable_name`, `my_function()`) using single backticks for short code mentions.
    - **Multi-line code blocks for Python code snippets. ALWAYS specify the language, typically 'python':**
```python
      def example_function():
          return "This is Python code"
      print(example_function())
```
    - Blockquotes (e.g., `> This is a quote`) if relevant.
    - Tables if data is tabular.
- Ensure code blocks are complete and runnable examples where appropriate. Explain the code clearly.
- Keep explanations concise but thorough.
- Maintain a friendly, patient, and encouraging tone.
- If a user asks something unrelated to Python or programming, politely state that your expertise is in Python and offer to help with Python-related questions.
"""

CONVERSATIONS_DIR = "conversations"


def ensure_dir():
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)


def session_paths(session_id: str) -> tuple[str, str]:
    return (
        os.path.join(CONVERSATIONS_DIR, f"{session_id}.json"),
        os.path.join(CONVERSATIONS_DIR, f"{session_id}.md"),
    )


def load_history(json_path: str) -> list[dict]:
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        return data.get("messages", [])
    return []


def save_json(json_path: str, history: list[dict], session_id: str):
    payload = {
        "session_id": session_id,
        "model": "gpt-4.1-2025-04-14",
        "updated_at": datetime.now().isoformat(),
        "message_count": len(history),
        "messages": history,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_md(md_path: str, history: list[dict], session_id: str):
    lines = [f"# Blue Pigeon — Session `{session_id}`\n"]
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        ts = msg.get("timestamp", "")
        if role == "user":
            lines.append(f"---\n\n### 🧑 User  \n*{ts}*\n\n{content}\n")
        else:
            lines.append(f"### 🤖 Assistant  \n*{ts}*\n\n{content}\n")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))


class Conversation:
    def __init__(self, session_id: str | None = None):
        ensure_dir()
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_path, self.md_path = session_paths(self.session_id)
        self.history: list[dict] = load_history(self.json_path)

    def ask(self, message: str) -> str:
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.history.append({"role": "user", "content": message, "timestamp": ts})

        # Build API messages (no timestamp field — OpenAI doesn't want it)
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            {"role": m["role"], "content": m["content"]} for m in self.history
        ]

        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4.1-2025-04-14",
                "messages": api_messages,
                "max_completion_tokens": 2000,
            },
            timeout=60,
        )

        r.raise_for_status()
        data = r.json()
        assistant_content = data["choices"][0]["message"]["content"]
        ts_reply = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.history.append({
            "role": "assistant",
            "content": assistant_content,
            "timestamp": ts_reply,
        })

        # Persist both formats after every exchange
        save_json(self.json_path, self.history, self.session_id)
        save_md(self.md_path, self.history, self.session_id)

        return assistant_content

    def clear(self):
        self.history.clear()
        save_json(self.json_path, self.history, self.session_id)
        save_md(self.md_path, self.history, self.session_id)


def list_sessions() -> list[str]:
    ensure_dir()
    return sorted(
        f.replace(".json", "")
        for f in os.listdir(CONVERSATIONS_DIR)
        if f.endswith(".json")
    )


if __name__ == "__main__":
    import sys

    # Single-shot
    if len(sys.argv) > 1:
        conv = Conversation()
        print(conv.ask(" ".join(sys.argv[1:])))
        sys.exit(0)

    # Interactive REPL
    print("Blue Pigeon Assistant — Python tutor")
    print("Commands: /clear /new /sessions /load <id> /exit\n")

    # Show existing sessions
    sessions = list_sessions()
    if sessions:
        print(f"Found {len(sessions)} saved session(s). Latest: {sessions[-1]}")
        print("Type /load <id> to resume, or just start typing for a new session.\n")

    conv = Conversation()
    print(f"Session: {conv.session_id}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() == "/exit":
            print("Goodbye!")
            break

        if user_input.lower() == "/clear":
            conv.clear()
            print("Memory cleared.\n")
            continue

        if user_input.lower() == "/new":
            conv = Conversation()
            print(f"New session: {conv.session_id}\n")
            continue

        if user_input.lower() == "/sessions":
            for s in list_sessions():
                print(f"  {s}")
            print()
            continue

        if user_input.lower().startswith("/load"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /load <session_id>\n")
                continue
            conv = Conversation(session_id=parts[1])
            print(f"Loaded session: {conv.session_id} ({len(conv.history)} messages)\n")
            continue

        # Regular message
        try:
            response = conv.ask(user_input)
            print(f"\n{response}\n")
        except Exception as e:
            print(f"Error: {e}\n")