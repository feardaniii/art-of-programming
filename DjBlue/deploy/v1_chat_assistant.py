import os
import requests

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
      # Your Python code here
      def example_function():
          return "This is Python code"
      print(example_function())
      ```
    - Blockquotes (e.g., `> This is a quote`) if relevant.
    - Tables if data is tabular.
- Ensure code blocks are complete and runnable examples where appropriate. Explain the code clearly.
- If providing instructions or steps, use ordered or unordered lists.
- Keep explanations concise but thorough.
- Maintain a friendly, patient, and encouraging tone.
- If a user asks something unrelated to Python or programming, politely state that your expertise is in Python and offer to help with Python-related questions.
"""


def ask(message: str) -> str:
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4.1-2025-04-14",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            "max_completion_tokens": 2000,
        },
        timeout=60,
    )

    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    import sys

    # Single-shot mode: python main.py "your question here"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(ask(question))

    # Interactive REPL mode: python main.py
    else:
        print("Blue Pigeon Assistant — Python tutor (type 'exit' to quit)\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break

            try:
                response = ask(user_input)
                print(f"\nAssistant:\n{response}\n")
            except Exception as e:
                print(f"Error: {e}\n")