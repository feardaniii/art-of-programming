from __future__ import annotations

# Standard library imports for text processing and file handling.
import re
import string
from collections import Counter
from pathlib import Path

# Third-party libraries for visualization.
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Minimal Romanian + English stopword list used during cleaning.
STOPWORDS = {
    "a", "ai", "al", "ale", "am", "ar", "asupra", "au", "avea", "avem", "azi",
    "ca", "care", "cat", "ce", "cel", "cei", "cele", "chiar", "cu", "cum",
    "daca", "dar", "de", "deci", "din", "doar", "dupa",
    "e", "el", "ele", "era", "este", "eu",
    "fi", "fie", "fost", "foarte", "fara",
    "i", "iar", "in", "insa", "intr", "intre", "isi",
    "la", "le", "li", "lui",
    "mai", "mea", "mele", "mereu", "mi", "mine", "mult", "multe", "multumesc",
    "ne", "ni", "nici", "noi", "nou", "nu",
    "o", "ori", "orice",
    "pe", "pentru", "peste", "poate", "prea", "prin",
    "sa", "sau", "se", "si", "spre", "sub", "sunt",
    "ta", "te", "ti", "toate", "tot", "toti", "tu",
    "un", "una", "unde", "unei", "unii", "unor", "va", "voi", "vostru",
    "an", "and", "are", "as", "at",
    "be", "been", "but", "by",
    "for", "from",
    "had", "has", "have", "he", "her", "his",
    "if", "into", "is", "it", "its",
    "of", "on", "or", "our", "out",
    "she", "so",
    "that", "the", "their", "them", "they", "this", "to",
    "was", "we", "were", "what", "which", "who", "with", "would",
    "you", "your",
}


# Step 1: Read the selected input text file.
def read_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


# Step 2: Normalize text and keep only meaningful tokens.
def clean_and_tokenize(text: str, stopwords: set[str] | None = None) -> list[str]:
    if stopwords is None:
        stopwords = set()

    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t.isalpha() and t not in stopwords]


# Step 3: Compute relative word frequency = count(word) / total_tokens.
def compute_relative_frequencies(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in counts.items()}


# Step 4: Show the top-N most frequent words in a readable table.
def print_top_frequencies(relative_freq: dict[str, float], top_n: int = 20) -> None:
    if not relative_freq:
        print("No words found after cleaning.")
        return

    print(f"\nTop {top_n} words by relative frequency:")
    print("-" * 45)
    print(f"{'Word':20s} {'Relative frequency':>20s}")
    print("-" * 45)
    for word, freq in sorted(relative_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        print(f"{word:20s} {freq:20.4f}")


# Step 5: Build and save a wordcloud image from frequencies.
def generate_wordcloud(relative_freq: dict[str, float], output_path: Path) -> None:
    if not relative_freq:
        print("Wordcloud skipped: no frequencies available.")
        return

    cloud = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        colormap="viridis",
    ).generate_from_frequencies(relative_freq)

    cloud.to_file(str(output_path))
    print(f"\nWordcloud saved to: {output_path}")

    plt.figure(figsize=(12, 7))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# Step 0: Let user choose the article file (GUI picker, then terminal fallback).
def choose_input_file() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="Choose a text file for analysis",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        root.destroy()
        if selected:
            return Path(selected)
    except Exception:
        pass

    while True:
        raw = input("Enter the path to a .txt file: ").strip().strip('"')
        path = Path(raw)
        if path.is_file() and path.suffix.lower() == ".txt":
            return path
        print("Invalid path. Please provide an existing .txt file.")


# Main flow: choose file -> clean/process -> show frequencies -> generate wordcloud.
def main() -> None:
    print("Word Frequency + Wordcloud Analyzer")
    print("=" * 40)

    input_path = choose_input_file()
    print(f"\nAnalyzing: {input_path}")

    text = read_text_file(input_path)
    tokens = clean_and_tokenize(text, stopwords=STOPWORDS)
    relative_freq = compute_relative_frequencies(tokens)

    print(f"Total tokens after cleaning: {len(tokens)}")
    print(f"Unique tokens: {len(relative_freq)}")
    print_top_frequencies(relative_freq, top_n=20)

    output_path = input_path.with_name(f"{input_path.stem}_wordcloud.png")
    generate_wordcloud(relative_freq, output_path)


if __name__ == "__main__":
    main()
