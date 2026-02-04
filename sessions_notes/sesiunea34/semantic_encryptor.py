# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np

# sentences = [
#     "The programmer wrote code for the application",
#     "The developer built software for the project"
# ]

# # Create TF-IDF vectors
# vectorizer = TfidfVectorizer()
# tfidf = vectorizer.fit_transform(sentences).toarray()

# # Normalize and round for similarity-preserving encoding
# def encode_vector(vec, decimals=2):
#     normalized = vec / np.linalg.norm(vec)
#     return "".join(f"{int(v * 100):02d}" for v in normalized)

# encoded_C = encode_vector(tfidf[0])
# encoded_D = encode_vector(tfidf[1])

# print("Encoded C:", encoded_C)
# print("Encoded D:", encoded_D)




# import string

# # Step 1: synonym normalization
# SYNONYMS = {
#     "programmer": "developer",
#     "wrote": "built",
#     "code": "software",
#     "application": "project"
# }

# def normalize(sentence):
#     words = sentence.lower().translate(
#         str.maketrans("", "", string.punctuation)
#     ).split()
#     return [SYNONYMS.get(word, word) for word in words]

# # Step 2: Caesar cipher
# def caesar_cipher(text, shift=4):
#     result = ""
#     for char in text:
#         if char.isalpha():
#             base = ord('a')
#             result += chr((ord(char) - base + shift) % 26 + base)
#         else:
#             result += char
#     return result

# def encrypt(sentence):
#     normalized_words = normalize(sentence)
#     normalized_words.sort()
#     joined = " ".join(normalized_words)
#     return caesar_cipher(joined)

# C = "The programmer wrote code for the application"
# D = "The developer built software for the project"

# print("Encrypted C:", encrypt(C))
# print("Encrypted D:", encrypt(D))



# import numpy as np

# # Two sentences
# sentence1 = "The programmer wrote code for the application"
# sentence2 = "The developer built software for the project"

# # Simple function to assign a small random vector to each word
# def word_to_vector(word, dim=10):
#     np.random.seed(sum(ord(c) for c in word) % 1000)
#     return np.random.rand(dim) * 0.1 + 0.45  # small values around 0.45

# # Convert a sentence to an "array" by averaging word vectors
# def sentence_to_array(sentence):
#     words = sentence.lower().split()
#     vectors = np.array([word_to_vector(w) for w in words])
#     return np.mean(vectors, axis=0)

# arr1 = sentence_to_array(sentence1)
# arr2 = sentence_to_array(sentence2)

# print("Array 1:", arr1)
# print("Array 2:", arr2)
# print("Distance:", np.linalg.norm(arr1 - arr2))


import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ==================================================
# CONFIGURARE
# ==================================================

SYNONYMS = {
    "programmer": "developer",
    "developer": "developer",
    "wrote": "built",
    "built": "built",
    "code": "software",
    "software": "software",
    "application": "project",
    "app": "project",
    "project": "project",
    "created": "built"
}

CAESAR_SHIFT = 4

# ==================================================
# METODA #4 — CRIPTARE SEMANTICĂ
# ==================================================

def normalize_sentence(sentence):
    """
    Curăță textul și înlocuiește sinonimele
    pentru a păstra sensul.
    """
    words = sentence.lower().translate(
        str.maketrans("", "", string.punctuation)
    ).split()

    return [SYNONYMS.get(word, word) for word in words]


def caesar_cipher(text, shift=CAESAR_SHIFT):
    """
    Aplică un cifru Caesar pentru ofuscare.
    """
    encrypted = ""
    for char in text:
        if char.isalpha():
            base = ord('a')
            encrypted += chr((ord(char) - base + shift) % 26 + base)
        else:
            encrypted += char
    return encrypted


def semantic_encrypt(sentence):
    """
    Pipeline complet:
    normalizare → sortare → criptare
    """
    normalized_words = normalize_sentence(sentence)
    normalized_words.sort()
    joined = " ".join(normalized_words)
    encrypted = caesar_cipher(joined)

    return normalized_words, encrypted


# ==================================================
# METODA #2 — TF-IDF + VECTORI
# ==================================================

def tfidf_encode(sentences):
    """
    Transformă propozițiile în vectori TF-IDF
    și returnează atât vectorii cât și versiunea lizibilă.
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences).toarray()

    readable_vectors = []
    for vec in vectors:
        norm = vec / np.linalg.norm(vec)
        readable_vectors.append(" ".join(f"{v:.2f}" for v in norm))

    return vectors, readable_vectors


def cosine_similarity(v1, v2):
    """
    Calculează similaritatea cosinus dintre doi vectori.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ==================================================
# INTERFAȚĂ INTERACTIVĂ
# ==================================================

def main():
    print("=" * 70)
    print("🔐 DEMO: CRIPTARE ȘI CODIFICARE CARE PĂSTREAZĂ SIMILARITATEA")
    print("=" * 70)

    print("\nAcest program aplică două metode diferite asupra")
    print("a două propoziții pentru a evalua similaritatea semantică.\n")

    s1 = input("Introdu Propoziția 1:\n> ")
    s2 = input("\nIntrodu Propoziția 2:\n> ")

    sentences = [s1, s2]

    # -----------------------------
    # METODA #4
    # -----------------------------
    print("\n" + "-" * 70)
    print("METODA #4 — Criptare semantică\n")

    norm1, enc1 = semantic_encrypt(s1)
    norm2, enc2 = semantic_encrypt(s2)

    print("📘 Propoziția 1")
    print("Cuvinte normalizate:", norm1)
    print("Text criptat       :", enc1)

    print("\n📗 Propoziția 2")
    print("Cuvinte normalizate:", norm2)
    print("Text criptat       :", enc2)

    # -----------------------------
    # METODA #2
    # -----------------------------
    print("\n" + "-" * 70)
    print("METODA #2 — Codificare TF-IDF\n")

    vectors, readable_vectors = tfidf_encode(sentences)

    print("📘 Vector TF-IDF Propoziția 1:")
    print(readable_vectors[0])

    print("\n📗 Vector TF-IDF Propoziția 2:")
    print(readable_vectors[1])

    # -----------------------------
    # SIMILARITATE COSINUS
    # -----------------------------
    similarity = cosine_similarity(vectors[0], vectors[1])

    print("\n" + "-" * 70)
    print(f"🔢 Similaritate cosinus între vectori: {similarity:.3f}")

    if similarity > 0.8:
        print("➡️ Interpretare: SIMILARITATE FOARTE MARE")
    elif similarity > 0.5:
        print("➡️ Interpretare: SIMILARITATE MEDIE")
    else:
        print("➡️ Interpretare: SIMILARITATE MICĂ")

    if enc1 == enc2:
        print("\n✅ Criptare semantică: rezultate IDENTICE")
    else:
        print("\nℹ️ Criptare semantică: rezultate SIMILARE, dar diferite")

    print("\nProces finalizat ✔️")
    print("=" * 70)


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
