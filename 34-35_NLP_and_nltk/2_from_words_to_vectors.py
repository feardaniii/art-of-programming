"""
================================================================================
NLP PART 2: FROM WORDS TO VECTORS
================================================================================

Course: The Art of Programming - NLP Module
Lesson: How Machines Learned That Words Have Meaning

THE JOURNEY:
------------
1950s-1990s: Words are arbitrary symbols (no meaning)
1990s-2000s: Words are frequency counts (Bag of Words, TF-IDF)
2013:        Words are points in space (Word2Vec) ← THE REVOLUTION
2017:        Context is everything (Transformers, BERT, GPT)

THE CORE INSIGHT:
-----------------
"You shall know a word by the company it keeps" - J.R. Firth, 1957

This is the DISTRIBUTIONAL HYPOTHESIS:
    Words that appear in similar contexts have similar meanings.

    "The ___ barked at the mailman"     → dog, puppy, hound
    "The ___ meowed at the bird"        → cat, kitten, feline

    A computer that sees millions of sentences learns:
    - 'dog' and 'puppy' appear in similar contexts → similar meaning
    - 'dog' and 'democracy' appear in different contexts → different meaning

This insight powers ALL modern NLP, from search engines to ChatGPT.

================================================================================
"""

import numpy as np
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from dataclasses import dataclass

# =============================================================================
# PART 1: BAG OF WORDS - THE NAIVE APPROACH
# =============================================================================

"""
BAG OF WORDS: Treat a document as a bag (unordered collection) of words.

    "I love this movie" → {"I": 1, "love": 1, "this": 1, "movie": 1}
    "I love love love this" → {"I": 1, "love": 3, "this": 1}

PROS:
    - Simple to understand and implement
    - Works surprisingly well for many tasks

CONS:
    - Loses word order: "dog bites man" = "man bites dog"
    - All words treated equally: "the" counts as much as "revolution"
    - No notion of meaning: "good" and "excellent" are unrelated
"""


def bag_of_words_demo():
    """
    Show how Bag of Words works and its limitations.
    """
    print("=" * 70)
    print("BAG OF WORDS: The Naive Approach")
    print("=" * 70)
    print()

    # The classic example that shows BoW loses meaning
    sentences = [
        "dog bites man",
        "man bites dog",
    ]

    print("These sentences have OPPOSITE meanings:")
    for s in sentences:
        print(f"  '{s}'")
    print()

    # Create BoW representation
    def text_to_bow(text: str) -> Dict[str, int]:
        words = text.lower().split()
        return dict(Counter(words))

    bow1 = text_to_bow(sentences[0])
    bow2 = text_to_bow(sentences[1])

    print("Bag of Words representation:")
    print(f"  '{sentences[0]}' → {bow1}")
    print(f"  '{sentences[1]}' → {bow2}")
    print()
    print(f"Are they equal? {bow1 == bow2}")
    print()
    print("PROBLEM: Bag of Words thinks these are IDENTICAL!")
    print("         Word order (and therefore meaning) is lost.")
    print()

    # Another problem: common words dominate
    print("-" * 70)
    print("Another problem: Common words dominate")
    print("-" * 70)
    print()

    documents = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "quantum physics explains the universe",
    ]

    # Build vocabulary
    all_words = set()
    for doc in documents:
        all_words.update(doc.lower().split())
    vocab = sorted(all_words)

    print("Documents:")
    for i, doc in enumerate(documents):
        print(f"  {i + 1}. '{doc}'")
    print()

    print(f"Vocabulary: {vocab}")
    print()

    # Count 'the' vs meaningful words
    the_count = sum(doc.lower().split().count('the') for doc in documents)
    physics_count = sum(doc.lower().split().count('physics') for doc in documents)

    print(f"Word counts across all documents:")
    print(f"  'the': {the_count} occurrences")
    print(f"  'physics': {physics_count} occurrence")
    print()
    print("PROBLEM: 'the' appears 5x more than 'physics'")
    print("         But 'physics' tells us MUCH more about document 3!")
    print()
    print("SOLUTION: TF-IDF - weight words by importance")
    print()


bag_of_words_demo()

# =============================================================================
# PART 2: TF-IDF - WEIGHTING BY IMPORTANCE
# =============================================================================

"""
TF-IDF: Term Frequency × Inverse Document Frequency

THE INTUITION:
--------------
A word is important to a document if:
    1. It appears FREQUENTLY in that document (TF - Term Frequency)
    2. It appears RARELY in other documents (IDF - Inverse Document Frequency)

Why does this work?

    "the" appears in EVERY document → low IDF → low importance
    "quantum" appears in ONE document → high IDF → high importance

    TF-IDF("the", doc3) = high TF × low IDF = LOW score
    TF-IDF("quantum", doc3) = some TF × high IDF = HIGH score

This is elegant: we don't need a list of stop words!
The math automatically downweights common words.
"""


class TFIDFVectorizer:
    """
    TF-IDF implementation from scratch for understanding.

    This is what sklearn.TfidfVectorizer does under the hood.
    """

    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.documents = []

    def fit(self, documents: List[str]):
        """
        Learn vocabulary and IDF weights from documents.
        """
        self.documents = documents
        n_docs = len(documents)

        # Build vocabulary
        all_words = set()
        for doc in documents:
            all_words.update(doc.lower().split())
        self.vocabulary = {word: i for i, word in enumerate(sorted(all_words))}

        # Calculate IDF for each word
        # IDF = log(N / df) where df = number of documents containing the word
        doc_freq = Counter()
        for doc in documents:
            unique_words = set(doc.lower().split())
            doc_freq.update(unique_words)

        for word in self.vocabulary:
            df = doc_freq[word]
            # Add 1 to avoid division by zero (smoothing)
            self.idf[word] = math.log((n_docs + 1) / (df + 1)) + 1

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Convert documents to TF-IDF vectors.
        """
        n_docs = len(documents)
        n_features = len(self.vocabulary)
        matrix = np.zeros((n_docs, n_features))

        for doc_idx, doc in enumerate(documents):
            words = doc.lower().split()
            word_counts = Counter(words)

            # Calculate TF-IDF for each word
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    word_idx = self.vocabulary[word]
                    tf = count / len(words)  # Normalized term frequency
                    tfidf = tf * self.idf[word]
                    matrix[doc_idx, word_idx] = tfidf

        # L2 normalize each document vector
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        matrix = matrix / norms

        return matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)


def tfidf_demo():
    """
    Demonstrate TF-IDF and why it works.
    """
    print("=" * 70)
    print("TF-IDF: Weighting Words by Importance")
    print("=" * 70)
    print()

    documents = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "quantum physics explains the universe",
        "physics and mathematics describe nature",
    ]

    print("Documents:")
    for i, doc in enumerate(documents):
        print(f"  {i + 1}. '{doc}'")
    print()

    # Calculate TF-IDF
    vectorizer = TFIDFVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Show IDF values
    print("IDF VALUES (Inverse Document Frequency):")
    print("-" * 50)
    print()

    # Sort by IDF (most distinctive first)
    sorted_idf = sorted(vectorizer.idf.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Word':<15} {'IDF':<10} {'Interpretation'}")
    print("-" * 50)

    for word, idf_val in sorted_idf[:8]:
        if idf_val > 1.7:
            interpretation = "rare → distinctive"
        elif idf_val > 1.4:
            interpretation = "somewhat rare"
        else:
            interpretation = "common → less distinctive"
        print(f"{word:<15} {idf_val:<10.3f} {interpretation}")

    print()
    print("INSIGHT: 'quantum' has high IDF (appears in 1 doc)")
    print("         'the' has low IDF (appears in all docs)")
    print()

    # Show TF-IDF scores for specific words in each document
    print("TF-IDF SCORES:")
    print("-" * 50)
    print()

    focus_words = ['the', 'physics', 'quantum', 'cat']

    print(f"{'Doc':<6}", end="")
    for word in focus_words:
        print(f"{word:<12}", end="")
    print()
    print("-" * 50)

    for doc_idx in range(len(documents)):
        print(f"Doc {doc_idx + 1}  ", end="")
        for word in focus_words:
            if word in vectorizer.vocabulary:
                word_idx = vectorizer.vocabulary[word]
                score = tfidf_matrix[doc_idx, word_idx]
                print(f"{score:<12.3f}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()

    print()
    print("OBSERVATION:")
    print("  • 'the' has LOW scores everywhere (common word)")
    print("  • 'quantum' has HIGH score only in Doc 3 (distinctive)")
    print("  • 'physics' scores in Docs 3 & 4 (appears in both)")
    print()

    # Document similarity
    print("=" * 70)
    print("DOCUMENT SIMILARITY WITH TF-IDF")
    print("=" * 70)
    print()

    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    print("Similarity Matrix:")
    print()
    print(f"{'':>8}", end="")
    for i in range(len(documents)):
        print(f"Doc{i + 1:>5}", end="")
    print()
    print("-" * 35)

    for i in range(len(documents)):
        print(f"Doc {i + 1}   ", end="")
        for j in range(len(documents)):
            sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])
            print(f"{sim:>7.3f}", end="")
        print()

    print()
    print("INTERPRETATION:")
    print("  • Doc1 & Doc2: High similarity (both about animals sitting)")
    print("  • Doc3 & Doc4: High similarity (both about physics)")
    print("  • Doc1 & Doc3: Low similarity (different topics)")
    print()
    print("TF-IDF captured topic similarity without understanding meaning!")
    print()


tfidf_demo()


# =============================================================================
# PART 3: THE LIMITATIONS OF COUNT-BASED METHODS
# =============================================================================

def show_limitations():
    """
    Show why we need something beyond TF-IDF.
    """
    print("=" * 70)
    print("LIMITATIONS: Why TF-IDF Isn't Enough")
    print("=" * 70)
    print()

    # Synonyms are unrelated
    print("PROBLEM 1: Synonyms Are Unrelated")
    print("-" * 50)
    print()

    docs = [
        "the car is fast",
        "the automobile is quick",
    ]

    print("These sentences mean the same thing:")
    for doc in docs:
        print(f"  '{doc}'")
    print()

    vectorizer = TFIDFVectorizer()
    matrix = vectorizer.fit_transform(docs)

    def cosine_similarity(v1, v2):
        dot = np.dot(v1, v2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    sim = cosine_similarity(matrix[0], matrix[1])
    print(f"TF-IDF similarity: {sim:.3f}")
    print()
    print("PROBLEM: TF-IDF sees 'car' and 'automobile' as unrelated!")
    print("         Same with 'fast' and 'quick'.")
    print("         No understanding of synonyms.")
    print()

    # Antonyms look the same
    print("PROBLEM 2: Antonyms Look Similar")
    print("-" * 50)
    print()

    docs = [
        "this movie is good",
        "this movie is bad",
    ]

    print("These sentences have OPPOSITE meanings:")
    for doc in docs:
        print(f"  '{doc}'")
    print()

    vectorizer = TFIDFVectorizer()
    matrix = vectorizer.fit_transform(docs)
    sim = cosine_similarity(matrix[0], matrix[1])

    print(f"TF-IDF similarity: {sim:.3f}")
    print()
    print("PROBLEM: High similarity despite opposite meanings!")
    print("         'good' and 'bad' share similar contexts.")
    print()

    # No semantic relationships
    print("PROBLEM 3: No Semantic Relationships")
    print("-" * 50)
    print()
    print("TF-IDF cannot answer:")
    print("  • Is 'king' related to 'queen'?")
    print("  • Is 'Paris' to 'France' as 'Tokyo' is to 'Japan'?")
    print("  • What words are similar to 'happy'?")
    print()
    print("We need a representation that captures MEANING.")
    print("Enter: Word Embeddings.")
    print()


show_limitations()

# =============================================================================
# PART 4: WORD EMBEDDINGS - THE REVOLUTION
# =============================================================================

"""
WORD EMBEDDINGS: Represent words as dense vectors in a continuous space.

THE KEY INSIGHT (Distributional Hypothesis):
    Words that appear in similar contexts have similar meanings.

    Context of "dog": "The ___ barked", "walked the ___", "pet ___"
    Context of "cat": "The ___ meowed", "petted the ___", "pet ___"

    'dog' and 'cat' share contexts → close in vector space

HOW WORD2VEC LEARNS (Skip-gram):
    Given "The quick brown fox jumps"

    Training examples (predict context from target):
        "brown" → predict "quick" (window=1)
        "brown" → predict "fox"   (window=1)

    The neural network learns word vectors such that:
        similar words → similar predictions → similar vectors

THE MAGIC:
    Nobody told the model that 'dog' and 'cat' are similar.
    Nobody defined what 'similar' means.
    The model DISCOVERED semantic relationships from context patterns.
"""


class SimpleWord2Vec:
    """
    A simplified Word2Vec implementation to understand the concept.

    Real Word2Vec uses neural networks and negative sampling.
    This version uses co-occurrence counting + SVD for clarity.
    """

    def __init__(self, vector_size: int = 50, window: int = 2):
        self.vector_size = vector_size
        self.window = window
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vectors = None

    def fit(self, sentences: List[List[str]]):
        """
        Learn word vectors from sentences.

        This uses the co-occurrence matrix approach:
        1. Count how often words appear near each other
        2. Apply SVD to reduce dimensionality
        3. The reduced vectors capture semantic relationships
        """
        # Build vocabulary
        words = set()
        for sentence in sentences:
            words.update(sentence)

        self.word_to_idx = {w: i for i, w in enumerate(sorted(words))}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        vocab_size = len(self.word_to_idx)

        # Build co-occurrence matrix
        cooccurrence = np.zeros((vocab_size, vocab_size))

        for sentence in sentences:
            for i, word in enumerate(sentence):
                word_idx = self.word_to_idx[word]

                # Look at context window
                start = max(0, i - self.window)
                end = min(len(sentence), i + self.window + 1)

                for j in range(start, end):
                    if i != j:
                        context_word = sentence[j]
                        context_idx = self.word_to_idx[context_word]
                        cooccurrence[word_idx, context_idx] += 1

        # Apply log transform (PPMI-like)
        cooccurrence = np.log1p(cooccurrence)

        # SVD to get dense vectors
        U, S, Vt = np.linalg.svd(cooccurrence, full_matrices=False)

        # Take first vector_size dimensions
        k = min(self.vector_size, vocab_size)
        self.vectors = U[:, :k] * np.sqrt(S[:k])

        return self

    def get_vector(self, word: str) -> np.ndarray:
        """Get vector for a word."""
        if word not in self.word_to_idx:
            raise KeyError(f"Word '{word}' not in vocabulary")
        idx = self.word_to_idx[word]
        return self.vectors[idx]

    def most_similar(self, word: str, topn: int = 5) -> List[Tuple[str, float]]:
        """Find most similar words."""
        if word not in self.word_to_idx:
            raise KeyError(f"Word '{word}' not in vocabulary")

        word_vec = self.get_vector(word)

        similarities = []
        for other_word, idx in self.word_to_idx.items():
            if other_word != word:
                other_vec = self.vectors[idx]
                # Cosine similarity
                sim = np.dot(word_vec, other_vec) / (
                        np.linalg.norm(word_vec) * np.linalg.norm(other_vec) + 1e-8
                )
                similarities.append((other_word, sim))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:topn]

    def analogy(self, a: str, b: str, c: str, topn: int = 3) -> List[Tuple[str, float]]:
        """
        Solve analogy: a is to b as c is to ?

        Vector arithmetic: result = b - a + c

        Example: king - man + woman ≈ queen
        """
        vec_a = self.get_vector(a)
        vec_b = self.get_vector(b)
        vec_c = self.get_vector(c)

        # The famous equation
        result_vec = vec_b - vec_a + vec_c

        # Find nearest words to result
        similarities = []
        exclude = {a, b, c}

        for word, idx in self.word_to_idx.items():
            if word not in exclude:
                word_vec = self.vectors[idx]
                sim = np.dot(result_vec, word_vec) / (
                        np.linalg.norm(result_vec) * np.linalg.norm(word_vec) + 1e-8
                )
                similarities.append((word, sim))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:topn]


def word_embeddings_demo():
    """
    Demonstrate word embeddings and why they capture meaning.
    """
    print("=" * 70)
    print("WORD EMBEDDINGS: The Geometric Revolution")
    print("=" * 70)
    print()

    print("THE CORE IDEA:")
    print("-" * 50)
    print()
    print("'You shall know a word by the company it keeps' - J.R. Firth")
    print()
    print("Words appearing in similar contexts have similar meanings:")
    print()
    print("  'The ___ barked loudly'      → dog, puppy, hound")
    print("  'The ___ purred contentedly' → cat, kitten")
    print("  'The ___ flew overhead'      → bird, plane, eagle")
    print()
    print("A model trained on millions of sentences learns:")
    print("  • 'dog' and 'puppy' share many contexts → similar vectors")
    print("  • 'dog' and 'democracy' share few contexts → different vectors")
    print()

    # Create a corpus that demonstrates the concept
    # This is small but structured to show the principle
    print("=" * 70)
    print("TRAINING ON A STRUCTURED CORPUS")
    print("=" * 70)
    print()

    # Corpus designed to have clear semantic relationships
    corpus = [
        # Royalty relationships
        ["the", "king", "rules", "the", "kingdom"],
        ["the", "queen", "rules", "the", "kingdom"],
        ["the", "king", "and", "queen", "live", "in", "palace"],
        ["the", "prince", "is", "son", "of", "king"],
        ["the", "princess", "is", "daughter", "of", "queen"],

        # Gender relationships
        ["the", "man", "works", "in", "the", "city"],
        ["the", "woman", "works", "in", "the", "city"],
        ["the", "boy", "plays", "in", "the", "park"],
        ["the", "girl", "plays", "in", "the", "park"],

        # Animal relationships
        ["the", "dog", "barks", "at", "strangers"],
        ["the", "puppy", "barks", "at", "strangers"],
        ["the", "cat", "meows", "for", "food"],
        ["the", "kitten", "meows", "for", "food"],

        # More context
        ["king", "and", "man", "are", "male"],
        ["queen", "and", "woman", "are", "female"],
        ["prince", "and", "boy", "are", "young", "male"],
        ["princess", "and", "girl", "are", "young", "female"],

        # Additional sentences for richer context
        ["the", "king", "wears", "a", "crown"],
        ["the", "queen", "wears", "a", "crown"],
        ["the", "man", "and", "woman", "are", "adults"],
        ["the", "dog", "and", "cat", "are", "pets"],
    ]

    print(f"Corpus: {len(corpus)} sentences")
    print()
    print("Sample sentences:")
    for sent in corpus[:5]:
        print(f"  '{' '.join(sent)}'")
    print("  ...")
    print()

    # Train embeddings
    print("Training word embeddings...")
    model = SimpleWord2Vec(vector_size=20, window=2)
    model.fit(corpus)
    print(f"Learned vectors for {len(model.word_to_idx)} words")
    print()

    # Show similar words
    print("=" * 70)
    print("DISCOVERING RELATIONSHIPS")
    print("=" * 70)
    print()

    test_words = ['king', 'dog', 'man']

    for word in test_words:
        try:
            similar = model.most_similar(word, topn=5)
            print(f"Words most similar to '{word}':")
            for similar_word, score in similar:
                print(f"  {similar_word:<12} {score:.3f}")
            print()
        except KeyError:
            print(f"'{word}' not in vocabulary")
            print()

    print("INSIGHT: The model discovered semantic relationships!")
    print("  • 'king' is similar to 'queen' (royalty)")
    print("  • 'dog' is similar to 'puppy', 'cat' (animals)")
    print("  • Nobody told the model these relationships!")
    print()

    # Word analogies
    print("=" * 70)
    print("WORD ANALOGIES: Vector Arithmetic")
    print("=" * 70)
    print()

    print("The famous equation:")
    print("  king - man + woman = ?")
    print()
    print("As vectors:")
    print("  vec('king') - vec('man') + vec('woman') ≈ vec('queen')")
    print()
    print("WHY does this work?")
    print("  vec('king') - vec('man') = 'royalty' direction")
    print("  Adding vec('woman') = 'royalty' + 'female' ≈ 'queen'")
    print()

    try:
        result = model.analogy('man', 'king', 'woman', topn=3)
        print("Results for 'man' is to 'king' as 'woman' is to ?:")
        for word, score in result:
            print(f"  {word:<12} {score:.3f}")
        print()

        # Try another analogy
        result2 = model.analogy('dog', 'puppy', 'cat', topn=3)
        print("Results for 'dog' is to 'puppy' as 'cat' is to ?:")
        for word, score in result2:
            print(f"  {word:<12} {score:.3f}")
        print()

    except KeyError as e:
        print(f"Note: {e}")
        print("(Our corpus is small - real Word2Vec uses billions of words)")

    print()
    print("NOTE: Our toy corpus has only 22 sentences.")
    print("Real Word2Vec (Google News) trained on 100 BILLION words.")
    print("With more data, analogies become remarkably accurate!")
    print()


word_embeddings_demo()


# =============================================================================
# PART 5: HOW WORD2VEC ACTUALLY LEARNS
# =============================================================================

def explain_word2vec_training():
    """
    Explain the actual Word2Vec training process.
    """
    print("=" * 70)
    print("HOW WORD2VEC LEARNS: The Training Process")
    print("=" * 70)
    print()

    print("TWO ARCHITECTURES:")
    print("-" * 50)
    print()
    print("1. SKIP-GRAM: Predict context from target word")
    print()
    print("   Sentence: 'The quick brown fox jumps'")
    print("   Target word: 'brown' (window=1)")
    print()
    print("   Training examples:")
    print("     Input: 'brown' → Predict: 'quick'")
    print("     Input: 'brown' → Predict: 'fox'")
    print()
    print("   The model learns: what contexts does 'brown' appear in?")
    print()

    print("2. CBOW (Continuous Bag of Words): Predict target from context")
    print()
    print("   Sentence: 'The quick brown fox jumps'")
    print("   Context: ['quick', 'fox'] (window=1)")
    print()
    print("   Training example:")
    print("     Input: ['quick', 'fox'] → Predict: 'brown'")
    print()
    print("   The model learns: what word fits this context?")
    print()

    print("-" * 50)
    print("THE INSIGHT:")
    print("-" * 50)
    print()
    print("By training millions of these predictions, the model learns")
    print("that words appearing in similar contexts get similar vectors.")
    print()
    print("  'dog' appears in: 'The ___ barked', 'walked the ___'")
    print("  'cat' appears in: 'The ___ meowed', 'petted the ___'")
    print("  'car' appears in: 'The ___ drove', 'parked the ___'")
    print()
    print("  → 'dog' and 'cat' share more contexts than 'dog' and 'car'")
    print("  → 'dog' and 'cat' end up closer in vector space")
    print()


explain_word2vec_training()


# =============================================================================
# PART 6: LIMITATIONS AND THE PATH FORWARD
# =============================================================================

def show_embedding_limitations():
    """
    Show limitations of word embeddings, setting up transformers.
    """
    print("=" * 70)
    print("LIMITATIONS OF WORD EMBEDDINGS")
    print("=" * 70)
    print()

    print("PROBLEM 1: One Vector Per Word (Polysemy)")
    print("-" * 50)
    print()
    print("  'bank' has ONE vector, but multiple meanings:")
    print("    'I went to the bank to deposit money'  (financial)")
    print("    'I sat on the river bank'              (geography)")
    print()
    print("  Word2Vec averages all meanings into one vector!")
    print()

    print("PROBLEM 2: No Context Sensitivity")
    print("-" * 50)
    print()
    print("  'The dog bit the man' vs 'The man bit the dog'")
    print()
    print("  Same word vectors, different meanings.")
    print("  Word order matters, but embeddings are static.")
    print()

    print("PROBLEM 3: Document Representation is Crude")
    print("-" * 50)
    print()
    print("  Common approach: average all word vectors")
    print()
    print("    doc_vector = mean([vec('the'), vec('dog'), vec('barked')])")
    print()
    print("  Problems:")
    print("    • 'not good' and 'good' have similar averages")
    print("    • Long documents dilute important words")
    print("    • Word order completely lost")
    print()

    print("THE SOLUTION: Attention & Transformers (2017)")
    print("-" * 50)
    print()
    print("  Key insight: Context should modify word meaning")
    print()
    print("    BERT: 'bank' gets DIFFERENT vectors based on context")
    print("    GPT:  Generates text by understanding full context")
    print()
    print("  This is what powers ChatGPT, Claude, and modern AI.")
    print()


show_embedding_limitations()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 70)
print("LESSON SUMMARY: From Words to Vectors")
print("=" * 70)
print()
print("THE JOURNEY:")
print("-" * 50)
print()
print("1. BAG OF WORDS")
print("   • Count word frequencies")
print("   • Loses word order: 'dog bites man' = 'man bites dog'")
print("   • All words equal: 'the' counts as much as 'revolution'")
print()
print("2. TF-IDF")
print("   • Weight by importance: TF × IDF")
print("   • Common words downweighted automatically")
print("   • Still no semantic understanding")
print()
print("3. WORD EMBEDDINGS (Word2Vec)")
print("   • Words as points in meaning-space")
print("   • Distributional hypothesis: context → meaning")
print("   • king - man + woman ≈ queen (DISCOVERED, not programmed!)")
print()
print("4. LIMITATIONS")
print("   • One vector per word (ignores polysemy)")
print("   • Static vectors (ignores context)")
print("   • Sets up the need for: Transformers!")
print()
print("THE KEY INSIGHT:")
print("-" * 50)
print()
print("'You shall know a word by the company it keeps'")
print()
print("This principle powers:")
print("  • Search engines (finding relevant documents)")
print("  • Recommendation systems (similar items)")
print("  • ChatGPT & Claude (understanding language)")
print()
print("NEXT: Attention mechanisms and Transformers")
print("      How context transforms meaning dynamically")
print()