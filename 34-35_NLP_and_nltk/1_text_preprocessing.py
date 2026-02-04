"""
================================================================================
NLP PART 1: TEXT PREPROCESSING
================================================================================

Course: The Art of Programming - NLP Module
Lesson: Why Computers Can't Read (And How We Fix It)

THE FUNDAMENTAL PROBLEM:
------------------------
Computers see text as arbitrary symbols. To a computer:
    "cat" and "dog" are equally different from "cat" and "cats"

    cat  = [99, 97, 116]        (ASCII codes)
    cats = [99, 97, 116, 115]   (one more number)
    dog  = [100, 111, 103]      (completely different numbers)

The computer has NO IDEA that "cat" and "cats" are related,
or that "cat" and "dog" are both animals.

Text preprocessing is about transforming messy human language
into a form where MEANING is preserved and NOISE is removed.

================================================================================
"""

import re
import string
from collections import Counter
from typing import List, Tuple, Callable
from dataclasses import dataclass

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

# Download required data (only needs to run once)
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)


# =============================================================================
# PART 1: THE PROBLEM - WHY PREPROCESSING MATTERS
# =============================================================================

def demonstrate_the_problem():
    """
    Show why raw text is problematic for computers.

    This is the "aha moment" - students need to understand the problem
    before we show them the solution.
    """
    print("=" * 70)
    print("THE PROBLEM: Why Computers Can't Understand Text")
    print("=" * 70)
    print()

    # Same meaning, different representations
    sentences = [
        "I love this movie!",
        "I LOVE this movie!!!",
        "i love this movie",
        "I loved this movie!",
        "I'm loving this movie!",
    ]

    print("These sentences mean roughly the same thing:")
    print("-" * 50)
    for s in sentences:
        print(f"  '{s}'")

    print()
    print("But to a computer (comparing raw strings):")
    print("-" * 50)

    base = sentences[0]
    for s in sentences[1:]:
        match = "✓ MATCH" if s == base else "✗ DIFFERENT"
        print(f"  '{base}' vs '{s}'")
        print(f"     → {match}")
        print()

    print("PROBLEM: 5 ways to say the same thing = 5 'different' inputs")
    print()
    print("Without preprocessing, a model sees these as completely unrelated.")
    print("It would need to learn each variation separately!")
    print()

    # The vocabulary explosion problem
    print("=" * 70)
    print("THE VOCABULARY EXPLOSION")
    print("=" * 70)
    print()

    word_forms = {
        "run": ["run", "runs", "ran", "running", "runner", "runners"],
        "happy": ["happy", "happier", "happiest", "happily", "happiness", "unhappy"],
        "go": ["go", "goes", "went", "going", "gone"],
    }

    total_forms = sum(len(forms) for forms in word_forms.values())

    print(f"Just 3 root words → {total_forms} different forms:")
    print()
    for root, forms in word_forms.items():
        print(f"  {root}: {forms}")

    print()
    print("In a real corpus:")
    print("  - English has ~170,000 words in common use")
    print("  - Each word averages ~5 inflected forms")
    print("  - That's 850,000 'different' tokens for a model to learn!")
    print()
    print("SOLUTION: Preprocessing reduces this complexity.")
    print("  'running', 'runs', 'ran' → 'run' (one token to learn)")
    print()


demonstrate_the_problem()

# =============================================================================
# PART 2: TOKENIZATION - SPLITTING TEXT INTO UNITS
# =============================================================================

"""
TOKENIZATION: The first decision that changes everything.

What is a token? It depends on your task!

    - Word tokens: "I", "love", "NLP"  (most common)
    - Character tokens: "I", " ", "l", "o", "v", "e"  (used in some deep learning)
    - Subword tokens: "un", "##believ", "##able"  (used by BERT, GPT)
    - Sentence tokens: ["First sentence.", "Second sentence."]

The tokenizer you choose affects everything downstream.
"""


def explore_tokenization():
    """
    Show different tokenization approaches and their trade-offs.
    """
    print("=" * 70)
    print("TOKENIZATION: Breaking Text Into Units")
    print("=" * 70)
    print()

    # Tricky examples that break naive tokenization
    tricky_texts = [
        "Dr. Smith went to Washington D.C. yesterday.",  # Abbreviations
        "I'll be there at 3:30pm - don't be late!",  # Contractions, times
        "The price is $19.99 (20% off!).",  # Numbers, symbols
        "Email me at john.doe@example.com ASAP.",  # Email, acronym
        "I can't believe it's not butter!",  # Multiple contractions
    ]

    print("Naive tokenization (split on spaces):")
    print("-" * 50)

    for text in tricky_texts[:2]:
        naive_tokens = text.split()
        print(f"  Text: {text}")
        print(f"  Tokens: {naive_tokens}")
        print(f"  Problems: ", end="")

        # Identify problems
        problems = []
        if any('.' in t and t != '.' for t in naive_tokens):
            problems.append("periods stuck to words")
        if any("'" in t for t in naive_tokens):
            problems.append("contractions unsplit")
        if any(t.endswith(',') or t.endswith('!') for t in naive_tokens):
            problems.append("punctuation attached")

        print(", ".join(problems) if problems else "none obvious")
        print()

    print("NLTK word_tokenize (smarter):")
    print("-" * 50)

    for text in tricky_texts[:2]:
        nltk_tokens = word_tokenize(text)
        print(f"  Text: {text}")
        print(f"  Tokens: {nltk_tokens}")
        print()

    # Show the difference in vocabulary size
    print("Impact on vocabulary size:")
    print("-" * 50)

    corpus = " ".join(tricky_texts)

    naive_vocab = set(corpus.split())
    nltk_vocab = set(word_tokenize(corpus))

    print(f"  Naive split: {len(naive_vocab)} unique tokens")
    print(f"  NLTK tokenize: {len(nltk_vocab)} unique tokens")
    print()
    print("  Naive includes: 'D.C.', 'yesterday.', '$19.99' (with punctuation)")
    print("  NLTK separates: 'D.C.', 'yesterday', '.', '$', '19.99'")
    print()

    # Sentence tokenization
    print("Sentence tokenization:")
    print("-" * 50)

    paragraph = "Dr. Smith earned $1.5M last year. That's impressive! Is it true? Yes, according to the SEC filing dated Jan. 15th."

    print(f"  Text: {paragraph}")
    print()

    # Naive approach
    naive_sentences = paragraph.split('. ')
    print(f"  Naive split on '. ':")
    for i, s in enumerate(naive_sentences):
        print(f"    {i + 1}. {s}")
    print()

    # NLTK approach
    nltk_sentences = sent_tokenize(paragraph)
    print(f"  NLTK sent_tokenize:")
    for i, s in enumerate(nltk_sentences):
        print(f"    {i + 1}. {s}")
    print()

    print("  NLTK correctly handles 'Dr.', '$1.5M', 'Jan.' as non-sentence-endings")
    print()


explore_tokenization()

# =============================================================================
# PART 3: NORMALIZATION - REDUCING VARIATION
# =============================================================================

"""
NORMALIZATION: Reduce superficial variation while preserving meaning.

Levels of normalization (from least to most aggressive):
    1. Case folding: "Hello" → "hello"
    2. Punctuation removal: "hello!" → "hello"
    3. Number handling: "100" → "<NUM>" or remove
    4. Stemming: "running" → "run"
    5. Lemmatization: "better" → "good"

Each level trades information for simplicity.
Choose based on your task!
"""


def explore_normalization():
    """
    Show normalization techniques and their trade-offs.
    """
    print("=" * 70)
    print("NORMALIZATION: Reducing Variation")
    print("=" * 70)
    print()

    # Case folding - when it helps and when it hurts
    print("CASE FOLDING")
    print("-" * 50)
    print()

    case_examples = [
        ("US vs us", "US (country) vs us (pronoun) - meaning lost!"),
        ("Apple vs apple", "Apple (company) vs apple (fruit) - meaning lost!"),
        ("WHO vs who", "WHO (organization) vs who (pronoun) - meaning lost!"),
        ("SCREAMING vs screaming", "Same word, different emphasis - safe to fold"),
    ]

    print("When case folding LOSES information:")
    for example, explanation in case_examples[:3]:
        parts = example.split(" vs ")
        print(f"  {parts[0]:10} → {parts[0].lower()}")
        print(f"  {parts[1]:10} → {parts[1].lower()}")
        print(f"  ⚠ {explanation}")
        print()

    print("When case folding is SAFE:")
    example, explanation = case_examples[3]
    parts = example.split(" vs ")
    print(f"  {parts[0]:10} → {parts[0].lower()}")
    print(f"  {parts[1]:10} → {parts[1].lower()}")
    print(f"  ✓ {explanation}")
    print()

    print("DECISION: Keep case for named entity recognition, fold for sentiment analysis")
    print()

    # Stemming vs Lemmatization - the core trade-off
    print("=" * 70)
    print("STEMMING vs LEMMATIZATION")
    print("=" * 70)
    print()

    print("STEMMING: Chop off suffixes using rules (fast, crude)")
    print("LEMMATIZATION: Look up dictionary form (slow, accurate)")
    print()

    test_words = [
        ("running", "verb"),
        ("runs", "verb"),
        ("ran", "verb"),
        ("better", "adjective"),
        ("caring", "verb"),
        ("university", "noun"),
        ("studies", "noun"),
        ("studying", "verb"),
    ]

    porter = PorterStemmer()
    lancaster = LancasterStemmer()  # More aggressive
    lemmatizer = WordNetLemmatizer()

    print(f"{'Word':<15} {'Porter':<12} {'Lancaster':<12} {'Lemma':<12} {'Notes'}")
    print("-" * 70)

    for word, pos in test_words:
        porter_stem = porter.stem(word)
        lancaster_stem = lancaster.stem(word)

        # Lemmatizer needs POS tag
        pos_map = {'verb': 'v', 'noun': 'n', 'adjective': 'a'}
        lemma = lemmatizer.lemmatize(word, pos=pos_map.get(pos, 'n'))

        # Identify problems
        notes = ""
        if porter_stem != lemma:
            if len(porter_stem) < 3:
                notes = "⚠ stem too short"
            elif porter_stem not in ['run', 'good', 'care', 'studi', 'univers']:
                notes = "⚠ stem ≠ root"

        print(f"{word:<15} {porter_stem:<12} {lancaster_stem:<12} {lemma:<12} {notes}")

    print()
    print("KEY INSIGHTS:")
    print("  • Porter: 'caring' → 'care' ✓ but 'university' → 'univers' ✗")
    print("  • Lancaster: more aggressive, 'university' → 'univers' ✗")
    print("  • Lemmatizer: 'better' → 'good' ✓ (knows irregular forms)")
    print()
    print("WHEN TO USE WHAT:")
    print("  • Stemming: Search engines, topic modeling (speed matters)")
    print("  • Lemmatization: Chatbots, sentiment analysis (accuracy matters)")
    print()


explore_normalization()

# =============================================================================
# PART 4: STOP WORDS - THE CONTROVERSIAL TOPIC
# =============================================================================

"""
STOP WORDS: Words that appear frequently but carry little meaning.

The conventional wisdom: Remove them.
The reality: It depends on your task!

    ✓ Remove for: Topic modeling, keyword extraction, search
    ✗ Keep for: Sentiment analysis ("not good" vs "good"), 
                Named entity recognition, Question answering

"To be or not to be" → "" (all stop words!)
"I am not happy" → "happy" (lost the negation!)
"""


def explore_stopwords():
    """
    Show when stop words matter and when they don't.
    """
    print("=" * 70)
    print("STOP WORDS: When to Remove, When to Keep")
    print("=" * 70)
    print()

    stop_words = set(stopwords.words('english'))

    print(f"English has {len(stop_words)} stop words in NLTK")
    print(f"Examples: {sorted(list(stop_words))[:20]}")
    print()

    # Dangerous examples
    print("WHEN REMOVING STOP WORDS DESTROYS MEANING:")
    print("-" * 50)

    dangerous_examples = [
        ("I am not happy with this product", "Sentiment analysis"),
        ("To be or not to be", "Literary analysis"),
        ("What is the capital of France?", "Question answering"),
        ("He is the CEO", "Named entity recognition"),
        ("This is not a good idea", "Opinion mining"),
    ]

    for text, task in dangerous_examples:
        tokens = word_tokenize(text.lower())
        filtered = [w for w in tokens if w not in stop_words]

        print(f"  Task: {task}")
        print(f"  Original: '{text}'")
        print(f"  After removal: '{' '.join(filtered)}'")

        # Highlight the problem
        lost_words = [w for w in tokens if w in stop_words]
        critical = {'not', 'no', 'is', 'are', 'was', 'were', 'what', 'who', 'the'}
        lost_critical = [w for w in lost_words if w in critical]

        if lost_critical:
            print(f"  ⚠ LOST CRITICAL WORDS: {lost_critical}")
        print()

    # When removal helps
    print("WHEN REMOVING STOP WORDS HELPS:")
    print("-" * 50)

    good_examples = [
        "The quick brown fox jumps over the lazy dog",
        "This is a document about machine learning and artificial intelligence",
    ]

    for text in good_examples:
        tokens = word_tokenize(text.lower())
        filtered = [w for w in tokens if w not in stop_words]

        print(f"  Original ({len(tokens)} tokens): '{text}'")
        print(f"  Filtered ({len(filtered)} tokens): '{' '.join(filtered)}'")
        print(f"  ✓ Core meaning preserved, noise removed")
        print()

    # Custom stop words
    print("DOMAIN-SPECIFIC STOP WORDS:")
    print("-" * 50)
    print()
    print("Medical domain might add: 'patient', 'doctor', 'hospital', 'treatment'")
    print("Legal domain might add: 'court', 'plaintiff', 'defendant', 'hereby'")
    print("Twitter analysis might add: 'rt', 'via', 'http', 'https'")
    print()
    print("LESSON: Stop word lists are task-dependent, not universal!")
    print()


explore_stopwords()


# =============================================================================
# PART 5: THE COMPLETE PIPELINE
# =============================================================================

@dataclass
class PreprocessingConfig:
    """
    Configuration for text preprocessing.

    Different tasks need different settings:

    SENTIMENT ANALYSIS:
        lowercase=True, remove_stopwords=False (keep 'not'!), 
        lemmatize=True, remove_numbers=True

    SEARCH ENGINE:
        lowercase=True, remove_stopwords=True,
        stem=True (faster than lemmatize), remove_numbers=False

    NAMED ENTITY RECOGNITION:
        lowercase=False (case matters!), remove_stopwords=False,
        lemmatize=False, remove_numbers=False
    """
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = True
    lemmatize: bool = True
    stem: bool = False  # Usually don't use both
    min_word_length: int = 2
    custom_stopwords: set = None


class TextPreprocessor:
    """
    Production-ready text preprocessing pipeline.

    Design principles:
    1. Configurable - different tasks need different settings
    2. Explainable - can show what each step does
    3. Efficient - processes text in one pass where possible
    """

    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()

        # Initialize tools lazily
        self._stemmer = None
        self._lemmatizer = None
        self._stop_words = None

    @property
    def stemmer(self):
        if self._stemmer is None:
            self._stemmer = PorterStemmer()
        return self._stemmer

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer

    @property
    def stop_words(self):
        if self._stop_words is None:
            self._stop_words = set(stopwords.words('english'))
            if self.config.custom_stopwords:
                self._stop_words.update(self.config.custom_stopwords)
        return self._stop_words

    def preprocess(self, text: str, explain: bool = False) -> List[str]:
        """
        Process text through the full pipeline.

        Args:
            text: Raw input text
            explain: If True, print each step

        Returns:
            List of processed tokens
        """
        steps = []

        if explain:
            steps.append(("Original", text))

        # Step 1: Lowercase
        if self.config.lowercase:
            text = text.lower()
            if explain:
                steps.append(("Lowercase", text))

        # Step 2: Remove URLs, emails, etc.
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)

        # Step 3: Remove numbers
        if self.config.remove_numbers:
            text = re.sub(r'\d+', ' ', text)
            if explain:
                steps.append(("Remove numbers", text))

        # Step 4: Remove punctuation
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
            if explain:
                steps.append(("Remove punctuation", text))

        # Step 5: Tokenize
        tokens = word_tokenize(text)
        if explain:
            steps.append(("Tokenize", tokens))

        # Step 6: Remove stop words
        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
            if explain:
                steps.append(("Remove stopwords", tokens))

        # Step 7: Stemming or Lemmatization
        if self.config.stem:
            tokens = [self.stemmer.stem(t) for t in tokens]
            if explain:
                steps.append(("Stem", tokens))
        elif self.config.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            if explain:
                steps.append(("Lemmatize", tokens))

        # Step 8: Remove short words
        if self.config.min_word_length > 1:
            tokens = [t for t in tokens if len(t) >= self.config.min_word_length]
            if explain:
                steps.append(("Min length filter", tokens))

        if explain:
            print("\nPreprocessing steps:")
            print("-" * 50)
            for step_name, result in steps:
                if isinstance(result, list):
                    print(f"  {step_name}: {result}")
                else:
                    print(f"  {step_name}: '{result[:80]}{'...' if len(result) > 80 else ''}'")
            print()

        return tokens


def demonstrate_pipeline():
    """
    Show the complete pipeline with different configurations.
    """
    print("=" * 70)
    print("COMPLETE PREPROCESSING PIPELINE")
    print("=" * 70)
    print()

    sample_text = """
    Dr. Johnson reviewed the patient's MRI results on 12/15/2023. 
    The findings are NOT concerning - no tumors detected! 
    Patient should continue current medication (Metformin 500mg).
    Follow-up appointment scheduled. Email: dr.johnson@hospital.com
    """

    print("Sample text:")
    print(f"  {sample_text.strip()}")
    print()

    # Configuration 1: For topic modeling
    print("Configuration 1: TOPIC MODELING")
    print("-" * 50)
    config1 = PreprocessingConfig(
        lowercase=True,
        remove_stopwords=True,
        lemmatize=True,
        remove_numbers=True
    )
    preprocessor1 = TextPreprocessor(config1)
    result1 = preprocessor1.preprocess(sample_text, explain=True)
    print(f"Final tokens: {result1}")
    print()

    # Configuration 2: For sentiment analysis
    print("Configuration 2: SENTIMENT ANALYSIS")
    print("-" * 50)
    config2 = PreprocessingConfig(
        lowercase=True,
        remove_stopwords=False,  # Keep 'not'!
        lemmatize=True,
        remove_numbers=True
    )
    preprocessor2 = TextPreprocessor(config2)
    result2 = preprocessor2.preprocess(sample_text, explain=True)
    print(f"Final tokens: {result2}")
    print()

    # Show the critical difference
    print("CRITICAL DIFFERENCE:")
    print("-" * 50)
    print(f"  Topic modeling: 'NOT' was {'removed' if 'not' not in result1 else 'kept'}")
    print(f"  Sentiment: 'NOT' was {'removed' if 'not' not in result2 else 'kept'}")
    print()
    print("  'NOT concerning' vs 'concerning' - opposite meanings!")
    print("  This is why preprocessing configuration matters.")
    print()


demonstrate_pipeline()


# =============================================================================
# PART 6: REAL-WORLD APPLICATION
# =============================================================================

def real_world_application():
    """
    Apply preprocessing to a realistic scenario.
    """
    print("=" * 70)
    print("REAL-WORLD APPLICATION: Analyzing Customer Reviews")
    print("=" * 70)
    print()

    reviews = [
        ("I absolutely LOVE this product! Best purchase ever!!!", 5),
        ("It's okay, nothing special. Does what it's supposed to do.", 3),
        ("Terrible quality. Broke after 2 days. DO NOT BUY!", 1),
        ("Not bad, but not great either. The price is reasonable.", 3),
        ("This is NOT what I expected - it's even BETTER! Amazing!", 5),
        ("Worst customer service I've ever experienced. Never again.", 1),
        ("Pretty good overall. Would recommend to friends.", 4),
        ("Absolutely horrible. Complete waste of money!!!", 1),
    ]

    print(f"Dataset: {len(reviews)} customer reviews with ratings (1-5)")
    print()

    # Preprocess for sentiment analysis
    config = PreprocessingConfig(
        lowercase=True,
        remove_stopwords=False,  # Keep negations!
        lemmatize=True,
        remove_numbers=True
    )
    preprocessor = TextPreprocessor(config)

    # Process all reviews
    processed_reviews = []
    for text, rating in reviews:
        tokens = preprocessor.preprocess(text)
        processed_reviews.append((text, tokens, rating))

    # Analyze word frequencies by sentiment
    positive_words = Counter()
    negative_words = Counter()

    for text, tokens, rating in processed_reviews:
        if rating >= 4:
            positive_words.update(tokens)
        elif rating <= 2:
            negative_words.update(tokens)

    print("RESULTS:")
    print("-" * 50)
    print()

    print("Sample processed reviews:")
    for text, tokens, rating in processed_reviews[:3]:
        sentiment = "Positive" if rating >= 4 else "Negative" if rating <= 2 else "Neutral"
        print(f"  [{sentiment}] Original: '{text[:50]}...'")
        print(f"           Tokens: {tokens}")
        print()

    print("Most common words in POSITIVE reviews:")
    for word, count in positive_words.most_common(10):
        print(f"  {word}: {count}")

    print()
    print("Most common words in NEGATIVE reviews:")
    for word, count in negative_words.most_common(10):
        print(f"  {word}: {count}")

    print()
    print("INSIGHT: Notice how 'not' appears in both positive and negative!")
    print("  'NOT what I expected - it's even BETTER' (positive)")
    print("  'DO NOT BUY' (negative)")
    print()
    print("This is why simple bag-of-words has limitations.")
    print("Next lesson: N-grams and context (TF-IDF, Word2Vec)")
    print()


real_world_application()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 70)
print("LESSON SUMMARY: Text Preprocessing")
print("=" * 70)
print()
print("WHAT YOU LEARNED:")
print("-" * 50)
print()
print("1. THE PROBLEM")
print("   • Computers see text as arbitrary symbols")
print("   • 'cat' and 'cats' are equally different from 'cat' and 'dog'")
print("   • Preprocessing maps surface variation to underlying meaning")
print()
print("2. TOKENIZATION")
print("   • Naive split breaks on abbreviations, contractions")
print("   • NLTK handles 'Dr.', 'don\\'t', '$19.99' correctly")
print("   • Choice of tokenizer affects everything downstream")
print()
print("3. NORMALIZATION")
print("   • Case folding: safe for sentiment, dangerous for NER")
print("   • Stemming: fast but crude ('caring' → 'car' is wrong)")
print("   • Lemmatization: accurate but slower ('better' → 'good')")
print()
print("4. STOP WORDS")
print("   • NOT universally useless!")
print("   • 'not good' → 'good' loses critical meaning")
print("   • Remove for topic modeling, keep for sentiment")
print()
print("5. THE PIPELINE")
print("   • Configuration depends on task")
print("   • Same text, different settings → different results")
print("   • No one-size-fits-all solution")
print()
print("NEXT: How do we turn these tokens into numbers?")
print("      TF-IDF, Word Embeddings, and the path to understanding.")
print()