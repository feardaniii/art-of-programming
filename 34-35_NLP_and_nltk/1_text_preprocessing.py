import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
from collections import Counter

"""
📝 TEXT PREPROCESSING: From Chaos to Clean Data

Human language is messy:
- "Running", "runs", "ran" - same meaning, different forms
- "the", "a", "is" - everywhere but meaningless
- "Dr. Smith lives on Wall St." - punctuation creates chaos
- "SCREAMING!!!" vs "screaming" - same word, different intensity

NLP preprocessing: standardize, clean, prepare for math.
"""

print("="*70)
print("📝 TEXT PREPROCESSING: The Foundation of NLP")
print("="*70)

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# ===== STEP 1: Raw Text - The Messy Reality =====
print("\n📄 STEP 1: Raw Text Example")
print("-" * 70)

raw_text = """
Dr. Johnson visited the clinic on 12/15/2023. Patient complained about
severe headaches & nausea!!! Prescribed medication - ibuprofen 200mg.
Follow-up appointment scheduled. Patient's condition improving significantly.
Email: patient@example.com. #HealthUpdate 💊
"""

print("Raw text:")
print(raw_text)
print(f"\nLength: {len(raw_text)} characters")
print(f"Contains: emails, dates, punctuation, emojis, hashtags, abbreviations")

# ===== STEP 2: Lowercasing =====
print("\n\n🔽 STEP 2: Lowercasing")
print("-" * 70)

lowercased = raw_text.lower()
print("After lowercasing:")
print(lowercased[:200])
print("\n💡 WHY: 'Doctor' and 'doctor' are the same word")
print("   Without lowercasing: treated as different tokens")

# ===== STEP 3: Remove Special Characters =====
print("\n\n🧹 STEP 3: Remove Special Characters & Numbers")
print("-" * 70)

def clean_text(text):
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

cleaned = clean_text(lowercased)
print("After cleaning:")
print(cleaned)
print("\n💡 REMOVED: emails, dates, numbers, punctuation, hashtags, emojis")

# ===== STEP 4: Tokenization =====
print("\n\n✂️  STEP 4: Tokenization - Split into Words")
print("-" * 70)

tokens = word_tokenize(cleaned)
print(f"Tokens (words): {tokens}")
print(f"\nTotal tokens: {len(tokens)}")
print("\n💡 TOKENIZATION: Break text into individual units (words)")
print("   'I love NLP' → ['I', 'love', 'NLP']")

# ===== STEP 5: Remove Stop Words =====
print("\n\n🚫 STEP 5: Remove Stop Words")
print("-" * 70)

stop_words = set(stopwords.words('english'))
print(f"English has {len(stop_words)} stop words")
print(f"Examples: {list(stop_words)[:20]}")

filtered_tokens = [word for word in tokens if word not in stop_words]
print(f"\nBefore stop word removal: {tokens}")
print(f"After stop word removal: {filtered_tokens}")
print(f"\nReduced from {len(tokens)} to {len(filtered_tokens)} tokens")
print("\n💡 STOP WORDS: Common words that carry little meaning")
print("   'the', 'a', 'is', 'are' - remove to focus on content")

# ===== STEP 6: Stemming vs Lemmatization =====
print("\n\n🌱 STEP 6: Stemming vs Lemmatization")
print("-" * 70)

test_words = ['running', 'runs', 'ran', 'runner', 'easily', 'fairly', 'caring', 'better', 'worse']

# Stemming: crude chopping
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in test_words]

# Lemmatization: intelligent reduction
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in test_words]

print("Original  →  Stemmed  →  Lemmatized")
print("-" * 50)
for orig, stem, lemma in zip(test_words, stemmed, lemmatized):
    print(f"{orig:12s} → {stem:10s} → {lemma}")

print("\n💡 STEMMING: Crude chopping (run, run, ran, runner → run, run, ran, runner)")
print("   Fast but imprecise: 'caring' → 'car' (wrong!)")
print("\n💡 LEMMATIZATION: Intelligent reduction using dictionary")
print("   'running', 'runs', 'ran' → 'run' (all same root)")
print("   Slower but accurate")

# Apply lemmatization to our text
lemmatized_tokens = [lemmatizer.lemmatize(word, pos='v') for word in filtered_tokens]
print(f"\nFinal processed tokens: {lemmatized_tokens}")

# ===== STEP 7: Complete Pipeline =====
print("\n\n🔄 STEP 7: Complete Preprocessing Pipeline")
print("-" * 70)

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """
    Production-grade text preprocessing pipeline
    Used in: Search engines, chatbots, sentiment analysis
    """
    # Lowercase
    text = text.lower()

    # Remove special characters
    text = clean_text(text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if w not in stop_words]

    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w, pos='v') for w in tokens]

    return tokens

# Test on real examples
examples = [
    "I'm absolutely LOVING this amazing product!!! Best purchase ever! 😍",
    "The patient complained about severe headaches and nausea symptoms.",
    "Running, runs, ran - they're all variations of the same action."
]

print("\nProcessing real examples:")
print("=" * 70)
for i, example in enumerate(examples, 1):
    processed = preprocess_text(example)
    print(f"\n{i}. Original: {example}")
    print(f"   Processed: {processed}")

# ===== STEP 8: Real Application - Health Notes Analysis =====
print("\n\n🏥 STEP 8: Real Application - Medical Notes Processing")
print("-" * 70)

medical_notes = [
    "Patient experiencing severe migraine headaches with nausea and sensitivity to light.",
    "Diagnosed with influenza. Prescribed antiviral medication and rest.",
    "Follow-up visit shows significant improvement in symptoms. Continue current treatment.",
    "Patient reports persistent cough and fever. Recommend chest X-ray.",
    "Allergic reaction to penicillin noted. Update medication list immediately."
]

print("Processing 5 medical notes...")
processed_notes = [preprocess_text(note) for note in medical_notes]

# Extract most common medical terms
all_terms = [term for note in processed_notes for term in note]
term_freq = Counter(all_terms)

print("\nMost common medical terms:")
for term, count in term_freq.most_common(10):
    print(f"   {term:20s}: {count} occurrences")

print("\n💡 REAL-WORLD USE:")
print("   • Extract symptoms from patient notes")
print("   • Identify medication mentions")
print("   • Track treatment effectiveness keywords")
print("   • Build medical terminology frequency database")

print("\n" + "="*70)
print("🎓 TEXT PREPROCESSING COMPLETE!")
print("="*70)
print("\nWhat you mastered:")
print("   ✓ Lowercasing & cleaning")
print("   ✓ Tokenization (text → words)")
print("   ✓ Stop word removal")
print("   ✓ Lemmatization (words → roots)")
print("   ✓ Complete preprocessing pipeline")
print("\n🚀 This is the foundation for EVERY NLP task!")
print("   Next: Turn these clean tokens into numbers (TF-IDF & Word2Vec)")
