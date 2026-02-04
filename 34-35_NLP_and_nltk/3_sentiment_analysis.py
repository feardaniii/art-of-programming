"""
================================================================================
NLP PART 3: SENTIMENT ANALYSIS
================================================================================

Course: The Art of Programming - NLP Module
Lesson: Teaching Machines to Read Emotions

THE GOAL:
---------
Given text, predict whether the sentiment is positive, negative, or neutral.

    "This product is amazing!" → POSITIVE
    "Terrible experience, waste of money" → NEGATIVE
    "It's okay, nothing special" → NEUTRAL

WHY THIS MATTERS:
-----------------
    Amazon:     Millions of reviews → which products are loved/hated?
    Twitter:    Real-time brand monitoring → is sentiment shifting?
    Support:    Thousands of tickets → which customers are angry?
    Finance:    News sentiment → market prediction signals
    Healthcare: Patient feedback → quality of care indicators

THE APPROACH:
-------------
    1. Preprocess text (Part 1)
    2. Convert to vectors (Part 2 - TF-IDF)
    3. Train a classifier (This lesson)
    4. Predict sentiment on new text

We'll explore WHY simple models work surprisingly well for sentiment,
and WHERE they fail (setting up the need for deep learning).

================================================================================
"""

import numpy as np
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math

# =============================================================================
# PART 1: THE DATASET
# =============================================================================

"""
A realistic sentiment analysis dataset.

In production, you'd use:
    - IMDB reviews (50,000 movie reviews)
    - Amazon reviews (millions of product reviews)
    - Twitter sentiment datasets
    - Yelp reviews

For learning, we'll create a structured dataset that demonstrates:
    - Clear positive/negative signals
    - Subtle cases
    - Edge cases where models fail
"""


def create_sentiment_dataset():
    """
    Create a dataset that demonstrates sentiment analysis challenges.

    Returns list of (text, label) tuples.
    """

    # POSITIVE reviews - clear signals
    positive_clear = [
        ("This product is absolutely amazing! Best purchase I've ever made.", "positive"),
        ("Excellent quality and fast shipping. Highly recommend!", "positive"),
        ("Love it! Exceeded all my expectations. Five stars!", "positive"),
        ("Outstanding customer service and great product. Will buy again.", "positive"),
        ("Perfect! Works exactly as described. Very happy with this.", "positive"),
        ("Fantastic experience from start to finish. Couldn't be happier.", "positive"),
        ("Brilliant! This solved all my problems. Worth every penny.", "positive"),
        ("Superb quality. Much better than expected. Absolutely love it.", "positive"),
        ("Incredible value for money. Best in its category.", "positive"),
        ("Wonderful product! My whole family loves it.", "positive"),
    ]

    # NEGATIVE reviews - clear signals
    negative_clear = [
        ("Terrible product. Complete waste of money. Avoid!", "negative"),
        ("Horrible experience. Worst purchase I've ever made.", "negative"),
        ("Awful quality. Broke within a week. Very disappointed.", "negative"),
        ("Disgusting customer service. Never buying from them again.", "negative"),
        ("Useless. Doesn't work at all. Total scam.", "negative"),
        ("Pathetic excuse for a product. Completely worthless.", "negative"),
        ("Dreadful. Arrived broken and customer service was unhelpful.", "negative"),
        ("Abysmal quality. Falling apart already. Save your money.", "negative"),
        ("Horrible. Nothing like the description. Total disappointment.", "negative"),
        ("Garbage. Threw it away immediately. Waste of time and money.", "negative"),
    ]

    # NEUTRAL reviews - mixed or bland signals
    neutral = [
        ("It's okay. Does what it's supposed to do. Nothing special.", "neutral"),
        ("Average product. Works as expected. Not great, not terrible.", "neutral"),
        ("Decent for the price. Some good features, some bad.", "neutral"),
        ("Middle of the road. Gets the job done but has issues.", "neutral"),
        ("Fair quality. Acceptable but room for improvement.", "neutral"),
        ("Standard product. Nothing to complain about, nothing to praise.", "neutral"),
        ("Adequate. Meets basic expectations but doesn't exceed them.", "neutral"),
        ("So-so. Has pros and cons. Depends on your needs.", "neutral"),
        ("Mediocre. Could be better, could be worse.", "neutral"),
        ("Reasonable purchase. Not amazing but not bad either.", "neutral"),
    ]

    # TRICKY cases - where simple models often fail
    tricky_positive = [
        # Negation of negative
        ("Not bad at all! Actually quite good.", "positive"),
        ("I don't hate it. In fact, I love it!", "positive"),
        ("This isn't terrible like the reviews said. It's great!", "positive"),
        # Sarcasm that reads negative
        ("Oh great, another amazing product that actually works!", "positive"),
    ]

    tricky_negative = [
        # Negation of positive
        ("This is not good. Not good at all.", "negative"),
        ("I don't love it. Actually, I hate it.", "negative"),
        ("Not what I expected. Very disappointed.", "negative"),
        # Positive words in negative context
        ("I wish this was good. Unfortunately, it's terrible.", "negative"),
        ("Would be great if it actually worked.", "negative"),
    ]

    tricky_neutral_hard = [
        # Mixed strong signals
        ("Love the design but hate the quality.", "neutral"),
        ("Great features but terrible customer service.", "neutral"),
        ("Amazing price but awful durability.", "neutral"),
    ]

    # Combine all
    all_data = (
            positive_clear + negative_clear + neutral +
            tricky_positive + tricky_negative + tricky_neutral_hard
    )

    return all_data


# =============================================================================
# PART 2: UNDERSTANDING NAIVE BAYES
# =============================================================================

"""
NAIVE BAYES: The Surprisingly Effective "Naive" Classifier

WHY "NAIVE"?
------------
It assumes words are INDEPENDENT given the class.

    P(review | positive) = P(word1 | positive) × P(word2 | positive) × ...

This is clearly FALSE! "New York" words are not independent.
But it works anyway. Why?

WHY IT WORKS:
-------------
1. For classification, we only need to COMPARE probabilities
   P(positive | review) vs P(negative | review)

2. Even if individual probabilities are wrong,
   their relative ordering is often correct

3. Text has MANY features (words)
   Errors tend to cancel out across many words

THE MATH:
---------
Using Bayes' theorem:

    P(class | document) = P(document | class) × P(class) / P(document)

We want the class with highest probability:

    argmax P(class | document) = argmax P(document | class) × P(class)

    = argmax P(word1 | class) × P(word2 | class) × ... × P(class)

Using log to avoid underflow:

    = argmax log(P(class)) + Σ log(P(wordi | class))
"""


class NaiveBayesClassifier:
    """
    Naive Bayes classifier implemented from scratch.

    This shows exactly what sklearn.MultinomialNB does under the hood.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Laplace smoothing parameter (prevents zero probabilities)
        """
        self.alpha = alpha
        self.class_log_priors = {}  # log P(class)
        self.word_log_probs = {}  # log P(word | class)
        self.vocabulary = set()
        self.classes = []

    def fit(self, documents: List[List[str]], labels: List[str]):
        """
        Train the classifier.

        For each class, learn:
            1. P(class) - how common is this class?
            2. P(word | class) - how common is each word in this class?
        """
        self.classes = list(set(labels))
        n_docs = len(documents)

        # Build vocabulary
        for doc in documents:
            self.vocabulary.update(doc)
        vocab_size = len(self.vocabulary)

        # Count documents and words per class
        class_doc_counts = Counter(labels)
        class_word_counts = {c: Counter() for c in self.classes}
        class_total_words = {c: 0 for c in self.classes}

        for doc, label in zip(documents, labels):
            class_word_counts[label].update(doc)
            class_total_words[label] += len(doc)

        # Calculate log probabilities
        for c in self.classes:
            # P(class) = number of docs in class / total docs
            self.class_log_priors[c] = math.log(class_doc_counts[c] / n_docs)

            # P(word | class) with Laplace smoothing
            # P(word | class) = (count(word, class) + alpha) / (total_words_in_class + alpha * vocab_size)
            self.word_log_probs[c] = {}
            denominator = class_total_words[c] + self.alpha * vocab_size

            for word in self.vocabulary:
                count = class_word_counts[c][word]
                prob = (count + self.alpha) / denominator
                self.word_log_probs[c][word] = math.log(prob)

        return self

    def predict_proba(self, document: List[str]) -> Dict[str, float]:
        """
        Calculate log probability for each class.

        Returns dict of {class: log_probability}
        """
        log_probs = {}

        for c in self.classes:
            # Start with log P(class)
            log_prob = self.class_log_priors[c]

            # Add log P(word | class) for each word
            for word in document:
                if word in self.vocabulary:
                    log_prob += self.word_log_probs[c][word]
                # Unknown words are ignored (could also use UNK token)

            log_probs[c] = log_prob

        return log_probs

    def predict(self, document: List[str]) -> str:
        """Predict the class with highest probability."""
        log_probs = self.predict_proba(document)
        return max(log_probs, key=log_probs.get)

    def get_feature_importance(self, class_name: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get most indicative words for a class.

        Returns words with highest P(word | class) / P(word | other_classes)
        """
        if class_name not in self.classes:
            raise ValueError(f"Unknown class: {class_name}")

        # Calculate log-odds ratio for each word
        word_scores = []
        other_classes = [c for c in self.classes if c != class_name]

        for word in self.vocabulary:
            log_prob_class = self.word_log_probs[class_name][word]

            # Average log prob in other classes
            log_prob_others = sum(
                self.word_log_probs[c][word] for c in other_classes
            ) / len(other_classes)

            # Log odds ratio
            score = log_prob_class - log_prob_others
            word_scores.append((word, score))

        return sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_n]


def demonstrate_naive_bayes():
    """
    Show how Naive Bayes works step by step.
    """
    print("=" * 70)
    print("NAIVE BAYES: The Surprisingly Effective Classifier")
    print("=" * 70)
    print()

    # Simple example to show the math
    print("A SIMPLE EXAMPLE:")
    print("-" * 50)
    print()

    # Tiny corpus
    training_data = [
        (["love", "this", "great", "product"], "positive"),
        (["amazing", "excellent", "love", "it"], "positive"),
        (["hate", "this", "terrible", "awful"], "negative"),
        (["horrible", "waste", "hate", "it"], "negative"),
    ]

    documents = [doc for doc, label in training_data]
    labels = [label for doc, label in training_data]

    print("Training data:")
    for doc, label in training_data:
        print(f"  {label:>8}: {' '.join(doc)}")
    print()

    # Train classifier
    nb = NaiveBayesClassifier(alpha=1.0)
    nb.fit(documents, labels)

    # Show learned probabilities
    print("LEARNED PROBABILITIES:")
    print("-" * 50)
    print()

    print("P(class) - Prior probabilities:")
    for c in nb.classes:
        prob = math.exp(nb.class_log_priors[c])
        print(f"  P({c}) = {prob:.3f}")
    print()

    print("P(word | class) - Word probabilities (selected words):")
    print()

    key_words = ["love", "hate", "great", "terrible", "this"]
    print(f"{'Word':<12} {'P(w|pos)':<12} {'P(w|neg)':<12}")
    print("-" * 36)

    for word in key_words:
        if word in nb.vocabulary:
            p_pos = math.exp(nb.word_log_probs["positive"][word])
            p_neg = math.exp(nb.word_log_probs["negative"][word])
            print(f"{word:<12} {p_pos:<12.4f} {p_neg:<12.4f}")

    print()
    print("INSIGHT:")
    print("  'love' is much more likely in positive reviews")
    print("  'hate' is much more likely in negative reviews")
    print("  'this' is equally likely (not discriminative)")
    print()

    # Classify new document
    print("=" * 70)
    print("CLASSIFYING A NEW REVIEW")
    print("=" * 70)
    print()

    new_review = ["love", "this", "amazing"]
    print(f"Review: '{' '.join(new_review)}'")
    print()

    print("Calculation:")
    print("-" * 50)

    for c in nb.classes:
        print(f"\nFor class '{c}':")

        log_prob = nb.class_log_priors[c]
        print(f"  log P({c}) = {log_prob:.3f}")

        for word in new_review:
            if word in nb.vocabulary:
                word_log_prob = nb.word_log_probs[c][word]
                log_prob += word_log_prob
                print(f"  + log P('{word}' | {c}) = {word_log_prob:.3f}")

        print(f"  = Total log probability: {log_prob:.3f}")

    print()
    prediction = nb.predict(new_review)
    print(f"PREDICTION: {prediction.upper()}")
    print()
    print("The positive class wins because 'love' and 'amazing'")
    print("are much more likely in positive reviews!")
    print()


demonstrate_naive_bayes()


# =============================================================================
# PART 3: THE FULL PIPELINE
# =============================================================================

class SentimentAnalyzer:
    """
    Complete sentiment analysis pipeline.

    Combines:
        1. Preprocessing (from Part 1)
        2. TF-IDF vectorization (from Part 2)
        3. Naive Bayes classification (this lesson)
    """

    def __init__(self):
        self.classifier = None
        self.vocabulary = set()
        self.idf = {}

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text for sentiment analysis.

        Key decisions for sentiment:
            - Keep negations ('not', 'never', "n't")
            - Keep intensifiers ('very', 'extremely')
            - Remove other stop words
        """
        # Lowercase
        text = text.lower()

        # Handle contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'ll", " will", text)

        # Remove punctuation except ! and ?
        text = re.sub(r'[^\w\s!?]', ' ', text)

        # Tokenize
        tokens = text.split()

        # Remove stop words BUT keep sentiment-relevant ones
        keep_words = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
            'very', 'extremely', 'really', 'absolutely', 'totally',
            'but', 'however', 'although', 'though',
        }

        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for', 'on',
            'with', 'at', 'by', 'from', 'as', 'is', 'was', 'are', 'were',
            'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'it', 'its',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they',
        }

        tokens = [t for t in tokens if t not in stop_words or t in keep_words]

        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 1]

        return tokens

    def fit(self, texts: List[str], labels: List[str]):
        """Train the sentiment analyzer."""
        # Preprocess all texts
        processed = [self.preprocess(text) for text in texts]

        # Train classifier
        self.classifier = NaiveBayesClassifier(alpha=1.0)
        self.classifier.fit(processed, labels)

        return self

    def predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Predict sentiment of a single text.

        Returns:
            (predicted_class, probability_distribution)
        """
        tokens = self.preprocess(text)
        prediction = self.classifier.predict(tokens)
        log_probs = self.classifier.predict_proba(tokens)

        # Convert log probs to normalized probabilities
        max_log_prob = max(log_probs.values())
        exp_probs = {c: math.exp(lp - max_log_prob) for c, lp in log_probs.items()}
        total = sum(exp_probs.values())
        probs = {c: p / total for c, p in exp_probs.items()}

        return prediction, probs

    def get_important_words(self, class_name: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get most indicative words for a sentiment class."""
        return self.classifier.get_feature_importance(class_name, top_n)


def run_full_pipeline():
    """
    Run the complete sentiment analysis pipeline.
    """
    print("=" * 70)
    print("COMPLETE SENTIMENT ANALYSIS PIPELINE")
    print("=" * 70)
    print()

    # Create dataset
    dataset = create_sentiment_dataset()
    texts = [text for text, label in dataset]
    labels = [label for text, label in dataset]

    print(f"Dataset: {len(dataset)} reviews")
    print(f"Classes: {Counter(labels)}")
    print()

    # Split into train/test
    # Simple split: use first 80% for training
    split_idx = int(len(dataset) * 0.8)

    train_texts, train_labels = texts[:split_idx], labels[:split_idx]
    test_texts, test_labels = texts[split_idx:], labels[split_idx:]

    print(f"Training set: {len(train_texts)} reviews")
    print(f"Test set: {len(test_texts)} reviews")
    print()

    # Train model
    print("Training sentiment analyzer...")
    analyzer = SentimentAnalyzer()
    analyzer.fit(train_texts, train_labels)
    print("Done!")
    print()

    # Show important words
    print("=" * 70)
    print("WORDS THAT INDICATE EACH SENTIMENT")
    print("=" * 70)
    print()

    for sentiment in ["positive", "negative", "neutral"]:
        words = analyzer.get_important_words(sentiment, top_n=8)
        print(f"{sentiment.upper()} indicators:")
        for word, score in words:
            bar = "█" * int(abs(score) * 10)
            print(f"  {word:<15} {score:>6.2f}  {bar}")
        print()

    # Evaluate on test set
    print("=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)
    print()

    correct = 0
    results = []

    for text, true_label in zip(test_texts, test_labels):
        pred_label, probs = analyzer.predict(text)
        is_correct = pred_label == true_label
        correct += is_correct
        results.append((text, true_label, pred_label, probs, is_correct))

    accuracy = correct / len(test_texts)
    print(f"Accuracy: {accuracy:.1%} ({correct}/{len(test_texts)})")
    print()

    # Show some predictions
    print("Sample predictions:")
    print("-" * 70)

    for text, true_label, pred_label, probs, is_correct in results[:6]:
        status = "✓" if is_correct else "✗"
        print(f"\n{status} '{text[:60]}...'")
        print(f"   True: {true_label:<10} Predicted: {pred_label:<10}")
        print(f"   Probabilities: ", end="")
        for c, p in sorted(probs.items()):
            print(f"{c}={p:.2f} ", end="")
        print()

    # Show errors
    print()
    print("=" * 70)
    print("ERROR ANALYSIS - Where the model fails")
    print("=" * 70)
    print()

    errors = [(t, true, pred, probs) for t, true, pred, probs, correct in results if not correct]

    if errors:
        print(f"Total errors: {len(errors)}")
        print()
        for text, true_label, pred_label, probs in errors:
            print(f"Text: '{text}'")
            print(f"True: {true_label}, Predicted: {pred_label}")
            print(f"Why it failed: ", end="")

            # Analyze why
            tokens = analyzer.preprocess(text)
            if "not" in text.lower():
                print("Contains negation - hard to handle with word independence")
            elif any(w in text.lower() for w in ["but", "however", "although"]):
                print("Contains contrast - mixed signals")
            else:
                print(f"Unusual word patterns: {tokens}")
            print()
    else:
        print("No errors! (Likely overfitting on small dataset)")

    return analyzer


analyzer = run_full_pipeline()


# =============================================================================
# PART 4: WHERE SIMPLE MODELS FAIL
# =============================================================================

def demonstrate_failures():
    """
    Show cases where bag-of-words sentiment analysis fails.

    These failures motivate the need for deep learning.
    """
    print()
    print("=" * 70)
    print("WHERE SIMPLE MODELS FAIL")
    print("=" * 70)
    print()

    # Test cases that are hard for bag-of-words
    hard_cases = [
        # Negation
        ("I don't love this product at all", "negative",
         "NEGATION: 'love' is positive, but 'don't love' is negative"),

        ("This is not bad", "positive",
         "DOUBLE NEGATION: 'not' + 'bad' = positive"),

        # Sarcasm
        ("Oh great, another product that doesn't work", "negative",
         "SARCASM: 'great' is positive word, but sentiment is negative"),

        ("Yeah right, like this is the best thing ever", "negative",
         "SARCASM: All positive words, negative meaning"),

        # Context-dependent
        ("The battery lasts a long time... in your dreams", "negative",
         "CONTEXT: Positive statement negated by 'in your dreams'"),

        # Mixed sentiment
        ("Great features but terrible build quality", "neutral",
         "MIXED: Positive + negative = hard to average"),

        # Comparative
        ("Better than nothing, I guess", "negative",
         "FAINT PRAISE: Technically positive comparison, negative sentiment"),

        # Domain-specific
        ("This phone is sick!", "positive",
         "SLANG: 'sick' usually negative, here means 'cool'"),
    ]

    print("Testing hard cases:")
    print("-" * 70)

    for text, true_sentiment, explanation in hard_cases:
        pred, probs = analyzer.predict(text)
        is_correct = pred == true_sentiment
        status = "✓" if is_correct else "✗"

        print(f"\n{status} '{text}'")
        print(f"   Expected: {true_sentiment}, Got: {pred}")
        print(f"   {explanation}")

    print()
    print("=" * 70)
    print("WHY THESE ARE HARD")
    print("=" * 70)
    print()
    print("1. NEGATION")
    print("   Bag-of-words treats 'not' and 'good' as separate features")
    print("   It can't understand 'not good' is opposite of 'good'")
    print()
    print("2. SARCASM")
    print("   Requires understanding speaker's intent")
    print("   Words are literally positive, meaning is negative")
    print()
    print("3. CONTEXT")
    print("   Same word different meaning in different contexts")
    print("   'sick' can be negative (ill) or positive (cool)")
    print()
    print("4. MIXED SENTIMENT")
    print("   Multiple opinions in one review")
    print("   Averaging them loses nuance")
    print()
    print("SOLUTION: Deep learning models that understand:")
    print("   - Word ORDER (not just presence)")
    print("   - Long-range DEPENDENCIES")
    print("   - CONTEXT-dependent word meanings")
    print()
    print("This is why we need: RNNs, Attention, Transformers, BERT, GPT...")
    print()


demonstrate_failures()


# =============================================================================
# PART 5: IMPROVING WITH N-GRAMS
# =============================================================================

def demonstrate_ngrams():
    """
    Show how n-grams partially solve the negation problem.
    """
    print("=" * 70)
    print("N-GRAMS: A Partial Solution")
    print("=" * 70)
    print()

    print("THE PROBLEM:")
    print("  'not good' should be different from 'good'")
    print("  But bag-of-words sees: ['not', 'good'] - same as ['good', 'not']")
    print()

    print("THE SOLUTION: N-GRAMS")
    print("  Instead of single words, use word sequences")
    print()

    text = "this product is not good"
    words = text.split()

    # Unigrams (single words)
    unigrams = words
    print(f"Text: '{text}'")
    print()
    print(f"Unigrams (n=1): {unigrams}")

    # Bigrams (word pairs)
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    print(f"Bigrams (n=2):  {bigrams}")

    # Trigrams (word triplets)
    trigrams = [f"{words[i]}_{words[i + 1]}_{words[i + 2]}" for i in range(len(words) - 2)]
    print(f"Trigrams (n=3): {trigrams}")

    print()
    print("NOW 'not_good' IS A SINGLE FEATURE!")
    print()
    print("  'not_good' → negative indicator")
    print("  'good' alone → positive indicator (but weaker with bigrams)")
    print()

    # Show the improvement
    print("COMPARISON:")
    print("-" * 50)
    print()

    positive_review = "this is very good"
    negative_review = "this is not good"

    print(f"Positive: '{positive_review}'")
    print(f"  Unigrams: {positive_review.split()}")
    print(
        f"  Bigrams: {[f'{positive_review.split()[i]}_{positive_review.split()[i + 1]}' for i in range(len(positive_review.split()) - 1)]}")
    print()

    print(f"Negative: '{negative_review}'")
    print(f"  Unigrams: {negative_review.split()}")
    print(
        f"  Bigrams: {[f'{negative_review.split()[i]}_{negative_review.split()[i + 1]}' for i in range(len(negative_review.split()) - 1)]}")
    print()

    print("With unigrams: Both have 'good' → hard to distinguish")
    print("With bigrams: 'very_good' vs 'not_good' → easy to distinguish!")
    print()
    print("LIMITATION: N-grams explode vocabulary size")
    print("  Unigrams: ~10,000 words")
    print("  Bigrams: ~1,000,000 pairs")
    print("  Trigrams: ~100,000,000 triplets")
    print()
    print("This is why deep learning (which learns patterns automatically)")
    print("eventually replaced n-grams for complex tasks.")
    print()


demonstrate_ngrams()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 70)
print("LESSON SUMMARY: Sentiment Analysis")
print("=" * 70)
print()
print("WHAT YOU LEARNED:")
print("-" * 50)
print()
print("1. NAIVE BAYES CLASSIFICATION")
print("   • 'Naive' = assumes word independence (false but useful)")
print("   • P(class | doc) ∝ P(class) × Π P(word | class)")
print("   • Simple, fast, works surprisingly well")
print()
print("2. THE COMPLETE PIPELINE")
print("   • Preprocess → TF-IDF → Classifier → Prediction")
print("   • Keep negations! ('not', 'never' matter for sentiment)")
print("   • Feature importance shows what words indicate what")
print()
print("3. WHERE SIMPLE MODELS FAIL")
print("   • Negation: 'not good' treated same as 'good'")
print("   • Sarcasm: words say positive, meaning is negative")
print("   • Context: 'sick' = ill vs 'sick' = cool")
print("   • Mixed: 'great features but terrible quality'")
print()
print("4. N-GRAMS AS PARTIAL SOLUTION")
print("   • 'not_good' as single feature")
print("   • Captures local word order")
print("   • But vocabulary explodes")
print()
print("THE PATH FORWARD:")
print("-" * 50)
print()
print("Simple models (Naive Bayes, Logistic Regression + TF-IDF):")
print("  ✓ Fast, interpretable, good baseline")
print("  ✗ Can't handle complex language phenomena")
print()
print("Deep learning (RNNs, Transformers, BERT, GPT):")
print("  ✓ Understand context, word order, long-range dependencies")
print("  ✗ Slower, need more data, less interpretable")
print()
print("In practice: Start simple, go deep when needed.")
print()