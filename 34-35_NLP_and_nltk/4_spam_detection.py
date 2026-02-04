# ========================================
# PART 1: Import Libraries
# ========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import os

# For text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# For machine learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# Download NLTK data (run once)
# nltk.download('stopwords')
# nltk.download('punkt')

# Set up visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# ========================================
# PART 1.5: Check/Generate Dataset
# ========================================
def generate_spam_dataset(filepath='spam.csv', n_samples=5000):
    """
    Generate a realistic spam dataset where spam/ham are distinguished
    by CONTENT patterns, not length.
    """
    print("📝 Generating realistic spam dataset...")
    np.random.seed(42)

    # ========================================
    # HAM MESSAGES - varied lengths (short to long)
    # ========================================
    ham_messages = [
        # Short (5-20 chars)
        "Ok", "Thanks!", "Sure", "On my way", "Got it", "Yes", "No problem",
        "Call me", "K", "Sounds good", "Will do", "Perfect", "See ya", "Later",
        "Miss you", "Love you", "Good luck", "Take care", "Be there soon",

        # Medium (20-60 chars)
        "Hey, are you coming to the party tonight?",
        "Can you pick up some milk on your way home?",
        "Meeting rescheduled to 3pm tomorrow",
        "Running late, be there in 10 minutes",
        "Did you see the game last night?",
        "What time should I pick you up?",
        "Sorry I missed your call, was in a meeting",
        "How was your weekend?",
        "The project deadline is next Friday",
        "Let me know when you're free to chat",
        "Can you send me the address again?",
        "Traffic is terrible, might be late",
        "Don't forget we have dinner with the Smiths tonight",
        "Your appointment is confirmed for Monday at 10am",
        "The restaurant was excellent, good choice",
        "Just checking in, how are you feeling today?",
        "Mom called, she wants you to call her back",
        "Remember to bring the documents tomorrow morning",
        "Got your message, will reply properly later tonight",
        "Can we reschedule our meeting to Thursday instead?",

        # Long (60-150 chars)
        "Hey! Just wanted to let you know that I finished the report you asked for. It's on your desk whenever you're ready to review it.",
        "The kids had a great time at the birthday party yesterday. Thanks so much for organizing everything, it was perfect!",
        "I've been thinking about what you said and I think you're right. Let's talk more about it when we meet on Saturday.",
        "Quick update: the contractor called and said they can start the kitchen renovation next Monday. Does that work for you?",
        "Happy anniversary! Can you believe it's been 10 years already? Dinner reservation is at 7pm at our favorite place.",
        "Just got out of the doctor's appointment. Everything looks good, nothing to worry about. Will tell you more at home.",
        "The flight is delayed by 2 hours so I won't land until 9pm. Don't wait up for me, I'll grab a taxi from the airport.",
        "Reminder: parent-teacher conference is tomorrow at 4pm. I can go if you're still stuck at work, just let me know.",
        "Found that book you were looking for at the used bookstore downtown. Picked it up for you, only cost me five dollars!",
        "Weather forecast says rain all weekend. Maybe we should move the barbecue to next week? Let me know what you think.",
    ]

    # ========================================
    # SPAM MESSAGES - also varied lengths!
    # ========================================
    spam_messages = [
        # Short spam (10-30 chars) - these exist!
        "FREE CASH NOW!", "Call 0800-WIN-BIG", "U WON! Claim now",
        "URGENT: Call back", "Free entry! Text WIN", "Act NOW!!!",
        "You're a WINNER!", "Claim ur prize", "FREE iPhone!",
        "Hot singles nearby", "Make money fast!", "Click here NOW",
        "Limited offer!", "Don't miss out!", "Reply to WIN",

        # Medium spam (30-80 chars)
        "CONGRATULATIONS! You've won a FREE gift card! Claim now!",
        "Your account has been compromised. Verify immediately.",
        "Make $5000 weekly working from home! No experience needed!",
        "You have been selected for a FREE vacation package!",
        "ALERT: Unusual activity detected on your account.",
        "Lose 30 pounds in 30 days! Doctors hate this trick!",
        "Your loan application has been APPROVED! Get cash today!",
        "WINNER! Reply with your details to claim your prize!",
        "Free trial! Cancel anytime! No credit card required!",
        "Someone viewed your profile! See who at this link:",
        "Cheap medications delivered to your door! Order now!",
        "Double your investment GUARANTEED! Limited spots available!",
        "Your package could not be delivered. Update address here:",
        "Congratulations! Your email won our daily lottery draw!",
        "FINAL NOTICE: Your subscription expires in 24 hours!",
        "Get rich quick! This simple trick will change your life!",

        # Long spam (80-150 chars)
        "Dear valued customer, we regret to inform you that your account will be suspended unless you verify your information within 24 hours.",
        "You have inherited $4,500,000 from a distant relative in Nigeria. Please reply with your bank details to process the transfer.",
        "WARNING: Your computer may be infected with dangerous viruses! Download our FREE antivirus software now to protect your data!",
        "Exclusive offer just for you! Buy one get one FREE on all luxury watches! This offer expires at midnight so act fast!",
        "Hello dear friend, I am a prince seeking help to transfer funds. You will receive 30% commission for your assistance.",
        "Your recent purchase requires verification. Click the secure link below to confirm your identity and avoid account suspension.",
        "Amazing news! You've been pre-approved for a platinum credit card with 0% APR! No credit check required! Apply now!",
        "This is not a drill! Your neighbor is making $347 per day from home using this one weird trick that banks don't want you to know!",
        "Security alert: We detected a login from an unrecognized device. If this wasn't you, secure your account immediately.",
        "Limited time offer: Get a FREE iPhone 15 Pro Max! Just complete our short survey and pay shipping. Only 50 left!",
    ]

    # Generate balanced lengths for both classes
    messages = []
    labels = []

    n_ham = int(n_samples * 0.87)
    n_spam = n_samples - n_ham

    # Generate ham with natural variation
    for _ in range(n_ham):
        msg = np.random.choice(ham_messages)
        # Sometimes add natural extensions
        if np.random.random() < 0.2 and len(msg) < 50:
            extensions = [" Let me know!", " Thanks!", " Talk soon.", " See you!", ""]
            msg = msg.rstrip('!?.') + np.random.choice(extensions)
        messages.append(msg)
        labels.append('ham')

    # Generate spam with natural variation
    for _ in range(n_spam):
        msg = np.random.choice(spam_messages)
        # Sometimes modify spam
        if np.random.random() < 0.3:
            msg = msg.upper() if np.random.random() < 0.5 else msg
        messages.append(msg)
        labels.append('spam')

    # Shuffle
    indices = np.random.permutation(len(messages))
    messages = [messages[i] for i in indices]
    labels = [labels[i] for i in indices]

    df = pd.DataFrame({'v1': labels, 'v2': messages})
    df.to_csv(filepath, index=False, encoding='utf-8')

    # Print length stats to verify overlap
    ham_lens = [len(m) for m, l in zip(messages, labels) if l == 'ham']
    spam_lens = [len(m) for m, l in zip(messages, labels) if l == 'spam']
    print(f"✅ Generated {n_samples} messages")
    print(f"   Ham  lengths: {np.min(ham_lens)}-{np.max(ham_lens)} chars (mean: {np.mean(ham_lens):.0f})")
    print(f"   Spam lengths: {np.min(spam_lens)}-{np.max(spam_lens)} chars (mean: {np.mean(spam_lens):.0f})")

    return filepath


# Check if dataset exists, generate if not
if not os.path.exists('spam.csv'):
    print("⚠️  spam.csv not found!")
    generate_spam_dataset('spam.csv', n_samples=5000)
else:
    print("✅ spam.csv found, loading...")






# ========================================
# PART 2: Load and Explore Data
# ========================================
print("Loading dataset...")
# Load the dataset
df = pd.read_csv("spam.csv", encoding='utf-8')

# Keep only the relevant columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Convert labels to binary (0 for ham, 1 for spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Basic exploration
print(f"Dataset shape: {df.shape}")
print(f"\nLabel distribution:\n{df['label'].value_counts()}")
print(f"\nPercentage of spam: {df['label'].mean()*100:.2f}%")

# Add text statistics
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print(f"\nAverage text length - Ham: {df[df['label']==0]['text_length'].mean():.2f}")
print(f"Average text length - Spam: {df[df['label']==1]['text_length'].mean():.2f}")

# Visualize label distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot of label distribution
df['label'].value_counts().plot(kind='bar', ax=axes[0], color=['green','red'])
axes[0].set_title("Distribution of Messages", fontsize=14, fontweight='bold')
axes[0].set_xticklabels(['Ham (Legitimate)', 'Spam'], rotation=0)
axes[0].set_xlabel('Message Type')
axes[0].set_ylabel('Count')

# Box plot of text length by label
df.boxplot(column='text_length', by='label', ax=axes[1])
axes[1].set_title("Text Length Distribution", fontsize=14, fontweight='bold')
axes[1].set_xlabel('Message Type (0=Ham, 1=Spam)')
axes[1].set_ylabel('Character Count')
plt.suptitle('')  # Remove default title
plt.tight_layout()
plt.show()

# Show sample messages
print("\n" + "="*50)
print("SAMPLE HAM MESSAGES:")
print("="*50)
for i, text in enumerate(df[df['label']==0]['text'].head(3), 1):
    print(f"{i}. {text[:100]}...")
    
print("\n" + "="*50)
print("SAMPLE SPAM MESSAGES:")
print("="*50)
for i, text in enumerate(df[df['label']==1]['text'].head(3), 1):
    print(f"{i}. {text[:100]}...")

# ========================================
# PART 3: Text Preprocessing
# ========================================
print("\n" + "="*50)
print("PREPROCESSING TEXT...")
print("="*50)

# Initialize preprocessing tools
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Clean and preprocess text data:
    1. Convert to lowercase
    2. Remove numbers
    3. Remove punctuation
    4. Remove stopwords
    5. Apply stemming
    """


# ========================================
# PART 4: Feature Extraction (TF-IDF)
# ========================================
print("\n" + "="*50)
print("VECTORIZING TEXT WITH TF-IDF...")
print("="*50)

# Create TF-IDF vectorizer
# vectorizer = TfidfVectorizer(
#     max_features=3000,  # Use top 3000 most important words
#     min_df=2,           # Ignore words that appear in less than 2 documents
#     max_df=0.95,        # Ignore words that appear in more than 95% of documents
#     ngram_range=(1, 2)  # Use both unigrams and bigrams
# )

# Transform text to numerical features



# ========================================
# PART 5: Split Data
# ========================================


# ========================================
# PART 6: Train Multiple Models
# ========================================


# ========================================
# PART 7: Model Evaluation
# ========================================)


# ========================================
# PART 8: Test with New Messages
# ========================================


# ========================================
# PART 9: Save the Model
# ========================================





















# Box Plots Refreshing
#
# MEDIAN AT TOP OF BOX              MEDIAN AT BOTTOM OF BOX
# (Left/Negative skew)              (Right/Positive skew)
#
#     ┌───────────┐                     ┌───────────┐
#     │───────────│ ← median here       │           │
#     │           │                     │           │
#     │           │                     │───────────│ ← median here
#     └───────────┘                     └───────────┘
#
# More data packed at TOP           More data packed at BOTTOM
# Few low values spread out         Few high values spread out





# =============================================================================
# CLASSIFICATION METRICS CHEATSHEET
# =============================================================================
#
#                        PREDICTED
#                    Positive  Negative
#  ACTUAL  Positive    TP        FN      ← FN = Missed (False Negative)
#          Negative    FP        TN      ← FP = False Alarm
#
# -----------------------------------------------------------------------------
# METRIC        FORMULA                 MEANING (Spam Detection Context)
# -----------------------------------------------------------------------------
# Accuracy      (TP+TN)/(ALL)           Overall correctness (misleading if imbalanced!)
# Precision     TP/(TP+FP)              "Of all SPAM predictions, how many were actually spam?"
#                                       High = Few false alarms (ham marked as spam)
# Recall        TP/(TP+FN)              "Of all ACTUAL spam, how many did we catch?"
#               (Sensitivity)           High = Few missed spam (spam in inbox)
# F1-Score      2*(P*R)/(P+R)           Balance of Precision & Recall (harmonic mean)
# Specificity   TN/(TN+FP)              "Of all ACTUAL ham, how many did we keep?"
# AUC-ROC       Area under ROC curve    Overall separability (1.0=perfect, 0.5=random)
#
# -----------------------------------------------------------------------------
# WHICH TO PRIORITIZE?
# -----------------------------------------------------------------------------
# Spam filter → Favor PRECISION (don't lose important emails to spam folder!)
# Disease test → Favor RECALL (don't miss sick patients!)
# Balanced → Use F1-SCORE
#
# -----------------------------------------------------------------------------
# QUICK INTUITION
# -----------------------------------------------------------------------------
# Precision = "Don't cry wolf"      (avoid false alarms)
# Recall    = "Don't miss the wolf" (catch all positives)
# F1        = "Balance both"
# =============================================================================






# =============================================================================
# ROC CURVE EXPLAINED
# =============================================================================
#
# ROC = Receiver Operating Characteristic
# Origin: WWII radar operators distinguishing enemy planes from noise
#
# =============================================================================
# THE CORE IDEA
# =============================================================================
#
# Your model outputs PROBABILITIES, not just labels:
#
#   "FREE WINNER CLICK NOW"  → 0.95 (95% likely spam)
#   "Hey, dinner at 7?"      → 0.12 (12% likely spam)
#
# THRESHOLD decides the cutoff:
#
#   threshold = 0.5 → if prob ≥ 0.5, predict SPAM
#   threshold = 0.3 → more aggressive (catches more spam, more false alarms)
#   threshold = 0.8 → more conservative (misses some spam, fewer false alarms)
#
# =============================================================================
# WHAT ROC PLOTS
# =============================================================================
#
# X-axis: False Positive Rate (FPR) = FP/(FP+TN) → "ham wrongly flagged as spam"
# Y-axis: True Positive Rate (TPR)  = TP/(TP+FN) → "spam correctly caught" (recall)
#
# Each point = one threshold setting
# ROC curve = all thresholds connected
#
#        ↑ TPR (want HIGH)
#    1.0 ┼──●━━━━━━━━━━━━━━━    ← Perfect model (AUC=1.0)
#        │  ┃                      Hugs top-left corner
#        │  ┃   /                  High TPR at low FPR
#    0.5 ┼  ┃  /
#        │  ┃ /  ← Diagonal = random guess (AUC=0.5)
#        │  ┃/                     Coin flip performance
#    0.0 ┼──●─────────────────→
#        0.0      0.5      1.0
#             FPR (want LOW)
#
# =============================================================================
# AUC (AREA UNDER CURVE)
# =============================================================================
#
#   AUC = 1.0  → Perfect (your model!)
#   AUC = 0.9  → Excellent
#   AUC = 0.8  → Good
#   AUC = 0.7  → Fair
#   AUC = 0.5  → Useless (random guess)
#   AUC < 0.5  → Worse than random (predictions inverted!)
#
# =============================================================================
# INTUITION: METAL DETECTOR ANALOGY
# =============================================================================
#
#   Sensitivity ↑  →  Catches more metal  →  But beeps at everything (high FPR)
#   Sensitivity ↓  →  Fewer false alarms  →  But misses some metal (low TPR)
#
#   Good detector = high TPR without high FPR (curve hugs top-left)
#   AUC = 1.0 means perfect separation at ANY threshold
#
# =============================================================================
# WHEN TO USE ROC
# =============================================================================
#
#   ✓ Comparing models (higher AUC = better)
#   ✓ Choosing optimal threshold for your use case
#   ✓ Works well for balanced AND imbalanced datasets
#   ✗ Don't use if you only care about one specific threshold
#
# =============================================================================




