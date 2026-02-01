import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

"""
🌌 WORDS AS VECTORS: The Geometric Revolution

1980s-2000s: Bag of Words & TF-IDF
   "Document is a bag of word counts"
   No meaning, just frequency

2013: Word2Vec
   "Words are points in meaning-space"
   King - Man + Woman = Queen (NOT PROGRAMMED!)
   Machines discovered meaning has geometry

This changed everything.
"""

print("="*70)
print("🌌 FROM WORDS TO VECTORS: The Geometric Revolution")
print("="*70)

# ===== STEP 1: Bag of Words - The Simple Beginning =====
print("\n📊 STEP 1: Bag of Words - Counting Without Meaning")
print("-" * 70)

documents = [
    "The doctor prescribed medicine for the patient",
    "The patient visited the doctor at the clinic",
    "Medicine helps patient recover from illness",
    "The clinic provides excellent medical care"
]

# Create Bag of Words
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print("Documents:")
for i, doc in enumerate(documents):
    print(f"{i+1}. {doc}")

print(f"\nVocabulary ({len(feature_names)} unique words):")
print(feature_names)

print("\nBag of Words Matrix:")
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=feature_names)
print(bow_df)

print("\n💡 EXPLANATION:")
print("   Each row = document")
print("   Each column = word")
print("   Numbers = word count in document")
print("   Problem: 'the' appears most but means least!")

# ===== STEP 2: TF-IDF - Weighting by Importance =====
print("\n\n⚖️  STEP 2: TF-IDF - Term Frequency × Inverse Document Frequency")
print("-" * 70)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                        columns=tfidf_vectorizer.get_feature_names_out())

print("TF-IDF Matrix:")
print(tfidf_df.round(3))

print("\n💡 TF-IDF FORMULA:")
print("   TF (Term Frequency) = count of word in document")
print("   IDF (Inverse Doc Freq) = log(total docs / docs containing word)")
print("   TF-IDF = TF × IDF")
print("\n   RESULT: Common words ('the') get LOW scores")
print("           Rare, meaningful words get HIGH scores")

# Compare 'the' vs 'medicine'
the_tfidf = tfidf_df['the'].mean()
medicine_tfidf = tfidf_df['medicine'].mean()
print(f"\n   Average TF-IDF scores:")
print(f"   'the': {the_tfidf:.4f} (common → low score)")
print(f"   'medicine': {medicine_tfidf:.4f} (meaningful → high score)")

# ===== STEP 3: Document Similarity =====
print("\n\n📏 STEP 3: Document Similarity - Finding Related Content")
print("-" * 70)

# Calculate similarity between documents
similarity_matrix = cosine_similarity(tfidf_matrix)
similarity_df = pd.DataFrame(similarity_matrix,
                             index=[f'Doc{i+1}' for i in range(len(documents))],
                             columns=[f'Doc{i+1}' for i in range(len(documents))])

print("Document Similarity Matrix (0=different, 1=identical):")
print(similarity_df.round(3))

print("\n💡 COSINE SIMILARITY:")
print("   Measures angle between document vectors")
print("   Doc1 & Doc2: Both mention doctor, patient → HIGH similarity")
print("   Doc1 & Doc3: Different focus → LOWER similarity")

# Find most similar documents
new_query = "patient needs medical treatment"
query_vector = tfidf_vectorizer.transform([new_query])
similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

print(f"\n🔍 Query: '{new_query}'")
print("\nMost similar documents:")
for i, sim in enumerate(similarities):
    print(f"   Doc {i+1}: {sim:.4f} - {documents[i]}")

print("\n💡 REAL-WORLD USE:")
print("   • Search engines: match query to documents")
print("   • Recommendation: find similar articles")
print("   • Plagiarism detection: compare document similarity")

# ===== STEP 4: Word2Vec - THE BREAKTHROUGH =====
print("\n\n✨ STEP 4: Word2Vec - Words as Points in Meaning-Space")
print("-" * 70)

# Health-domain sentences for training
health_sentences = [
    "doctor treats patient with medicine",
    "nurse assists doctor at hospital",
    "patient takes medicine for illness",
    "hospital provides medical care",
    "medicine helps cure disease",
    "doctor diagnoses patient symptoms",
    "nurse monitors patient health",
    "medical care improves patient recovery",
    "hospital employs doctor and nurse",
    "patient visits doctor for treatment",
    "prescription medicine reduces symptoms",
    "medical staff includes doctor nurse",
    "patient health improves with treatment",
    "doctor prescribes medicine for patient",
    "nurse administers medicine to patient"
]

# Tokenize
tokenized_sentences = [sentence.split() for sentence in health_sentences]

print("Training Word2Vec on health domain...")
print(f"Corpus: {len(health_sentences)} sentences")

# Train Word2Vec model
w2v_model = Word2Vec(sentences=tokenized_sentences,
                     vector_size=50,  # 50 dimensions
                     window=3,         # Context window
                     min_count=1,
                     workers=4,
                     epochs=100)

print(f"\nVocabulary size: {len(w2v_model.wv)} words")
print(f"Vector dimensions: {w2v_model.wv.vector_size}")

# ===== STEP 5: Word Similarity - Discovering Relationships =====
print("\n\n🔗 STEP 5: Word Relationships - Machines Discover Meaning")
print("-" * 70)

# Find similar words
test_words = ['doctor', 'patient', 'medicine']

for word in test_words:
    if word in w2v_model.wv:
        similar = w2v_model.wv.most_similar(word, topn=3)
        print(f"\n'{word}' is similar to:")
        for similar_word, score in similar:
            print(f"   {similar_word:12s} (similarity: {score:.4f})")

print("\n💡 THE MAGIC:")
print("   Model DISCOVERED these relationships!")
print("   Nobody programmed: 'doctor similar to nurse'")
print("   It learned from context: they appear in similar situations")

# ===== STEP 6: Word Analogies - The Famous Example =====
print("\n\n🎯 STEP 6: Word Analogies - Vector Arithmetic")
print("-" * 70)

print("The famous example: King - Man + Woman = ?")
print("\nIn our medical domain:")

# Try medical analogy: doctor - hospital + clinic = ?
try:
    # This is vector arithmetic!
    result = w2v_model.wv.most_similar(
        positive=['doctor', 'patient'],
        negative=['hospital'],
        topn=1
    )
    print(f"\ndoctor + patient - hospital = {result[0][0]}")
    print(f"Confidence: {result[0][1]:.4f}")
except:
    print("Need more data for analogies, but the concept works!")

print("\n💡 HOW IT WORKS:")
print("   Words are vectors (arrays of numbers)")
print("   'King' = [0.2, 0.8, 0.3, ...]")
print("   'Man' = [0.1, 0.7, 0.2, ...]")
print("   'Woman' = [0.1, 0.1, 0.9, ...]")
print("   ")
print("   King - Man + Woman = [0.2-0.1+0.1, 0.8-0.7+0.1, ...] = ~Queen")
print("   ")
print("   Meaning is GEOMETRY!")

# ===== STEP 7: Visualizing Word Embeddings =====
print("\n\n📊 STEP 7: Visualizing Word Space")
print("-" * 70)

# Get vectors for key medical terms
medical_terms = ['doctor', 'nurse', 'patient', 'medicine',
                 'hospital', 'treatment', 'care', 'health']

vectors = []
labels = []
for term in medical_terms:
    if term in w2v_model.wv:
        vectors.append(w2v_model.wv[term])
        labels.append(term)

# Reduce 50D to 2D using PCA
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=200, alpha=0.6, c=range(len(labels)), cmap='viridis')

for i, label in enumerate(labels):
    plt.annotate(label, xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=12, fontweight='bold')

plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)
plt.title('Word2Vec: Medical Terms in 2D Space\n(50D compressed to 2D for visualization)',
         fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('word2vec_medical_space.png', dpi=150, bbox_inches='tight')

print("Visualization saved: word2vec_medical_space.png")
print("\n💡 WHAT YOU SEE:")
print("   Words cluster by meaning!")
print("   'doctor' & 'nurse' are close")
print("   'patient' is near healthcare terms")
print("   'medicine' & 'treatment' are related")
print("   ")
print("   This spatial structure is what ChatGPT uses!")

# ===== STEP 8: Application - Document Similarity with Embeddings =====
print("\n\n🏥 STEP 8: Real Application - Medical Note Similarity")
print("-" * 70)

medical_notes = [
    "patient presents with fever and cough symptoms",
    "doctor prescribed antibiotics for infection",
    "patient showing improvement after treatment",
    "nurse monitored patient vital signs"
]

def document_vector(doc, model):
    """Convert document to vector by averaging word vectors"""
    words = doc.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

# Convert notes to vectors
note_vectors = [document_vector(note, w2v_model) for note in medical_notes]

# Find similar notes
query = "patient has fever need treatment"
query_vector = document_vector(query, w2v_model)

similarities = [cosine_similarity([query_vector], [note_vec])[0][0]
               for note_vec in note_vectors]

print(f"Query: '{query}'")
print("\nMost similar medical notes:")
for i, (note, sim) in enumerate(zip(medical_notes, similarities)):
    print(f"   {i+1}. [{sim:.4f}] {note}")

print("\n" + "="*70)
print("🎓 WORD EMBEDDINGS MASTERED!")
print("="*70)
print("\nThe Revolution:")
print("   1980s: Words are symbols → count them (TF-IDF)")
print("   2013: Words are vectors → meaning is geometry (Word2Vec)")
print("   2017: Context matters → attention mechanisms (Transformers)")
print("   2023: ChatGPT → all built on this foundation")
print("\n💡 YOU NOW UNDERSTAND:")
print("   How machines learned that words have MEANING")
print("   Why 'King - Man + Woman = Queen' works")
print("   The geometric nature of language")
print("\n🚀 This is what powers modern AI!")
