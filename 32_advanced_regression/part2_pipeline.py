import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Încarcă dataset-ul
df = pd.read_csv('apartamente_bucuresti.csv')

# ========================================
# PARTEA 1: SEPARAREA FEATURES & TARGET
# ========================================

# Features (X) și Target (y)
X = df.drop('pret', axis=1)
y = df['pret']

print("📊 STRUCTURA DATELOR:")
print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"\nColoane în X:\n{X.columns.tolist()}")

# ========================================
# PARTEA 2: IDENTIFICAREA TIPURILOR
# ========================================

# Identifică automat coloanele numerice și categorice
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\n🔢 NUMERICAL FEATURES ({len(numerical_features)}):")
print(numerical_features)

print(f"\n🏷️ CATEGORICAL FEATURES ({len(categorical_features)}):")
print(categorical_features)

# ========================================
# PARTEA 3: CREAREA TRANSFORMERS
# ========================================

# Transformer pentru features numerice
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),    # Completează cu media
    ('scaler', StandardScaler())                     # Normalizează
])

# Transformer pentru features categorice
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Completează cu cel mai frecvent
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Encodare
])

print("\n✅ TRANSFORMERS CREAȚI:")
print("  1. Numerical: SimpleImputer(mean) → StandardScaler")
print("  2. Categorical: SimpleImputer(most_frequent) → OneHotEncoder")

# ========================================
# PARTEA 4: COLUMN TRANSFORMER
# ========================================

# Combinăm transformers-ii folosind ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # Drop orice altă coloană nespecificată
)

print("\n🔧 COLUMN TRANSFORMER CREAT!")
print(f"  - Va procesa {len(numerical_features)} numerical features")
print(f"  - Va procesa {len(categorical_features)} categorical features")

# ========================================
# PARTEA 5: TRAIN-TEST SPLIT
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📦 TRAIN-TEST SPLIT:")
print(f"Training set: {X_train.shape[0]} apartamente ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} apartamente ({X_test.shape[0]/len(X)*100:.1f}%)")

# ========================================
# PARTEA 6: FIT & TRANSFORM
# ========================================

# Fit preprocessor pe training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\n🎯 PREPROCESSING COMPLET:")
print(f"  Înainte: {X_train.shape} → După: {X_train_processed.shape}")
print(f"  Features create: {X_train_processed.shape[1]}")

# ========================================
# PARTEA 7: ÎNȚELEGEREA OUTPUT-ULUI
# ========================================

# Obține numele features după OneHotEncoding
feature_names = []

# Numerical features (same names)
feature_names.extend(numerical_features)

# Categorical features (get encoded names from OneHotEncoder)
cat_encoder = preprocessor.named_transformers_['cat']['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
feature_names.extend(cat_feature_names)

print(f"\n📋 TOATE FEATURES DUPĂ PREPROCESSING ({len(feature_names)}):")
for i, name in enumerate(feature_names[:20], 1):  # Afișăm primele 20
    print(f"  {i}. {name}")
if len(feature_names) > 20:
    print(f"  ... și {len(feature_names) - 20} mai multe")

# Exemplu: Cum arată o singură observație după preprocessing
print(f"\n🔍 EXEMPLU - PRIMA OBSERVAȚIE TRAIN:")
print(f"Înainte (primele 5 features):\n{X_train.iloc[0][:5]}")
print(f"\nDupă preprocessing (primele 10 values):\n{X_train_processed[0][:10]}")

# ========================================
# CE SE ÎNTÂMPLĂ ÎN SPATE?
# ========================================

print("""
\n💡 CE FACE COLUMN TRANSFORMER:

1. PENTRU NUMERICAL FEATURES:
   - suprafata: 75 mp (cu missing) → impute cu mean → scale cu StandardScaler → 0.42
   - etaj: 3 → scale → -0.15
   - numar_camere: 2 → scale → 0.38

2. PENTRU CATEGORICAL FEATURES:
   - zona: "Floreasca" → OneHotEncoder → [0,0,0,1,0,0,0,0]
   - balcon: "da" → OneHotEncoder → [1,0]
   - parcare: "nu" → OneHotEncoder → [0,1]

3. CONCATENEAZĂ TOT:
   [0.42, -0.15, 0.38, ..., 0,0,0,1,0,0,0,0, 1,0, 0,1]

4. REZULTAT:
   - Vector numeric complet
   - Gata pentru ML model
   - ZERO data leakage (test folosește parametrii din train!)

🚀 URMĂTORUL PAS: Antrenarea modelelor!
""")