# ============================================================
# PROJECT 2: Movie Review Sentiment Analyzer
# Skills: Python, NLP, Logistic Regression, TF-IDF
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import warnings
warnings.filterwarnings('ignore')

# ── 1. Sample Movie Reviews Dataset ──
reviews = [
    ("This movie was absolutely fantastic! I loved every second.", "positive"),
    ("An incredible journey, beautifully acted and directed.", "positive"),
    ("One of the best films I have ever seen. Truly inspiring.", "positive"),
    ("Great performances and a gripping storyline. Highly recommend.", "positive"),
    ("A masterpiece of cinema. The plot was brilliant and engaging.", "positive"),
    ("Wonderful film with amazing visuals and emotional depth.", "positive"),
    ("I was blown away by the storytelling and character development.", "positive"),
    ("This exceeded all my expectations. A must-watch for everyone!", "positive"),
    ("Beautifully crafted movie with powerful performances.", "positive"),
    ("I laughed, I cried. One of the most touching films ever made.", "positive"),
    ("Entertaining and fun from start to finish!", "positive"),
    ("A delightful surprise. Really enjoyed every moment.", "positive"),
    ("Solid acting and a well-written script. Very enjoyable.", "positive"),
    ("Good film overall, though a bit slow in the middle.", "positive"),
    ("Not bad at all. Some nice moments and decent performances.", "positive"),
    ("Terrible movie. Waste of time and money.", "negative"),
    ("I fell asleep halfway through. Incredibly boring plot.", "negative"),
    ("The worst film I have seen in years. Awful acting.", "negative"),
    ("Disappointed. The trailer was better than the actual movie.", "negative"),
    ("A complete mess of a story. No direction whatsoever.", "negative"),
    ("Dull, predictable and forgettable. Do not bother watching.", "negative"),
    ("The script was weak and the characters were not believable.", "negative"),
    ("I wanted to walk out but stayed hoping it would get better. It didn't.", "negative"),
    ("Poorly directed and poorly acted. A major letdown.", "negative"),
    ("Total waste of 2 hours. Nothing made sense.", "negative"),
    ("Boring and uninspired. The director clearly had no vision.", "negative"),
    ("The movie dragged on and on with no payoff at the end.", "negative"),
    ("Terrible dialogue and shallow characters. Very disappointing.", "negative"),
    ("One of the most overrated movies I have ever watched.", "negative"),
    ("Not worth your time. Skip this one entirely.", "negative"),
]

df = pd.DataFrame(reviews, columns=['review', 'sentiment'])
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("=" * 55)
print("    PROJECT 2: MOVIE REVIEW SENTIMENT ANALYZER")
print("=" * 55)
print(f"\n📊 Dataset Overview:")
print(f"   Total reviews : {len(df)}")
print(f"   Positive      : {(df['sentiment']=='positive').sum()}")
print(f"   Negative      : {(df['sentiment']=='negative').sum()}")

# ── 2. Text Cleaning ──
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned'] = df['review'].apply(clean_text)

# ── 3. Train/Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label'], test_size=0.25, random_state=42
)

# ── 4. TF-IDF + Logistic Regression ──
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words=None)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42, C=0.5)
model.fit(X_train_vec, y_train)

# ── 5. Evaluate ──
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Trained Successfully!")
print(f"   Accuracy : {acc * 100:.1f}%")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative','Positive']))

# ── 6. Live Predictions ──
print("🎬 Predicting Sentiment on New Reviews:")
new_reviews = [
    "This film was a breathtaking experience. Loved it!",
    "Completely boring and a waste of my evening.",
    "Decent movie, not great but not terrible either.",
    "The worst acting I have ever seen in my entire life.",
    "A hidden gem! So touching and well made.",
]

for review in new_reviews:
    vec = vectorizer.transform([clean_text(review)])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label = "😊 POSITIVE" if pred == 1 else "😞 NEGATIVE"
    conf = max(proba) * 100
    print(f"   {label} ({conf:.0f}%) → \"{review[:50]}\"")
