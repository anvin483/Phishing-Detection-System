import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from utils.url_features import extract_url_features

# Load dataset
data = pd.read_csv('datasets/url_phishing.csv')

# Fix column names
data = data[['URL', 'label']]
data.dropna(inplace=True)

data['URL'] = data['URL'].astype(str)
data['label'] = data['label'].astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data['URL'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF on URL chars
vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Extract extra features
X_train_extra = np.vstack([extract_url_features(u) for u in X_train])
X_test_extra = np.vstack([extract_url_features(u) for u in X_test])

# Combine (HYBRID)
X_train_final = hstack([X_train_tfidf, X_train_extra])
X_test_final = hstack([X_test_tfidf, X_test_extra])

# Train model
model = MultinomialNB()
model.fit(X_train_final, y_train)

# Save
pickle.dump(model, open('model/url_model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/url_vectorizer.pkl', 'wb'))

accuracy = model.score(X_test_final, y_test)
print(f"✅ Hybrid URL Model Accuracy: {round(accuracy * 100, 2)}%")
