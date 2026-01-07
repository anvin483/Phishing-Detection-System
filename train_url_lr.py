import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

from utils.url_features import extract_url_features

# ================= LOAD DATA =================

data = pd.read_csv('datasets/url_phishing.csv')
data = data[['URL', 'label']].dropna()

X_train, X_test, y_train, y_test = train_test_split(
    data['URL'], data['label'], test_size=0.2, random_state=42
)

# ================= TF-IDF FEATURES =================

vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ================= EXTRA URL FEATURES =================

X_train_extra = np.array([extract_url_features(u)[0] for u in X_train])
X_test_extra = np.array([extract_url_features(u)[0] for u in X_test])

X_train_extra = csr_matrix(X_train_extra)
X_test_extra = csr_matrix(X_test_extra)

# ================= FINAL FEATURE SET =================

X_train_final = hstack([X_train_tfidf, X_train_extra])
X_test_final = hstack([X_test_tfidf, X_test_extra])

# ================= TRAIN MODEL =================

model = LogisticRegression(max_iter=1000)
model.fit(X_train_final, y_train)

# ================= SAVE =================

pickle.dump(model, open('model/url_lr_model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/url_vectorizer.pkl', 'wb'))

print("✅ Logistic Regression URL model trained successfully")
