import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv('datasets/url_phishing.csv')

# Keep required columns
data = data[['url', 'label']]

# Drop missing values
data.dropna(inplace=True)

# Ensure correct type
data['url'] = data['url'].astype(str)
data['label'] = data['label'].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['url'], data['label'], test_size=0.2, random_state=42
)

# URL-specific vectorization (IMPORTANT)
vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save model & vectorizer
pickle.dump(model, open('model/url_model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/url_vectorizer.pkl', 'wb'))

# Accuracy
accuracy = model.score(X_test_vec, y_test)
print(f"URL Model Accuracy: {round(accuracy * 100, 2)}%")

print("✅ URL phishing model trained & saved successfully")
