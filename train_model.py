import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv('datasets/Phishing.csv')

# Keep required columns
data = data[['Email Text', 'Email Type']]

# Rename columns
data.columns = ['text', 'label']

# Drop missing values
data.dropna(subset=['text', 'label'], inplace=True)

# Ensure text is string
data['text'] = data['text'].astype(str)

# Convert labels to binary
data['label'] = data['label'].map({
    'Safe Email': 0,
    'Phishing Email': 1
})

# Drop unmapped labels
data.dropna(subset=['label'], inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
pickle.dump(model, open('model/phishing_model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

# Accuracy
accuracy = model.score(X_test_vec, y_test)
print(f"Model Accuracy: {round(accuracy * 100, 2)}%")

print("✅ Email phishing model trained & saved successfully")
