import pickle
import numpy as np
import matplotlib.pyplot as plt

model = pickle.load(open('model/url_lr_model.pkl', 'rb'))

# Get coefficients
coefficients = model.coef_[0]

# Top features
top_idx = np.argsort(np.abs(coefficients))[-15:]
top_weights = coefficients[top_idx]

plt.figure(figsize=(8,5))
plt.barh(range(len(top_weights)), top_weights)
plt.title("Top URL Phishing Indicators (Logistic Regression)")
plt.xlabel("Feature Weight")
plt.tight_layout()
plt.savefig("static/feature_importance.png")
plt.show()
