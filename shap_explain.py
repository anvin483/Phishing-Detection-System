import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from utils.url_features import extract_url_features

# ================= LOAD MODEL =================
url_model = pickle.load(open("model/url_model.pkl", "rb"))
url_vectorizer = pickle.load(open("model/url_vectorizer.pkl", "rb"))

print(f"Model type: {type(url_model)}")

# ================= SAMPLE URLS =================
sample_urls = [
    "http://secure-login-paypal.xyz",
    "https://amazon.com/account",
    "http://verify-bank-update.ml",
    "https://google.com"
]

# ================= FEATURE EXTRACTION =================
tfidf_features = url_vectorizer.transform(sample_urls)

extra_features = []
for url in sample_urls:
    features = extract_url_features(url)
    extra_features.append(features.flatten())

extra_features = np.array(extra_features)
extra_features_sparse = csr_matrix(extra_features)

# Combine hybrid features
X = hstack([tfidf_features, extra_features_sparse])

# ================= DEBUG INFO =================
print(f"TF-IDF features shape: {tfidf_features.shape}")
print(f"Extra features shape: {extra_features.shape}")
print(f"Combined features shape: {X.shape}")

# ================= FEATURE NAMES =================
tfidf_names = url_vectorizer.get_feature_names_out()
engineered_names = [
    "url_length",
    "num_dots",
    "num_digits",
    "has_https",
    "has_suspicious_words",
    "has_suspicious_tld"
]

feature_names = list(tfidf_names) + engineered_names

# ================= SHAP EXPLAINER =================
try:
    # For Naive Bayes, we need to use KernelExplainer or create a wrapper
    # KernelExplainer is model-agnostic but slower
    
    # Create a prediction function
    def predict_fn(x):
        return url_model.predict_proba(x)[:, 1]  # Probability of phishing class
    
    # Sample a background dataset (using all our samples as background)
    background = shap.sample(X, min(100, X.shape[0]))  # Use up to 100 samples
    
    print("Creating SHAP explainer (this may take a moment)...")
    explainer = shap.KernelExplainer(
        predict_fn,
        background,
        link="identity"
    )
    
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X, nsamples=100)
    
    # ================= PLOT =================
    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values,
        X.toarray(),
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False
    )
    
    plt.title("Top URL Phishing Indicators (SHAP)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("static/shap_url_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ SHAP graph saved as static/shap_url_importance.png")
    
    # Also create a waterfall plot for individual prediction
    plt.figure(figsize=(10, 6))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        feature_names=feature_names,
        max_display=10,
        show=False
    )
    plt.title(f"SHAP Explanation for: {sample_urls[0]}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("static/shap_url_waterfall.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ SHAP waterfall plot saved as static/shap_url_waterfall.png")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()