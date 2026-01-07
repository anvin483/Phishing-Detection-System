from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import shap
from scipy.sparse import hstack, csr_matrix
from utils.url_features import extract_url_features
import os
from datetime import datetime

app = Flask(__name__)

# ================= LOAD MODELS =================

# Email phishing model
email_model = pickle.load(open('model/phishing_model.pkl', 'rb'))
email_vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

# URL phishing model (HYBRID)
url_model = pickle.load(open('model/url_model.pkl', 'rb'))
url_vectorizer = pickle.load(open('model/url_vectorizer.pkl', 'rb'))

# ================= SHAP SETUP =================

# Create SHAP explainer once (for efficiency)
def get_shap_explainer():
    """Initialize SHAP explainer with background data"""
    try:
        # Load or create background dataset
        background_urls = [
            "https://google.com",
            "https://amazon.com",
            "http://secure-login.xyz",
            "https://github.com",
            "http://verify-account.tk"
        ]
        
        # Extract features for background
        tfidf_bg = url_vectorizer.transform(background_urls)
        extra_bg = []
        for url in background_urls:
            feat = extract_url_features(url).flatten()
            extra_bg.append(feat)
        extra_bg = np.array(extra_bg)
        extra_bg_sparse = csr_matrix(extra_bg)
        
        background_X = hstack([tfidf_bg, extra_bg_sparse])
        
        # Create prediction function
        def predict_fn(x):
            return url_model.predict_proba(x)[:, 1]
        
        # Initialize explainer
        explainer = shap.KernelExplainer(
            predict_fn,
            background_X,
            link="identity"
        )
        
        return explainer, background_X
    except Exception as e:
        print(f"Error creating SHAP explainer: {e}")
        return None, None

# Initialize explainer globally
shap_explainer, shap_background = get_shap_explainer()

def generate_shap_for_url(url_text, final_vec):
    """Generate SHAP visualization for a specific URL"""
    try:
        if shap_explainer is None:
            return None
        
        # Get feature names
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
        
        # Compute SHAP values
        shap_values = shap_explainer.shap_values(final_vec, nsamples=50)
        
        # Get top contributing features
        shap_abs = np.abs(shap_values[0])
        top_indices = np.argsort(shap_abs)[-10:][::-1]
        
        # Create bar plot with better styling
        fig, ax = plt.subplots(figsize=(10, 6))
        
        top_features = [feature_names[i] for i in top_indices]
        top_values = [shap_values[0][i] for i in top_indices]
        
        colors = ['#ff4444' if v > 0 else '#4CAF50' for v in top_values]
        
        bars = ax.barh(range(len(top_features)), top_values, color=colors, alpha=0.85, edgecolor='#333', linewidth=0.5)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=10, fontweight='500')
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11, fontweight='600')
        ax.set_title(f'Top Features Influencing Prediction\n"{url_text[:60]}..."', 
                     fontsize=12, fontweight='bold', pad=15)
        ax.axvline(x=0, color='#333', linestyle='--', linewidth=1, alpha=0.7)
        ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'shap_url_{timestamp}.png'
        filepath = os.path.join('static', filename)
        
        clean_old_shap_images()
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename
        
    except Exception as e:
        print(f"Error generating SHAP: {e}")
        return None

def clean_old_shap_images():
    """Remove old SHAP images to save space"""
    try:
        static_dir = 'static'
        shap_files = [f for f in os.listdir(static_dir) if f.startswith('shap_url_')]
        shap_files.sort(reverse=True)
        
        # Keep only the 10 most recent
        for old_file in shap_files[10:]:
            try:
                os.remove(os.path.join(static_dir, old_file))
            except:
                pass
    except Exception as e:
        print(f"Error cleaning old SHAP files: {e}")

# ================= ROUTES =================

@app.route('/', methods=['GET', 'POST'])
def home():
    # Initialize all variables
    email_result = None
    url_result = None
    email_confidence = None
    url_confidence = None
    email_text = None
    url_text = None
    active_tab = 'email'
    shap_image = None

    if request.method == 'POST':
        # Determine which tab was submitted
        active_tab = request.form.get('tab', 'email')

        # ---------- EMAIL DETECTION ----------
        email_text = request.form.get('email_text', '').strip()
        if email_text:
            active_tab = 'email'

            try:
                vect_text = email_vectorizer.transform([email_text])
                pred = email_model.predict(vect_text)[0]
                proba = email_model.predict_proba(vect_text)[0]

                email_confidence = round(max(proba) * 100, 2)
                email_result = (
                    "🚨 Phishing Email Detected"
                    if pred == 1
                    else "✅ Safe Email"
                )
            except Exception as e:
                email_result = f"❌ Error analyzing email: {str(e)}"
                email_confidence = 0

        # ---------- URL DETECTION (HYBRID ML) ----------
        url_text = request.form.get('url_text', '').strip()
        if url_text:
            active_tab = 'url'

            try:
                # 1️⃣ TF-IDF features
                tfidf_vec = url_vectorizer.transform([url_text])

                # 2️⃣ Engineered URL features
                extra_features = extract_url_features(url_text)
                extra_features = np.array(extra_features).flatten().reshape(1, -1)
                extra_vec = csr_matrix(extra_features)

                # 3️⃣ Combine HYBRID features
                final_vec = hstack([tfidf_vec, extra_vec])

                # 4️⃣ ML prediction
                pred = url_model.predict(final_vec)[0]
                proba = url_model.predict_proba(final_vec)[0]
                ml_score = proba[1]

                # 5️⃣ Heuristic rules
                suspicious_keywords = [
                    'login', 'verify', 'secure', 'account',
                    'update', 'paypal', 'amazon', 'bank'
                ]
                suspicious_tlds = ['.xyz', '.top', '.tk', '.ml', '.ga']

                heuristic_flag = (
                    any(k in url_text.lower() for k in suspicious_keywords)
                    and any(tld in url_text.lower() for tld in suspicious_tlds)
                )

                # 6️⃣ FINAL DECISION
                if ml_score > 0.6 or heuristic_flag:
                    url_result = "🚨 Phishing URL Detected"
                else:
                    url_result = "✅ Safe URL"

                url_confidence = round(ml_score * 100, 2)

                # 7️⃣ Generate LIVE SHAP explanation
                shap_image = generate_shap_for_url(url_text, final_vec)

            except Exception as e:
                url_result = f"❌ Error analyzing URL: {str(e)}"
                url_confidence = 0
                import traceback
                print(traceback.format_exc())

    return render_template(
        'index.html',
        email_result=email_result,
        email_confidence=email_confidence,
        email_text=email_text,
        url_result=url_result,
        url_confidence=url_confidence,
        url_text=url_text,
        active_tab=active_tab,
        shap_image=shap_image
    )

# ================= RUN =================

if __name__ == '__main__':
    app.run(debug=True)