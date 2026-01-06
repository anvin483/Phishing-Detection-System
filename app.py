from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained models
email_model = pickle.load(open('model/phishing_model.pkl', 'rb'))
email_vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    email_result = None
    confidence = None

    if request.method == 'POST':
        email_text = request.form.get('email_text')

        if email_text:
            vect_text = email_vectorizer.transform([email_text])
            prediction = email_model.predict(vect_text)[0]
            proba = email_model.predict_proba(vect_text)[0]

            confidence = round(max(proba) * 100, 2)

            if prediction == 1:
                email_result = "🚨 Phishing Email Detected"
            else:
                email_result = "✅ Safe Email"

    return render_template(
        'index.html',
        email_result=email_result,
        confidence=confidence
    )

if __name__ == '__main__':
    app.run(debug=True)
