from flask import Flask, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""

    if request.method == 'POST':
        text = request.form['text']
        text_vector = vectorizer.transform([text])
        result = model.predict(text_vector)[0]

        if result == 1:
            prediction = "Phishing Email ⚠️"
        else:
            prediction = "Safe Email ✅"

    return f"""
    <h2>Phishing Detection System</h2>
    <form method="post">
        <textarea name="text" rows="6" cols="60" placeholder="Paste email text here..."></textarea><br><br>
        <input type="submit" value="Check">
    </form>
    <h3>{prediction}</h3>
    """

if __name__ == '__main__':
    app.run(debug=True)
