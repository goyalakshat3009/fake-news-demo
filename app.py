from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords automatically if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
\
app = Flask(__name__)

# Load model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

port_stem = PorterStemmer()
stop_words = set(stopwords.words("english"))


def stemming(content):
    content = re.sub(r'[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()

    content = [
        port_stem.stem(word)
        for word in content
        if word not in stop_words
    ]

    return ' '.join(content)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    news = request.form['news']

    processed_news = stemming(news)

    vector_input = vectorizer.transform([processed_news])

    prediction = model.predict(vector_input)[0]

    confidence = round(
        max(model.predict_proba(vector_input)[0]) * 100,
        2
    )

    if prediction == 0:
        result = "✅ REAL NEWS"
        status = "real"
    else:
        result = "❌ FAKE NEWS"
        status = "fake"

    return render_template(
        'index.html',
        prediction=result,
        confidence=confidence,
        status=status,
        news_text=news
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)