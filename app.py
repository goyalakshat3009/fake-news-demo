from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

port_stem = PorterStemmer()
stop_words = set(stopwords.words("english"))


def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()

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

    transformed_news = stemming(news)

    vector_input = vectorizer.transform([transformed_news])

    prediction = model.predict(vector_input)

    if prediction[0] == 0:
        result = "REAL NEWS"
        status = "real"
    else:
        result = "FAKE NEWS"
        status = "fake"

    return render_template(
        'index.html',
        prediction=result,
        status=status,
        news_text=news
    )


if __name__ == '__main__':
    app.run(debug=True)