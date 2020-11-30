import pickle

import spacy
from flask import Flask, render_template, request

nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
import string
punct = string.punctuation

import spacy

nlp = spacy.load('en_core_web_sm')

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#from google.colab import files
#uploaded = files.upload()

# Commented out IPython magic to ensure Python compatibility.
#setup
import pandas as pd
# %matplotlib inline
from sklearn.feature_extraction.text import TfidfVectorizer

from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)

import string
punct = string.punctuation

#data = pd.read_csv(io.BytesIO(uploaded['Zenbra_cleanedReviews.csv']),sep=',\s+', delimiter=',', encoding="utf-8", skipinitialspace=True)
data= pd.read_csv("Zenbra_cleanedReviews.csv", encoding="latin-1")
data=data.dropna()


def text_data_cleaning(sentence):
	doc = nlp(sentence)

	tokens = []
	for token in doc:
		if token.lemma_ != "-PRON-":
			temp = token.lemma_.lower().strip()
		else:
			temp = token.lower_
		tokens.append(temp)

	cleaned_tokens = []
	for token in tokens:
		if token not in stopwords and token not in punct:
			cleaned_tokens.append(token)
	return cleaned_tokens

tfidf = TfidfVectorizer(tokenizer = text_data_cleaning)
classifier = LinearSVC()

X = data['cleanedReviews']
y = data['is_bad_review']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape, X_test.shape

clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])

clf.fit(X_train, y_train)


# load the model from disk

zenbramodel = pickle.load(open('zenbramodel.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]

		my_prediction = zenbramodel.predict(data)
	return render_template('result.html',prediction = my_prediction)
if __name__ == '__main__':
	app.run(debug=True)
