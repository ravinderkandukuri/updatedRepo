from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
import string
punct = string.punctuation


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