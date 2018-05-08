from flask import Flask, request, render_template
from keras.models import load_model, model_from_json
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing import sequence
from flask import jsonify
import os, sys, json
import numpy as np
from wtforms import Form, TextAreaField, validators

MODEL_DIR = './models'

max_words = 100

app = Flask(__name__)

class ReviewForm(Form):
    text = TextAreaField('',[validators.DataRequired(), validators.length(min=15)])

print("loading model...")
json_model = open(os.path.join(MODEL_DIR, 'model.json'),'r')
model = json_model.read()
json_model.close()
model = model_from_json(model)

model.load_weights(os.path.join(MODEL_DIR, 'CNN_best.hdf5'))
#model = load_model(os.path.join(MODEL_DIR, 'CNN_best.hdf5'))
with open(os.path.join(MODEL_DIR, 'dictionary.json')) as jsonfile:
    word_index = json.load(jsonfile)
tokenizer = Tokenizer(num_words = max_words)


def preprocess_text(text):
    word_sequence = text_to_word_sequence(text)
    indices_sequence = np.array([[word_index[word] if word in word_index else 0 for word in word_sequence]])
    indices_sequence = sequence.pad_sequences(indices_sequence, maxlen=max_words)
    x = np.array([indices_sequence.flatten()])
    return x

def classify(text):
    label = {0: 'negative', 1: 'positive'}
    x = preprocess_text(text)
    #print(x)
    #print(model)
    y = model.predict(x)
    #print(y)
    predicted_class = y[0].argmax(axis=-1)
    proba = np.max(y)
    #print(predicted_class)
    return label[predicted_class], proba


@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=["POST"])
def results():
    #try:
    #text = request.get_json()["text"]
    #print(text)
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['text']
        print(review)
        y, prob = classify(review)
        print(y,prob)
        return render_template('results.html',content=review,prediction=y,probability=round(prob*100, 2))
    return render_template('reviewform.html', form=form)

    #return jsonify({'prediction': str(predicted_class)})
    #except:
    #    response = jsonify({"error":'problem predicting'})
    #    response.status_code = 400
    #    return response

if __name__=="__main__":
    app.run(host='0.0.0.0', port = 4444)
