import pandas as pd
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template
from b2_preprocessing_function import CustomPreprocess

# Use a simple list of stopwords as an alternative to NLTK
stopwords_list = {
    "a", "an", "the", "and", "or", "but", "if", "while", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"
}

# Set maximum sequence length
maxlen = 100

# Load pretrained model
model_path = 'c1_lstm_model_acc_0.864.h5'
pretrained_lstm_model = load_model(model_path)

# Load tokenizer
with open('b3_tokenizer.json') as f:
    tokenizer_json = json.load(f)
    loaded_tokenizer = tokenizer_from_json(tokenizer_json)

# Initialize custom preprocessing class
custom = CustomPreprocess()

# Create Flask app
app = Flask(__name__)

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Define predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input from form
        query_asis = [str(x) for x in request.form.values()]

        # Preprocess input queries
        query_processed_list = [custom.preprocess_text(query) for query in query_asis]

        # Tokenize and pad sequences
        query_tokenized = loaded_tokenizer.texts_to_sequences(query_processed_list)
        query_padded = pad_sequences(query_tokenized, padding='post', maxlen=maxlen)

        # Get predictions
        query_sentiments = pretrained_lstm_model.predict(query_padded)

        # Generate response
        if query_sentiments[0][0] > 0.5:
            result = f"Positive Review with probable IMDb rating as: {np.round(query_sentiments[0][0] * 10, 1)}"
        else:
            result = f"Negative Review with probable IMDb rating as: {np.round(query_sentiments[0][0] * 10, 1)}"

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
