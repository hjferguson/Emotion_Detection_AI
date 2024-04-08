from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

#I initiate a flask app
app = Flask(__name__)

#grab the model that is saved after running train_model.py
model = tf.keras.models.load_model('../../model/model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']

    #process the data.