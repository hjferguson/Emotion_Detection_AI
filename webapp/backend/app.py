from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('../../model/model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    # Decode the image from base64
    image_data = request.json['image']
    header, encoded = image_data.split(",", 1)
    binary_data = base64.b64decode(encoded)

    # Convert binary data to PIL Image
    image = Image.open(io.BytesIO(binary_data))

    # Resize and convert to grayscale
    image = image.resize((48, 48)).convert('L')

    # im saving the image for testing for my sanity.
    save_path = "received_images"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image.save(os.path.join(save_path, "processed_image.jpg"))

    # Convert PIL Image to numpy array and normalize (put in range 0-1)
    image_array = np.asarray(image) / 255.0

    # Add batch dimension
    image_array = image_array.reshape((1, 48, 48, 1))

    # Predict
    print("Input shape:", image_array.shape)
    prediction = model.predict(image_array)
    print("Raw prediction:", prediction)
    
    response = np.argmax(prediction, axis=1)

    return jsonify({"emotion": int(response[0])})

if __name__ == '__main__':
    app.run(debug=True)
