from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import os
from efficientnet_pytorch import EfficientNet

app = Flask(__name__)
CORS(app)

# Load the PyTorch model
model_path = '../../model/efficientnet_model.pth'
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=6)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

@app.route('/predict', methods=['POST'])
def predict():
    # Decode the image from base64
    image_data = request.json['image']
    header, encoded = image_data.split(",", 1)
    binary_data = base64.b64decode(encoded)

    # Convert binary data to PIL Image
    image = Image.open(io.BytesIO(binary_data))

    # Resize and convert the image for EfficientNet (make sure to match the preprocessing used during training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # EfficientNet uses RGB images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return jsonify({"emotion": predicted.item()})

if __name__ == '__main__':
    app.run(debug=True)
