import os
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from models import TransferLearningResNet  # Replace with the correct model architecture
from ultralytics import YOLO

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the class names (should match the classes from training)
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Define image transformations (should match the ones used during training)
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure this matches your model's input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained YOLO model for fracture detection
fracture_model_path = r"C:\Users\adars\OneDrive\Documents\Project-Pookie\best.pt"
fracture_model = YOLO(fracture_model_path)

# Load the trained TransferLearningResNet model for tumor detection
resnet_model_path = r"C:\Users\adars\OneDrive\Documents\Project-Pookie\resnet_model.pth"
resnet_model = TransferLearningResNet(num_classes=len(class_names))  # Adjust architecture as needed
resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=device, weights_only=True))
resnet_model.to(device)
resnet_model.eval()  # Set model to evaluation mode

# Function to predict the class of a single image using ResNet
def predict_image(model, image):
    image = image.convert('RGB')  # Ensure image is in RGB format
    image = image_transforms(image)  # Apply the same transforms as training
    image = image.unsqueeze(0)  # Add batch dimension

    image = image.to(device)  # Move image to device
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    return predicted_class

# Function to predict fracture using YOLO
def predict_fracture(image):
    image = image.convert('RGB')
    results = fracture_model(image)

    if not results or not hasattr(results[0], 'probs') or results[0].probs is None:
        raise ValueError('Unexpected results format from fracture YOLO model prediction')

    predicted_probs = results[0].probs.data
    predicted_confidence = predicted_probs.max().item()

    # If confidence is above threshold, classify as "fractured", else "not fractured"
    predicted_class = "fractured" if predicted_confidence >= 0.9 else "not fractured"

    return predicted_class

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'AI Medical Image Analysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    try:
        image = Image.open(io.BytesIO(image_file.read()))
    except UnidentifiedImageError:
        return jsonify({'error': 'Cannot identify image file'}), 400

    model_type = request.form.get('model_type')

    try:
        if model_type == 'fracture':
            # YOLO model prediction for fracture
            fracture_class = predict_fracture(image)
            return jsonify({
                'fracture_yolo_class': fracture_class
            })

        elif model_type == 'tumor':
            # TransferLearningResNet model prediction
            tumor_class = predict_image(resnet_model, image)
            if tumor_class == 'no_tumor':
                tumor_prediction = "No Tumor Detected"
            else:
                tumor_prediction = "Tumor Detected"
            return jsonify({
                'resnet_class': tumor_prediction
            })

        else:
            return jsonify({'error': 'Invalid model type selected'}), 400

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)
