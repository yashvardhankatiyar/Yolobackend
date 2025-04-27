import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

# Load pre-trained YOLO model from local path (update the path as needed)
model_path = 'models/yolov5.pt'  # Path where your model is located in Render's file system
model = torch.load(model_path)  # Load the model directly

@app.route('/')
def home():
    return "Server is running!"

@app.route('/analyze', methods=['POST'])
def analyze_image():
    data = request.get_json()
    image_data = data.get('image')

    if not image_data:
        return jsonify({'message': 'No image provided'}), 400

    try:
        # Remove base64 header
        header, base64_data = image_data.split(',', 1)
        img_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(img_data))

        # Run inference on the image using YOLOv5
        results = model(image)  # Use the loaded model for inference

        # Extract detected objects
        detected_objects = []
        detections = results.xyxy[0]  # Get detection results (x1, y1, x2, y2, confidence, class)
        for det in detections:
            class_id = int(det[5])  # Class ID of the detected object
            class_name = model.names[class_id]  # Map class ID to class name
            detected_objects.append(class_name)  # Add detected object name to list

        # Remove duplicates (optional, if you only want unique objects)
        detected_objects = list(set(detected_objects))

        # Send back the detected objects
        return jsonify({'message': 'Objects detected', 'objects': detected_objects})

    except Exception as e:
        print(e)
        return jsonify({'message': 'Error processing image'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
