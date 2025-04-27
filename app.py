from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

# Load pre-trained YOLOv5 model (download automatically if not present)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(device)
model.eval()

@app.route('/')
def home():
    return "Server is running!"

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'message': 'No image provided'}), 400

        # Remove base64 header if present
        if ',' in image_data:
            _, base64_data = image_data.split(',', 1)
        else:
            base64_data = image_data
        
        img_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(img_data)).convert('RGB')  # ensure 3 channels

        # Run inference on the image using YOLOv5
        results = model(image, size=640)  # You can change input size (default 640)
        
        # Extract detected objects
        detected_objects = []
        detections = results.xyxy[0]  # (x1, y1, x2, y2, confidence, class)
        for det in detections:
            class_id = int(det[5])  # Class ID
            class_name = model.names[class_id]  # Map to class name
            detected_objects.append(class_name)

        detected_objects = list(set(detected_objects))  # Unique objects

        return jsonify({'message': 'Objects detected', 'objects': detected_objects})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'message': 'Error processing image', 'error': str(e)}), 500

if __name__ == '__main__':
    # For local testing only (Render uses Gunicorn)
    app.run(host='0.0.0.0', port=5000)
