from flask import Flask, render_template, request, jsonify, Blueprint, redirect, url_for, session, send_from_directory, send_file
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import base64
from collections import defaultdict
import os
import tempfile
import pandas as pd
import torch


app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')  # Use an environment variable in production

# Dummy user credentials (Replace with a database in production)
USER_CREDENTIALS = {
    'admin': 'password123',
    'user': 'test123'
}

@app.route('/', methods=['GET', 'POST'])
def login():
    """Render login page and handle authentication."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            return render_template('login.html', error="Both fields are required")

        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            session['user'] = username
            return redirect(url_for('index'))  # Redirect to index after login
        else:
            return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout and redirect to login page."""
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/index')
def index():
    """Render index.html only if the user is logged in."""
    if 'user' not in session:
        return redirect(url_for('login'))  # Redirect to login if user is not authenticated
    return render_template('index.html')

# Load CSV file into a pandas DataFrame - Using relative path
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'animal_safety_tips.csv')
last_detected_animal = None  # Global variable to store last detected animal
try:
    animal_details_df = pd.read_csv(CSV_FILE_PATH)
    animal_details_dict = animal_details_df.set_index('Animal').to_dict(orient='index')
except FileNotFoundError:
    print(f"Warning: Could not find {CSV_FILE_PATH}. Using empty dictionary.")
    animal_details_dict = {}

# Define supported animals based on model capabilities
SUPPORTED_ANIMALS = {
    'person': 'person',
    'bird': 'wild',
    'cat': 'pet',
    'dog': 'pet',
    'horse': 'pet',
    'sheep': 'wild',
    'cow': 'pet',
    'elephant': 'wild',
    'bear': 'wild',
    'zebra': 'wild',
    'giraffe': 'wild'
}

# Cache the model to avoid reloading
model = YOLO('yolov8x.pt').to('cuda') # Load model at startup with FP16 precision

def get_model():
    global model
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO('yolov8x.pt').to(device)
    return model

def get_animal_details(animal_name):
    """Retrieve animal details from the CSV dictionary."""
    return animal_details_dict.get(animal_name.capitalize(), {
        'Description': 'Details not available.',
        'Safety Tips': 'No tips available.'
    })

def process_detection(frame, detection, class_name, conf_threshold=0.3):
    """Process single detection and return detection info, including details."""
    if len(detection) < 6:  # Ensure detection has enough elements
        return None
        
    x1, y1, x2, y2, conf = detection[:5]
    
    if conf < conf_threshold:
        return None
    
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
    if (x2 - x1) * (y2 - y1) < 100:  # Filter out too small detections
        return None
    
    animal_type = SUPPORTED_ANIMALS.get(class_name.lower(), 'unknown')
    animal_details = get_animal_details(class_name)
    
    color_map = {
        'pet': (0, 255, 0),    # Green
        'wild': (0, 0, 255),   # Red
        'farm': (255, 165, 0), # Orange
        'person': (255, 0, 255) # Purple
    }
    
    color = color_map.get(animal_type, (128, 128, 128))
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Add label with confidence
    label = f"{class_name} ({conf:.2f})"
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return {
        'class': class_name,
        'type': animal_type,
        'confidence': float(conf),
        'bbox': [int(x1), int(y1), int(x2), int(y2)],
        'description': animal_details.get('Description', 'Description not available'),
        'safety_tips': animal_details.get('Safety Tips', 'Safety tips not available')
    }

def process_image(img, conf_threshold=0.3):
    """Process image and return detections with optional safety information."""
    # Prepare image tensor (no resizing, use original resolution)
    img_tensor = torch.from_numpy(img).to('cuda')  # Convert to tensor and use FP16 precision

    # Perform inference
    results = model(img_tensor)  # Run the model inference
    
    detections = []
    for r in results.boxes.data.tolist():
        class_name = results.names[int(r[5])]
        if class_name.lower() in SUPPORTED_ANIMALS:
            detection = process_detection(img, r, class_name, conf_threshold)
            if detection:
                detections.append(detection)

    detected_type = 'unknown'
    if detections:
        # Get dominant animal type based on confidence
        detections_summary = defaultdict(float)
        for det in detections:
            detections_summary[det['type']] += det['confidence']
        detected_type = max(detections_summary.items(), key=lambda x: x[1])[0]

    # Draw bounding boxes on the image for visualization
    for detection in detections:
        bbox = detection['bbox']
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(img, f"{detection['class']} ({detection['confidence']:.2f})", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return detections, detected_type, img  # Return the original image with bounding boxes

def process_video(video_path, conf_threshold=0.3):
    model = get_model().to("cuda")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    temp_dir = os.path.abspath("temp")
    os.makedirs(temp_dir, exist_ok=True)

    video_filename = f'processed_{os.getpid()}.mp4'
    temp_output = os.path.join(temp_dir, video_filename)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)  

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        cap.release()
        raise ValueError("Error creating output video file. Check if 'temp/' is writable.")

    frame_count = 0
    unique_detections = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        results = model(frame, conf=conf_threshold)[0]

        for r in results.boxes.data.tolist():
            class_name = results.names[int(r[5])]
            if class_name.lower() in SUPPORTED_ANIMALS:
                detection = process_detection(frame, r, class_name, conf_threshold)
                if detection:
                    existing_conf = unique_detections.get(class_name, {}).get('confidence', 0)
                    if detection['confidence'] > existing_conf:
                        unique_detections[class_name] = detection

        out.write(frame)
        frame_count += 1
        print(f"Frame {frame_count} written.")  # Debugging

    cap.release()
    out.release()

    detected_type = "unknown"
    if unique_detections:
        detected_type = max(unique_detections.values(), key=lambda d: d['confidence'])['type']

    if not os.path.exists(temp_output):
        raise FileNotFoundError(f"Processed video file not found: {temp_output}")

    print(f"Processed video saved at: {temp_output}")
    return temp_output, list(unique_detections.values()), detected_type

@app.route('/detect', methods=['POST'])
def detect():
    global last_detected_animal

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_type = request.form.get('type', '')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        if file_type == 'image':
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            detections, detected_type, processed_img = process_image(img)

            if not detections:
                return jsonify({
                    'error': 'No animals detected',
                    'processed_image': None
                }), 200

            highest_conf_detection = max(detections, key=lambda d: d['confidence'])

            if highest_conf_detection != last_detected_animal:
                last_detected_animal = highest_conf_detection
                _, buffer = cv2.imencode('.jpg', processed_img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')

                return jsonify({
                    'detections': [highest_conf_detection],
                    'dominant_type': detected_type,
                    'processed_image': f'data:image/jpeg;base64,{img_base64}',
                    'alert_required': detected_type == 'wild'
                })

        elif file_type == 'video':
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)

            temp_input = os.path.join(temp_dir, f'input_{os.getpid()}.mp4')
            file.save(temp_input)

            try:
                output_path, unique_detections, detected_type = process_video(temp_input)

                if not unique_detections:
                    return jsonify({'error': 'No animals detected'}), 200

                highest_conf_detection = max(unique_detections, key=lambda d: d['confidence'])

                if highest_conf_detection != last_detected_animal:
                    last_detected_animal = highest_conf_detection

                    return jsonify({
                        'detections': [highest_conf_detection],
                        'dominant_type': detected_type,
                        'alert_required': detected_type == 'wild'
                    })

            finally:
                os.remove(temp_input) if os.path.exists(temp_input) else None

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
