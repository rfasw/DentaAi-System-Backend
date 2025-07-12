import os
import cv2
import numpy as np
import json
import time
from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask_cors import CORS
import platform
import tensorflow as tf
import uuid
from io import BytesIO
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
CORS(app)

# Configuration
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'webm'}

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load models
print("Loading models...")
DESKTOP_MODEL_PATH = 'best_dental_model.h5'
MOBILE_MODEL_PATH = 'dental_model.tflite'
CLASS_NAMES_PATH = 'class_names.json'

desktop_model = None
mobile_interpreter = None
class_names = []

try:
    if os.path.exists(DESKTOP_MODEL_PATH):
        desktop_model = load_model(DESKTOP_MODEL_PATH)
    if os.path.exists(MOBILE_MODEL_PATH):
        mobile_interpreter = tf.lite.Interpreter(model_path=MOBILE_MODEL_PATH)
        mobile_interpreter.allocate_tensors()
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
except Exception as e:
    print(f"Error loading models: {str(e)}")

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_frame(frame, target_size=(160, 160)):
    try:
        frame = cv2.resize(frame, target_size)
        return frame / 255.0
    except Exception as e:
        app.logger.error(f"Error preprocessing frame: {str(e)}")
        return None

def predict_frame(frame, is_mobile=False):
    try:
        frame = preprocess_frame(frame)
        if frame is None:
            return None, None
            
        input_frame = np.expand_dims(frame, axis=0)
        
        if is_mobile and mobile_interpreter:
            input_details = mobile_interpreter.get_input_details()
            output_details = mobile_interpreter.get_output_details()
            mobile_interpreter.set_tensor(input_details[0]['index'], input_frame.astype(np.float32))
            mobile_interpreter.invoke()
            pred = mobile_interpreter.get_tensor(output_details[0]['index'])[0]
        elif desktop_model:
            pred = desktop_model.predict(input_frame, verbose=0)[0]
        else:
            raise Exception("No valid model available")
            
        class_idx = np.argmax(pred)
        return class_names[class_idx], float(pred[class_idx])
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        raise

def detect_device_type(user_agent):
    mobile_keywords = ['iphone', 'ipod', 'android', 'blackberry', 'windows phone', 'nokia', 'samsung', 'mobile']
    return any(keyword in (user_agent or '').lower() for keyword in mobile_keywords)

# Routes
@app.route('/capture_teeth', methods=['POST'])
def capture_teeth():
    """Handle teeth capture from camera"""
    try:
        # Check if camera access is possible
        if os.environ.get('DISABLE_CAMERA', 'false').lower() == 'true':
            return jsonify({
                "error": "Camera access disabled on server",
                "status": "error",
                "solution": "Please use file upload instead"
            }), 400
            
        # Get parameters
        duration = min(10, max(1, int(request.args.get('duration', 5))))
        fps = min(30, max(5, int(request.args.get('fps', 15))))
        
        # Try to open camera
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Camera not accessible")
                
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Process frames
            predictions = []
            frames_processed = 0
            start_time = time.time()
            
            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Predict every nth frame
                if frames_processed % max(1, int(30/fps)) == 0:
                    try:
                        pred, confidence = predict_frame(frame, detect_device_type(request.headers.get('User-Agent')))
                        if pred:
                            predictions.append((pred, confidence))
                    except:
                        continue
                
                frames_processed += 1
                
            cap.release()
            
            if not predictions:
                raise RuntimeError("No frames processed successfully")
                
            # Get most common prediction
            from collections import Counter
            pred_counter = Counter([p[0] for p in predictions])
            most_common = pred_counter.most_common(1)[0][0]
            avg_confidence = sum(p[1] for p in predictions if p[0] == most_common) / pred_counter[most_common]
            
            return jsonify({
                "prediction": most_common,
                "confidence": round(avg_confidence, 4),
                "status": "success",
                "frames_processed": len(predictions)
            })
            
        except Exception as e:
            return jsonify({
                "error": str(e),
                "status": "error",
                "solution": "Check camera permissions or use file upload"
            }), 500
            
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400

@app.route('/predict_image', methods=['POST'])
def predict_image():
    """Handle image file upload"""
    try:
        if 'file' not in request.files:
            raise ValueError("No file uploaded")
            
        file = request.files['file']
        if not file or file.filename == '':
            raise ValueError("Empty filename")
            
        if not allowed_file(file.filename):
            raise ValueError("Invalid file type")
            
        # Save temp file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Read image
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError("Could not read image")
            
        # Predict
        pred, confidence = predict_frame(img, detect_device_type(request.headers.get('User-Agent'))))
        
        if not pred:
            raise RuntimeError("Prediction failed")
            
        return jsonify({
            "prediction": pred,
            "confidence": round(confidence, 4),
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    finally:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

# System info endpoint
@app.route('/system_info', methods=['GET'])
def system_info():
    return jsonify({
        "status": "success",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "models_loaded": {
            "desktop": desktop_model is not None,
            "mobile": mobile_interpreter is not None
        },
        "class_names": class_names,
        "camera_available": False if os.environ.get('DISABLE_CAMERA') else True
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])