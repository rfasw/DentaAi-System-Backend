import os
import cv2
import numpy as np
import json
import time
import threading
from flask import Flask, request, jsonify, Response, send_file
from tensorflow.keras.models import load_model
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
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Global variables for video capture
active_streams = {}
camera_lock = threading.Lock()

# Load models
print("Loading models...")
DESKTOP_MODEL_PATH = 'best_dental_model.h5'
desktop_model = load_model(DESKTOP_MODEL_PATH) if os.path.exists(DESKTOP_MODEL_PATH) else None

MOBILE_MODEL_PATH = 'dental_model.tflite'
if os.path.exists(MOBILE_MODEL_PATH):
    mobile_interpreter = tf.lite.Interpreter(model_path=MOBILE_MODEL_PATH)
    mobile_interpreter.allocate_tensors()
    mobile_input_details = mobile_interpreter.get_input_details()
    mobile_output_details = mobile_interpreter.get_output_details()
else:
    mobile_interpreter = None

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Medication database
with open('medications.json', 'r') as f:
    MEDICATION_DATABASE = json.load(f)

def preprocess_frame(frame):
    """Preprocess frame for prediction"""
    if frame is None:
        return None
    frame = cv2.resize(frame, (160, 160))
    return frame / 255.0

def predict_frame(frame, is_mobile=False):
    """Make prediction on a single frame"""
    processed_frame = preprocess_frame(frame)
    if processed_frame is None:
        return None, 0.0
    
    input_frame = np.expand_dims(processed_frame, axis=0)
    
    if is_mobile and mobile_interpreter:
        mobile_interpreter.set_tensor(mobile_input_details[0]['index'], input_frame.astype(np.float32))
        mobile_interpreter.invoke()
        predictions = mobile_interpreter.get_tensor(mobile_output_details[0]['index'])[0]
    elif desktop_model:
        predictions = desktop_model.predict(input_frame, verbose=0)[0]
    else:
        return None, 0.0
    
    class_idx = np.argmax(predictions)
    return class_names[class_idx], float(predictions[class_idx])

def video_capture_thread(stream_id, fps=15, duration=10):
    """Background thread for video capture and processing"""
    with camera_lock:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            active_streams[stream_id]['status'] = 'error'
            active_streams[stream_id]['error'] = 'Could not open camera'
            return
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = os.path.join(PROCESSED_FOLDER, f'{stream_id}.mp4')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        start_time = time.time()
        frame_count = 0
        predictions = []
        confidences = []
        
        try:
            while (time.time() - start_time) < duration and active_streams[stream_id]['active']:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Write frame to video file
                out.write(frame)
                frame_count += 1
                
                # Process every Nth frame based on FPS
                if frame_count % max(1, int(30/fps)) == 0:
                    prediction, confidence = predict_frame(frame)
                    if prediction:
                        predictions.append(prediction)
                        confidences.append(confidence)
                        active_streams[stream_id]['predictions'].append({
                            'prediction': prediction,
                            'confidence': confidence,
                            'timestamp': time.time()
                        })
                
            # Calculate final prediction
            if predictions:
                final_prediction = max(set(predictions), key=predictions.count)
                final_confidence = sum(confidences)/len(confidences)
                active_streams[stream_id]['final_prediction'] = final_prediction
                active_streams[stream_id]['final_confidence'] = final_confidence
                active_streams[stream_id]['video_path'] = output_path
            
        finally:
            cap.release()
            out.release()
            active_streams[stream_id]['active'] = False
            active_streams[stream_id]['completed'] = True

@app.route('/start_capture', methods=['POST'])
def start_capture():
    """Start a new video capture session"""
    stream_id = str(uuid.uuid4())
    duration = request.json.get('duration', 10)
    fps = request.json.get('fps', 15)
    
    active_streams[stream_id] = {
        'active': True,
        'completed': False,
        'predictions': [],
        'final_prediction': None,
        'final_confidence': None,
        'video_path': None,
        'error': None
    }
    
    # Start capture thread
    thread = threading.Thread(
        target=video_capture_thread,
        args=(stream_id, fps, duration)
    )
    thread.start()
    
    return jsonify({
        "status": "success",
        "stream_id": stream_id,
        "message": "Capture started"
    })

@app.route('/prediction_stream/<stream_id>')
def prediction_stream(stream_id):
    """SSE stream for real-time predictions"""
    def generate():
        while stream_id in active_streams and not active_streams[stream_id]['completed']:
            if active_streams[stream_id]['predictions']:
                prediction = active_streams[stream_id]['predictions'].pop(0)
                yield f"data: {json.dumps(prediction)}\n\n"
            time.sleep(0.1)
        
        # Send final result when completed
        if stream_id in active_streams and active_streams[stream_id]['completed']:
            if active_streams[stream_id]['final_prediction']:
                yield f"data: {json.dumps({
                    'final_prediction': active_streams[stream_id]['final_prediction'],
                    'final_confidence': active_streams[stream_id]['final_confidence'],
                    'video_url': f'/video/{stream_id}',
                    'status': 'completed'
                })}\n\n"
            elif active_streams[stream_id]['error']:
                yield f"data: {json.dumps({
                    'error': active_streams[stream_id]['error'],
                    'status': 'error'
                })}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/video/<stream_id>')
def get_video(stream_id):
    """Download captured video"""
    if stream_id not in active_streams or not active_streams[stream_id]['completed']:
        return jsonify({"error": "Video not available"}), 404
    
    video_path = active_streams[stream_id]['video_path']
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
    
    return send_file(
        video_path,
        as_attachment=True,
        download_name=f"dental_scan_{stream_id}.mp4",
        mimetype='video/mp4'
    )

@app.route('/stop_capture/<stream_id>', methods=['POST'])
def stop_capture(stream_id):
    """Stop an active capture session"""
    if stream_id in active_streams:
        active_streams[stream_id]['active'] = False
        return jsonify({"status": "success", "message": "Capture stopped"})
    return jsonify({"error": "Invalid stream ID"}), 404

@app.route('/system_info')
def system_info():
    """Get system information"""
    camera_available = False
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            camera_available = True
            cap.release()
    except:
        pass
    
    return jsonify({
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "models_loaded": {
            "desktop": bool(desktop_model),
            "mobile": bool(mobile_interpreter)
        },
        "class_names": class_names,
        "camera_available": camera_available,
        "status": "success"
    })

def get_health_recommendations():
    """Recommendations for healthy teeth"""
    return [
        {"text": "Brush twice daily with fluoride toothpaste", "icon": "🪥"},
        {"text": "Floss at least once a day", "icon": "🧵"},
        {"text": "Limit sugary foods and drinks", "icon": "🚫🍬"},
        {"text": "Visit your dentist regularly for check-ups", "icon": "👨‍⚕️"},
        {"text": "Consider dental sealants for added protection", "icon": "🛡️"},
        {"text": "Use mouthwash to reduce plaque and bacteria", "icon": "💧"},
        {"text": "Replace your toothbrush every 3-4 months", "icon": "⏳"},
        {"text": "Drink plenty of water throughout the day", "icon": "💧"}
    ]

def get_medication_info(disease):
    """Get medication information"""
    disease = disease.lower()
    return MEDICATION_DATABASE.get(disease, [])

def cleanup_old_files():
    """Clean up old files in upload and processed folders"""
    now = time.time()
    max_age = 3600  # 1 hour
    
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                if file_age > max_age:
                    try:
                        os.remove(file_path)
                        app.logger.info(f"Deleted old file: {file_path}")
                    except Exception as e:
                        app.logger.error(f"Error deleting file {file_path}: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Schedule cleanup every 10 minutes
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=cleanup_old_files, trigger="interval", minutes=10)
    scheduler.start()
    
    app.run(host='0.0.0.0', port=port, threaded=True)