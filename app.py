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

# Debug mode - set to False in production
app.config['DEBUG'] = True

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load models based on device type
print("Loading models...")
start_time = time.time()

# Load desktop model (full Keras model)
DESKTOP_MODEL_PATH = 'best_dental_model.h5'
desktop_model = None
if os.path.exists(DESKTOP_MODEL_PATH):
    try:
        desktop_model = load_model(DESKTOP_MODEL_PATH)
        print(f"Desktop model loaded from {DESKTOP_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading desktop model: {str(e)}")

# Load mobile model (optimized for mobile devices)
MOBILE_MODEL_PATH = 'dental_model.tflite'
mobile_interpreter = None
if os.path.exists(MOBILE_MODEL_PATH):
    try:
        mobile_interpreter = tf.lite.Interpreter(model_path=MOBILE_MODEL_PATH)
        mobile_interpreter.allocate_tensors()
        mobile_input_details = mobile_interpreter.get_input_details()
        mobile_output_details = mobile_interpreter.get_output_details()
        print(f"Mobile model (TFLite) loaded from {MOBILE_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading mobile model: {str(e)}")

# Load class names
CLASS_NAMES_PATH = 'class_names.json'
class_names = []
if os.path.exists(CLASS_NAMES_PATH):
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        print(f"Class names loaded: {class_names}")
    except Exception as e:
        print(f"Error loading class names: {str(e)}")
else:
    print(f"Class names file not found at {CLASS_NAMES_PATH}")

print(f"Models loaded in {time.time() - start_time:.2f} seconds")

def allowed_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def allowed_image_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def preprocess_frame(frame):
    """Preprocess frame for model prediction"""
    if frame is None:
        return None
        
    try:
        frame = cv2.resize(frame, (160, 160))  # Changed from (224, 224) to (160, 160)
        frame = frame / 255.0
        return frame
    except Exception as e:
        app.logger.error(f"Error preprocessing frame: {str(e)}")
        return None

def predict_image(img_path, is_mobile=False):
    """Predict dental condition from image"""
    try:
        img = image.load_img(img_path, target_size=(160, 160))  # Changed from (224, 224) to (160, 160)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        if is_mobile and mobile_interpreter:
            # Mobile prediction using TFLite
            mobile_interpreter.set_tensor(mobile_input_details[0]['index'], img_array.astype(np.float32))
            mobile_interpreter.invoke()
            predictions = mobile_interpreter.get_tensor(mobile_output_details[0]['index'])[0]
        elif desktop_model:
            # Desktop prediction using full Keras model
            predictions = desktop_model.predict(img_array, verbose=0)[0]
        else:
            raise Exception("No valid model available for prediction")
        
        class_idx = np.argmax(predictions)
        return class_names[class_idx], float(predictions[class_idx])
    except Exception as e:
        app.logger.error(f"Error predicting image: {str(e)}")
        raise

def detect_device_type(user_agent):
    """Detect if request is from mobile device"""
    if not user_agent:
        return False
    
    user_agent = user_agent.lower()
    mobile_keywords = ['iphone', 'ipod', 'android', 'blackberry', 
                       'windows phone', 'nokia', 'samsung', 'mobile']
    
    return any(keyword in user_agent for keyword in mobile_keywords)

# Medication database with free image URLs
MEDICATION_DATABASE = {
    "cavity": [
        {
            "name": "Fluoride Toothpaste",
            "generic": "Sodium Fluoride",
            "description": "Helps prevent tooth decay by strengthening tooth enamel",
            "image": "",
            "image_alt": "Fluoride toothpaste tube",
            "dosage": "Use a pea-sized amount twice daily",
            "side_effects": "May cause mild gum irritation in some users"
        },
        {
            "name": "Dental Filling",
            "generic": "Composite Resin",
            "description": "Restores decayed teeth to normal function and shape",
            "image": "",
            "image_alt": "Dental filling procedure",
            "dosage": "Applied by dentist as needed",
            "side_effects": "Temporary sensitivity after procedure"
        },
        {
            "name": "Anticavity Mouthwash",
            "generic": "Sodium Fluoride",
            "description": "Provides extra protection against cavities",
            "image": "",
            "image_alt": "Anticavity mouthwash bottle",
            "dosage": "10ml twice daily after brushing",
            "side_effects": "May temporarily stain teeth with prolonged use"
        }
    ],
    "gingivitis": [
        {
            "name": "Antibacterial Mouthwash",
            "generic": "Chlorhexidine",
            "description": "Reduces bacteria that cause gum disease",
            "image": "",
            "image_alt": "Antibacterial mouthwash bottle",
            "dosage": "15ml twice daily for 30 seconds",
            "side_effects": "May alter taste sensation temporarily"
        },
        {
            "name": "Medicated Toothpaste",
            "generic": "Stannous Fluoride",
            "description": "Reduces gingivitis and bleeding gums",
            "image": "",
            "image_alt": "Medicated toothpaste tube",
            "dosage": "Use twice daily instead of regular toothpaste",
            "side_effects": "May cause temporary tooth staining"
        },
        {
            "name": "Gum Health Rinse",
            "generic": "Cetylpyridinium Chloride",
            "description": "Targets gum inflammation and promotes healing",
            "image": "",
            "image_alt": "Gum health rinse bottle",
            "dosage": "Use 20ml after brushing twice daily",
            "side_effects": "Mild burning sensation possible"
        }
    ],
    "tooth decay": [
        {
            "name": "Calcium Supplements",
            "generic": "Calcium Carbonate",
            "description": "Strengthens teeth and prevents decay",
            "image": "",
            "image_alt": "Calcium supplement tablets",
            "dosage": "500mg twice daily with meals",
            "side_effects": "May cause constipation in some users"
        },
        {
            "name": "Antibiotic Treatment",
            "generic": "Amoxicillin",
            "description": "Treats bacterial infections in teeth",
            "image": "",
            "image_alt": "Antibiotic capsules",
            "dosage": "As prescribed by dentist (typically 500mg 3x daily)",
            "side_effects": "May cause stomach upset, take with food"
        },
        {
            "name": "Dental Restoration Kit",
            "generic": "Glass Ionomer Cement",
            "description": "Temporary solution for tooth decay repair",
            "image": "",
            "image_alt": "Dental restoration kit",
            "dosage": "Apply as directed for temporary fillings",
            "side_effects": "Temporary sensitivity possible"
        }
    ],
    "healthy teeth": [
        {
            "name": "Regular Toothpaste",
            "generic": "Sodium Fluoride",
            "description": "Maintains oral health and fresh breath",
            "image": "",
            "image_alt": "Regular toothpaste tube",
            "dosage": "Use twice daily",
            "side_effects": "None when used as directed"
        },
        {
            "name": "Dental Floss",
            "generic": "Nylon or PTFE",
            "description": "Removes plaque between teeth",
            "image": "",
            "image_alt": "Dental floss container",
            "dosage": "Use once daily",
            "side_effects": "May cause gum bleeding if used too aggressively"
        },
        {
            "name": "Tongue Cleaner",
            "generic": "Stainless steel or plastic",
            "description": "Removes bacteria from tongue surface",
            "image": "",
            "image_alt": "Tongue cleaning tool",
            "dosage": "Use once daily",
            "side_effects": "None when used gently"
        }
    ]
}

def get_medication_info(disease):
    """Get medication information with static image URLs"""
    disease = disease.lower()
    return MEDICATION_DATABASE.get(disease, [])

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

def annotate_frame(frame, prediction, confidence):
    """Add prediction annotations to the frame"""
    if frame is None:
        return None
        
    try:
        # Draw prediction text
        text = f"{prediction} ({confidence:.1%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Get text size to position it properly
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position text at top left
        x = 10
        y = text_height + 10
        
        # Draw background rectangle
        cv2.rectangle(frame, (x - 5, y - text_height - 5), 
                     (x + text_width + 5, y + 5), 
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness)
        
        return frame
    except Exception as e:
        app.logger.error(f"Error annotating frame: {str(e)}")
        return frame

@app.route('/medication_images', methods=['GET'])
def get_medication_images():
    """Endpoint to get all medication images data for frontend reference"""
    try:
        # Return the complete medication database with image URLs
        return jsonify({
            "status": "success",
            "medication_data": MEDICATION_DATABASE,
            "note": "Replace example.com with your actual image hosting domain"
        })
    except Exception as e:
        app.logger.error(f"Error getting medication images: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/')
def home():
    return """
    <h1>Dental Health Classifier API</h1>
    <p>Endpoints:</p>
    <ul>
        <li>POST /scan_teeth - For video file scans</li>
        <li>POST /capture_teeth - Capture from camera (index 0)</li>
        <li>POST /predict_image - For single image predictions</li>
        <li>GET /video/<filename> - Download processed video</li>
        <li>GET /system_info - Server system information</li>
        <li>GET /medication_images - Get all medication image references</li>
    </ul>
    <p>Debug mode: {}</p>
    <p>Available models: Desktop ({}) | Mobile ({})</p>
    """.format(
        app.config['DEBUG'],
        "Loaded" if desktop_model else "Not available",
        "Loaded" if mobile_interpreter else "Not available"
    )

@app.route('/scan_teeth', methods=['POST'])
def scan_teeth():
    """Endpoint for video file upload and prediction"""
    app.logger.info("Received teeth scan request")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part', 'status': 'error'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 'error'}), 400
        
    if not allowed_video_file(file.filename):
        return jsonify({'error': 'Invalid file type', 'status': 'error'}), 400
    
    # Detect device type from User-Agent
    user_agent = request.headers.get('User-Agent', '')
    is_mobile = detect_device_type(user_agent)
    app.logger.info(f"Device detection: User-Agent={user_agent[:50]}... | Mobile={is_mobile}")
    
    unique_id = str(uuid.uuid4())
    processed_filename = f"processed_{unique_id}.mp4"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        
    try:
        # Save video to temporary file
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)
        app.logger.info(f"Video saved to {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            os.remove(video_path)
            return jsonify({'error': 'Could not open video', 'status': 'error'}), 500
        
        # Validate video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 1000:
            fps = 30.0
            app.logger.warning(f"Invalid FPS, using default 30.0")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 0:
            frame_count = 0
            
        duration = frame_count / fps if fps > 0 else 0
        app.logger.info(f"Video info: {frame_count} frames, {fps:.1f} FPS, {duration:.1f} seconds")
        
        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Validate dimensions
        if width <= 0 or height <= 0:
            cap.release()
            os.remove(video_path)
            return jsonify({'error': 'Invalid video dimensions', 'status': 'error'}), 400
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))
        
        predictions = []
        confidences = []
        processed_count = 0
        frame_skip = max(1, int(round(fps)))  # Process ~1 frame/sec
        start_time = time.time()
        max_duration = 5  # Max processing time (seconds)

        # Robust frame processing
        frame_index = 0
        while (time.time() - start_time) < max_duration:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Write original frame to output video
            out.write(frame)
            
            # Process frames at ~1 FPS rate
            if frame_index % frame_skip == 0:
                try:
                    # Preprocess and predict
                    processed_frame = preprocess_frame(cv2.resize(frame.copy(), (160, 160)))  # Changed from (224, 224) to (160, 160)
                    if processed_frame is None:
                        continue
                        
                    input_frame = np.expand_dims(processed_frame, axis=0)
                    
                    if is_mobile and mobile_interpreter:
                        # Mobile prediction using TFLite
                        mobile_interpreter.set_tensor(mobile_input_details[0]['index'], input_frame.astype(np.float32))
                        mobile_interpreter.invoke()
                        pred = mobile_interpreter.get_tensor(mobile_output_details[0]['index'])[0]
                    elif desktop_model:
                        # Desktop prediction
                        pred = desktop_model.predict(input_frame, verbose=0)[0]
                    else:
                        cap.release()
                        out.release()
                        os.remove(video_path)
                        return jsonify({'error': 'No model available for prediction', 'status': 'error'}), 500
                    
                    class_idx = np.argmax(pred)
                    class_name = class_names[class_idx]
                    confidence = float(pred[class_idx])
                    
                    predictions.append(class_idx)
                    confidences.append(confidence)
                    processed_count += 1
                    
                    # Add prediction to frame (we'll annotate the frame in the output)
                    frame = annotate_frame(frame, class_name, confidence)
                except Exception as e:
                    app.logger.error(f"Error processing frame {frame_index}: {str(e)}")
                    continue
            
            frame_index += 1
        
        cap.release()
        out.release()
        os.remove(video_path)
        app.logger.info(f"Processed {processed_count} frames")
        
        # Fallback for no predictions
        if not predictions:
            app.logger.warning("No frames processed - attempting fallback")
            cap2 = cv2.VideoCapture(video_path)
            if cap2.isOpened():
                ret, frame = cap2.read()
                if ret:
                    try:
                        # Process single frame
                        processed_frame = preprocess_frame(cv2.resize(frame.copy(), (160, 160)))  # Changed from (224, 224) to (160, 160)
                        if processed_frame is not None:
                            input_frame = np.expand_dims(processed_frame, axis=0)
                            
                            if is_mobile and mobile_interpreter:
                                mobile_interpreter.set_tensor(mobile_input_details[0]['index'], input_frame.astype(np.float32))
                                mobile_interpreter.invoke()
                                pred = mobile_interpreter.get_tensor(mobile_output_details[0]['index'])[0]
                            elif desktop_model:
                                pred = desktop_model.predict(input_frame, verbose=0)[0]
                            else:
                                cap2.release()
                                return jsonify({'error': 'No model available for prediction', 'status': 'error'}), 500
                            
                            class_idx = np.argmax(pred)
                            class_name = class_names[class_idx]
                            confidence = float(pred[class_idx])
                            
                            predictions = [class_idx]
                            confidences = [confidence]
                            processed_count = 1
                            
                            app.logger.info(f"Fallback processed 1 frame")
                    except Exception as e:
                        app.logger.error(f"Error in fallback processing: {str(e)}")
                cap2.release()
        
        if not predictions:
            return jsonify({'error': 'No frames processed', 'status': 'error'}), 500
        
        # Get most frequent prediction
        prediction_idx = max(set(predictions), key=predictions.count)
        class_name = class_names[prediction_idx]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Prepare response
        response = {
            "prediction": class_name,
            "confidence": round(confidence, 4),
            "status": "success",
            "processed_frames": processed_count,
            "model_used": "mobile" if is_mobile else "desktop",
            "video_url": f"/video/{processed_filename}",
            "video_id": unique_id
        }
        
        # Add recommendations or medications
        if class_name.lower() == "healthy teeth":
            response["recommendations"] = get_health_recommendations()
        else:
            response["medications"] = get_medication_info(class_name)
        
        app.logger.info(f"Prediction: {class_name} ({confidence:.2%}) using {'mobile' if is_mobile else 'desktop'} model")
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Error processing video: {str(e)}")
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/capture_teeth', methods=['POST'])
def capture_teeth():
    """Endpoint to capture video from camera (index 0) and predict"""
    app.logger.info("Received camera capture request")
    
    # Detect device type from User-Agent
    user_agent = request.headers.get('User-Agent', '')
    is_mobile = detect_device_type(user_agent)
    app.logger.info(f"Device detection: User-Agent={user_agent[:50]}... | Mobile={is_mobile}")
    
    unique_id = str(uuid.uuid4())
    processed_filename = f"processed_{unique_id}.mp4"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
    
    # Get request parameters
    duration = request.args.get('duration', default=5, type=int)
    fps = request.args.get('fps', default=15, type=int)
    
    # Validate parameters
    if duration < 1 or duration > 10:
        return jsonify({'error': 'Invalid duration (1-10 seconds)', 'status': 'error'}), 400
    if fps < 5 or fps > 30:
        return jsonify({'error': 'Invalid FPS (5-30)', 'status': 'error'}), 400
        
    try:
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open camera', 'status': 'error'}), 500
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))
        
        predictions = []
        confidences = []
        processed_frames = 0
        total_frames = duration * fps
        
        app.logger.info(f"Capturing from camera for {duration} seconds at {fps} FPS")
        
        # Start capture timer
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every Nth frame based on FPS
            if processed_frames % max(1, int(30/fps)) == 0:
                try:
                    # Preprocess and predict
                    processed_frame = preprocess_frame(cv2.resize(frame.copy(), (160, 160)))  # Changed from (224, 224) to (160, 160)
                    if processed_frame is None:
                        continue
                        
                    input_frame = np.expand_dims(processed_frame, axis=0)
                    
                    if is_mobile and mobile_interpreter:
                        # Mobile prediction using TFLite
                        mobile_interpreter.set_tensor(mobile_input_details[0]['index'], input_frame.astype(np.float32))
                        mobile_interpreter.invoke()
                        pred = mobile_interpreter.get_tensor(mobile_output_details[0]['index'])[0]
                    elif desktop_model:
                        # Desktop prediction
                        pred = desktop_model.predict(input_frame, verbose=0)[0]
                    else:
                        cap.release()
                        out.release()
                        return jsonify({'error': 'No model available for prediction', 'status': 'error'}), 500
                    
                    class_idx = np.argmax(pred)
                    class_name = class_names[class_idx]
                    confidence = float(pred[class_idx])
                    
                    predictions.append(class_idx)
                    confidences.append(confidence)
                    
                    # Add prediction to frame
                    frame = annotate_frame(frame, class_name, confidence)
                except Exception as e:
                    app.logger.error(f"Error processing frame: {str(e)}")
                    continue
            
            # Write frame to output video
            out.write(frame)
            processed_frames += 1
        
        cap.release()
        out.release()
        app.logger.info(f"Processed {len(predictions)} predictions from {processed_frames} frames")
        
        if not predictions:
            return jsonify({'error': 'No frames processed', 'status': 'error'}), 500
        
        # Get most frequent prediction
        prediction_idx = max(set(predictions), key=predictions.count)
        class_name = class_names[prediction_idx]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Prepare response
        response = {
            "prediction": class_name,
            "confidence": round(confidence, 4),
            "status": "success",
            "processed_frames": len(predictions),
            "model_used": "mobile" if is_mobile else "desktop",
            "video_url": f"/video/{processed_filename}",
            "video_id": unique_id
        }
        
        # Add recommendations or medications
        if class_name.lower() == "healthy teeth":
            response["recommendations"] = get_health_recommendations()
        else:
            response["medications"] = get_medication_info(class_name)
        
        app.logger.info(f"Prediction: {class_name} ({confidence:.2%}) using {'mobile' if is_mobile else 'desktop'} model")
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Error capturing from camera: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/video/<filename>', methods=['GET'])
def get_video(filename):
    """Endpoint to download processed video"""
    try:
        # Secure filename check
        if not filename.endswith('.mp4') or '..' in filename or filename.startswith('/'):
            return jsonify({'error': 'Invalid filename', 'status': 'error'}), 400
            
        video_path = os.path.join(PROCESSED_FOLDER, filename)
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video not found', 'status': 'error'}), 404
            
        # Stream video response
        return send_file(
            video_path,
            as_attachment=True,
            download_name=f"dental_scan_{filename}",
            mimetype='video/mp4'
        )
    except Exception as e:
        app.logger.error(f"Error serving video: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/predict_image', methods=['POST'])
def predict_image_endpoint():
    """Alternative endpoint for single image prediction"""
    app.logger.info("Received image prediction request")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part', 'status': 'error'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 'error'}), 400
        
    if not allowed_image_file(file.filename):
        return jsonify({'error': 'Invalid file type', 'status': 'error'}), 400
    
    # Detect device type from User-Agent
    user_agent = request.headers.get('User-Agent', '')
    is_mobile = detect_device_type(user_agent)
    app.logger.info(f"Device detection: User-Agent={user_agent[:50]}... | Mobile={is_mobile}")
        
    try:
        # Save temporary file
        filename = secure_filename(file.filename)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)
        app.logger.info(f"Image saved to {img_path}")
        
        # Make prediction
        class_name, confidence = predict_image(img_path, is_mobile)
        os.remove(img_path)
        
        # Prepare response
        response = {
            "prediction": class_name,
            "confidence": round(confidence, 4),
            "status": "success",
            "model_used": "mobile" if is_mobile else "desktop"
        }
        
        # Add recommendations or medications
        if class_name.lower() == "healthy teeth":
            response["recommendations"] = get_health_recommendations()
        else:
            response["medications"] = get_medication_info(class_name)
        
        app.logger.info(f"Prediction: {class_name} ({confidence:.2%}) using {'mobile' if is_mobile else 'desktop'} model")
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/system_info', methods=['GET'])
def system_info():
    """Endpoint to get system information"""
    # Check camera availability
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
        "debug_mode": app.config['DEBUG'],
        "camera_available": camera_available,
        "status": "success"
    })

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
    
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])