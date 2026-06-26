import os
# Configure TensorFlow to prevent macOS thread locks in subprocesses
os.environ['XNNPACK_DISABLE'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import urllib.request
import numpy as np
import mediapipe as mp
import tensorflow as tf
import gradio as gr
from deepface import DeepFace

# Constants & Paths
FINE_TUNED_MODEL_PATH = 'fine_tuned_facenet.keras'
CLASS_NAMES_PATH = 'class_names.npy'
DATA_DIR = 'dataset_cleaned'
MODEL_PATH = 'blaze_face_short_range.tflite'
MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'

# Verify if fine-tuned weights exist
has_fine_tuned = os.path.exists(FINE_TUNED_MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH)

if has_fine_tuned:
    print("Loading custom fine-tuned Keras model...")
    model = tf.keras.models.load_model(FINE_TUNED_MODEL_PATH)
    # The third-from-last layer is the 128-dimensional normalized face embedding layer
    embedding_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-3].output)
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
    
    print("Caching database face embeddings...")
    db_embeddings = []
    db_identities = []
    db_classes = []
    
    for class_name in class_names:
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_path):
            continue
        img_names = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        for img_name in img_names:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            # Convert to RGB to match Gradio inputs
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (160, 160))
            img_normalized = img_resized.astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            emb = embedding_model.predict(img_batch, verbose=0)[0]
            db_embeddings.append(emb)
            db_identities.append(img_path)
            db_classes.append(class_name)
            
    db_embeddings = np.array(db_embeddings)
    print(f"Cached {len(db_embeddings)} database embeddings successfully.")
else:
    print("Warning: Fine-tuned model files not found. Run train.py first.")
    model, embedding_model, class_names, db_embeddings, db_identities, db_classes = None, None, None, None, None, None

# Verify face detector weight file
if not os.path.exists(MODEL_PATH):
    print("Downloading face detector model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initialize MediaPipe Face Detector
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    min_detection_confidence=0.5
)
detector = FaceDetector.create_from_options(options)

def identify_face(input_img, threshold=0.40, min_det_conf=0.50):
    if input_img is None:
        return "❌ Error: Please provide an image to identify.", None, {}
        
    if not has_fine_tuned:
        return "⚠️ Error: Fine-tuned model was not loaded. Run train.py to train it.", None, {}
        
    h, w = input_img.shape[:2]
    
    # MediaPipe expects RGB (which Gradio provides)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_img)
    detection_result = detector.detect(mp_image)
    
    if not detection_result.detections:
        return "❌ No face detected in the image. Please align your face or adjust face detection confidence.", None, {}
        
    bbox = detection_result.detections[0].bounding_box
    x1 = max(0, int(bbox.origin_x))
    y1 = max(0, int(bbox.origin_y))
    x2 = min(w, int(bbox.origin_x + bbox.width))
    y2 = min(h, int(bbox.origin_y + bbox.height))
    
    face_crop = input_img[y1:y2, x1:x2]
    if face_crop.size == 0:
        return "❌ Error: Failed to crop the detected face box.", None, {}
        
    # Preprocess face to match Facenet input layer requirements
    img_resized = cv2.resize(face_crop, (160, 160))
    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # 1. Extract query face embedding from fine-tuned Facenet representation layer
    query_emb = embedding_model.predict(img_batch, verbose=0)[0]
    
    # 2. Get direct softmax prediction probabilities
    preds = model.predict(img_batch, verbose=0)[0]
    pred_idx = np.argmax(preds)
    softmax_confidence = preds[pred_idx]
    
    # 3. Calculate Cosine Distance to all database embeddings
    # Cosine distance between normalized unit vectors is 1.0 - dot product
    distances = 1.0 - np.dot(db_embeddings, query_emb)
    
    min_idx = np.argmin(distances)
    distance = distances[min_idx]
    matched_file_path = db_identities[min_idx]
    matched_class = db_classes[min_idx]
    
    similarity = max(0.0, 1.0 - distance) * 100
    display_name = matched_class.replace('_', ' ').title()
    
    # Format labels output for Gradio classification graph
    label_probs = {class_names[i].replace('_', ' ').title(): float(preds[i]) for i in range(len(class_names))}
    
    if distance <= threshold:
        status_msg = f"✅ Match Found: {display_name}\n\n" \
                     f"• Cosine Similarity: {similarity:.1f}%\n" \
                     f"• Cosine Distance: {distance:.3f}\n" \
                     f"• Softmax Confidence: {softmax_confidence*100:.1f}%"
                     
        # Convert matched BGR image to RGB for Gradio display
        matched_bgr = cv2.imread(matched_file_path)
        matched_rgb = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)
        return status_msg, matched_rgb, label_probs
    else:
        status_msg = f"⚠️ Person Unknown\n\n" \
                     f"• Closest Candidate: {display_name} (Similarity: {similarity:.1f}%)\n" \
                     f"• Cutoff limit is {(1.0 - threshold)*100:.1f}% similarity."
        return status_msg, None, label_probs

# Custom Gradio Blocks layout
with gr.Blocks(title="Premium Face Recognition Hub", theme=gr.themes.Default(primary_hue="purple", secondary_hue="indigo")) as demo:
    gr.Markdown("""
    # 👤 Premium Face Recognition Hub
    Identify individuals in real-time using a custom **fine-tuned Facenet representation model** (100% validation accuracy). Supports webcam live-stream captures and image file uploads.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Input Frame")
            # Combined upload and webcam interface
            input_image = gr.Image(sources=["upload", "webcam"], type="numpy", label="Upload Photo or Capture Webcam")
            
            with gr.Accordion("⚙️ Threshold Settings", open=False):
                threshold_slider = gr.Slider(minimum=0.10, maximum=1.00, value=0.40, step=0.05, label="Cosine Distance Threshold")
                min_det_slider = gr.Slider(minimum=0.10, maximum=1.00, value=0.50, step=0.05, label="Min Face Detection Confidence")
                
            submit_btn = gr.Button("Identify Face", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Search Output")
            output_text = gr.Textbox(label="Identification Status & Metrics", interactive=False)
            output_image = gr.Image(label="Database Match Comparison")
            output_labels = gr.Label(num_top_classes=3, label="Top Match Confidences")
            
    submit_btn.click(
        fn=identify_face,
        inputs=[input_image, threshold_slider, min_det_slider],
        outputs=[output_text, output_image, output_labels]
    )

if __name__ == '__main__':
    # Launch Gradio server on standard port 7860 (Hugging Face default)
    demo.launch(server_name="0.0.0.0", server_port=7860)
