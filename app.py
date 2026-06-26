import streamlit as st
import cv2
import os
import sys
import urllib.request
import numpy as np
import mediapipe as mp
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Premium Face Recognition Hub",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS (Dark Theme, Glassmorphism, Outfit Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0e1118 0%, #151922 100%);
        color: #f1f3f9;
    }
    
    /* Glassmorphic Container Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
        transition: all 0.3s ease-in-out;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.12);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.55);
    }
    
    /* Typography & Headers */
    .title-text {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle-text {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 25px;
    }
    
    /* Result Banners */
    .status-banner {
        border-radius: 10px;
        padding: 12px 20px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 15px;
        border-left: 5px solid;
    }
    
    .match-success {
        border-color: #10b981;
        background: rgba(16, 185, 129, 0.1);
        color: #34d399;
    }
    
    .match-unknown {
        border-color: #ef4444;
        background: rgba(239, 68, 68, 0.1);
        color: #f87171;
    }
    
    /* Metric blocks */
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="title-text">👤 Premium Face Recognition Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Identify faces in real-time using custom fine-tuned deep representations. Supports live camera inputs and photo uploads.</div>', unsafe_allow_html=True)

# Sidebar Configurations
st.sidebar.markdown("### ⚙️ System Settings")
threshold = st.sidebar.slider(
    "Cosine Distance Threshold", 
    min_value=0.10, max_value=1.00, value=0.40, step=0.05,
    help="Cosine distance cut-off. Lower values yield stricter matching (fewer false positives)."
)
min_confidence = st.sidebar.slider(
    "Min Face Detection Confidence", 
    min_value=0.10, max_value=1.00, value=0.50, step=0.05,
    help="Confidence threshold for the MediaPipe face detector."
)

# Constants
INPUT_DIR  = 'dataset'
DATA_DIR   = 'dataset_cleaned'
IMG_SIZE   = (224, 224)
MODEL_PATH = os.path.join(os.getcwd(), 'blaze_face_short_range.tflite')
MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'

# Verify face detector model is locally cached
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading face detector model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initialize MediaPipe Face Detector
@st.cache_resource
def get_face_detector(model_path, confidence):
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        min_detection_confidence=confidence
    )
    return FaceDetector.create_from_options(options)

# Helper function to detect and crop face
def crop_face(img_np, detector):
    h, w = img_np.shape[:2]
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)
    
    if detection_result.detections:
        bbox = detection_result.detections[0].bounding_box
        x1 = max(0, int(bbox.origin_x))
        y1 = max(0, int(bbox.origin_y))
        x2 = min(w, int(bbox.origin_x + bbox.width))
        y2 = min(h, int(bbox.origin_y + bbox.height))
        
        face_crop = img_np[y1:y2, x1:x2]
        if face_crop.size > 0:
            face_resized = cv2.resize(face_crop, IMG_SIZE)
            return face_resized, (x1, y1, x2, y2)
            
    return None, None

# Environment check for fine-tuned weights
FINE_TUNED_MODEL_PATH = 'fine_tuned_facenet.keras'
CLASS_NAMES_PATH = 'class_names.npy'
has_fine_tuned = os.path.exists(FINE_TUNED_MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH)

# Load the models
@st.cache_resource
def load_models_and_cache_embeddings(model_path, class_names_path, data_dir):
    # Set thread environment variables before importing TF
    os.environ['XNNPACK_DISABLE'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    
    # Load fine-tuned Keras model
    model = tf.keras.models.load_model(model_path)
    # The second-to-last layer of the model (index -3, before dropout and dense classification_head) 
    # outputs the 128-dimensional L2-normalized face embedding.
    embedding_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-3].output)
    class_names = np.load(class_names_path, allow_pickle=True)
    
    # Build database embeddings cache in-memory
    db_embeddings = []
    db_identities = []
    db_classes = []
    
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            continue
        img_names = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        for img_name in img_names:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (160, 160)) # Facenet expected size
            img_normalized = img_resized.astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Predict embedding
            emb = embedding_model.predict(img_batch, verbose=0)[0]
            db_embeddings.append(emb)
            db_identities.append(img_path)
            db_classes.append(class_name)
            
    return model, embedding_model, class_names, np.array(db_embeddings), db_identities, db_classes

# UI Sidebar status info
if has_fine_tuned:
    st.sidebar.success("🎯 Custom Fine-Tuned Model Active")
    with st.spinner("Loading models and caching database..."):
        model, embedding_model, class_names, db_embeddings, db_identities, db_classes = load_models_and_cache_embeddings(
            FINE_TUNED_MODEL_PATH, CLASS_NAMES_PATH, DATA_DIR
        )
else:
    st.sidebar.warning("⚠️ Using Pre-trained Facenet weights (Run train.py to activate fine-tuning)")
    from deepface import DeepFace

# Selector for Input Source
source_type = st.radio("Choose Input Source:", ["📤 Upload Photo", "📸 Live Camera Input"])

uploaded_file = None
if source_type == "📤 Upload Photo":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "webp"])
else:
    uploaded_file = st.camera_input("Capture live photo")

if uploaded_file is not None:
    # Read the file to OpenCV BGR image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Split view columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📸 Captured Frame")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🔍 Analysis & Match Result")
        
        # Run MediaPipe face detection & crop
        try:
            detector = get_face_detector(MODEL_PATH, min_confidence)
            cropped_face, bbox = crop_face(image, detector)
            
            if cropped_face is not None:
                # Show detected cropped face
                cropped_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                st.image(cropped_rgb, caption="Cropped Face (Target Model Input)", width=150)
                
                # Perform Face Identification
                with st.spinner("Performing search..."):
                    if has_fine_tuned:
                        # 1. Image preprocessing
                        img_resized = cv2.resize(cropped_rgb, (160, 160))
                        img_normalized = img_resized.astype('float32') / 255.0
                        img_batch = np.expand_dims(img_normalized, axis=0)
                        
                        # 2. Extract embedding from custom model
                        query_emb = embedding_model.predict(img_batch, verbose=0)[0]
                        
                        # 3. Softmax prediction for classification confidence
                        preds = model.predict(img_batch, verbose=0)[0]
                        pred_idx = np.argmax(preds)
                        softmax_confidence = preds[pred_idx] * 100
                        
                        # 4. Calculate Cosine Distance to all database embeddings
                        # Since embeddings are L2 normalized, distance is 1.0 - dot_product
                        distances = 1.0 - np.dot(db_embeddings, query_emb)
                        
                        min_idx = np.argmin(distances)
                        distance = distances[min_idx]
                        matched_file_path = db_identities[min_idx]
                        person_name = db_classes[min_idx]
                        
                        # 5. Get top matches sorted by distance
                        sorted_indices = np.argsort(distances)
                        top_matches = []
                        for idx in sorted_indices[:3]:
                            top_matches.append({
                                'name': db_classes[idx],
                                'distance': distances[idx],
                                'identity': db_identities[idx]
                            })
                            
                    else:
                        # Fallback using standard DeepFace matching logic
                        temp_crop_path = "temp_crop.jpg"
                        cv2.imwrite(temp_crop_path, cropped_face)
                        try:
                            sys.stdout = open(os.devnull, 'w')
                            sys.stderr = open(os.devnull, 'w')
                            results = DeepFace.find(
                                img_path=temp_crop_path,
                                db_path=DATA_DIR,
                                model_name='Facenet',
                                distance_metric='cosine',
                                enforce_detection=False,
                                silent=True
                            )
                            sys.stdout = sys.__stdout__
                            sys.stderr = sys.__stderr__
                            df = results[0] if results else None
                        except Exception as dfe:
                            sys.stdout = sys.__stdout__
                            sys.stderr = sys.__stderr__
                            st.error(f"DeepFace processing error: {dfe}")
                            df = None
                            
                        if os.path.exists(temp_crop_path):
                            os.remove(temp_crop_path)
                            
                        if df is not None and not df.empty:
                            df = df.sort_values('distance').reset_index(drop=True)
                            best_match = df.iloc[0]
                            matched_file_path = best_match['identity']
                            person_name = os.path.basename(os.path.dirname(matched_file_path))
                            distance = best_match['distance']
                            softmax_confidence = None
                            
                            # Construct top matches formatted
                            top_matches = []
                            for idx, row in df.head(3).iterrows():
                                top_matches.append({
                                    'name': os.path.basename(os.path.dirname(row['identity'])),
                                    'distance': row['distance'],
                                    'identity': row['identity']
                                })
                        else:
                            person_name = None
                            distance = 1.0
                            top_matches = []
                
                # Render Match Status Banner
                similarity = max(0.0, 1.0 - distance) * 100
                display_name = person_name.replace('_', ' ').title() if person_name else "Unknown"
                
                if person_name and distance <= threshold:
                    st.markdown(f'<div class="status-banner match-success">✅ Match Found: {display_name}</div>', unsafe_allow_html=True)
                    
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.markdown(f'<div class="metric-value">{similarity:.1f}%</div><div class="metric-label">Cosine Similarity</div>', unsafe_allow_html=True)
                    with col_m2:
                        if has_fine_tuned:
                            st.markdown(f'<div class="metric-value">{softmax_confidence:.1f}%</div><div class="metric-label">Classification Confidence</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="metric-value">{distance:.3f}</div><div class="metric-label">Cosine Distance</div>', unsafe_allow_html=True)
                            
                    st.progress(similarity / 100.0)
                    
                    # Side by Side comparison
                    st.markdown("---")
                    st.markdown("**Image Comparison:**")
                    comp1, comp2 = st.columns(2)
                    with comp1:
                        st.caption("Input Face")
                        st.image(cropped_rgb, use_container_width=True)
                    with comp2:
                        st.caption("Matched Database Image")
                        if os.path.exists(matched_file_path):
                            st.image(Image.open(matched_file_path), use_container_width=True)
                        else:
                            st.warning("Database source image not found on disk.")
                            
                    # Show Top 3 database matches
                    st.markdown("---")
                    st.write("📊 **Database Matching Breakdown:**")
                    seen = set()
                    for item in top_matches:
                        match_disp = item['name'].replace('_', ' ').title()
                        match_dist = item['distance']
                        match_sim = max(0.0, 1.0 - match_dist) * 100
                        st.write(f"- **{match_disp}**: `{match_sim:.1f}%` similarity (dist: `{match_dist:.3f}`)")
                        
                else:
                    st.markdown('<div class="status-banner match-unknown">⚠️ Person Unknown</div>', unsafe_allow_html=True)
                    if person_name:
                        st.info(f"Closest candidate is **{display_name}** but similarity score (`{similarity:.1f}%`) is below the threshold limit (`{(1.0-threshold)*100:.1f}%`).")
                        with st.expander("View closest match details"):
                            if os.path.exists(matched_file_path):
                                st.image(Image.open(matched_file_path), caption=f"Closest: {display_name}", width=150)
                    else:
                        st.info("No candidates match the input face within the search directory.")
            else:
                st.error("❌ No face detected in the image. Ensure the camera is aligned and adjust 'Min Face Detection Confidence' if needed.")
                
        except Exception as e:
            st.error(f"Processing error: {e}")
            st.info("Ensure the dataset folders and trained Keras model are generated correctly.")
            
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("👈 Please capture a live photo or upload an image file using the options above to begin.")
