# 👤 Premium Face Recognition Hub

This project is a high-performance face recognition application designed to detect and identify 10 specific people using a custom fine-tuned **Facenet** deep learning model. It features a polished, dark-themed glassmorphic Streamlit interface supporting both local image upload and real-time live webcam capture.

## 🚀 Getting Started

### 1. Environment Setup

It is highly recommended to run this project in the preconfigured `ml` conda environment:

```bash
# Activate the ml conda environment
conda activate ml

# Verify python path
which python3
# Should be: /Users/zach/opt/miniconda3/envs/ml/bin/python3
```

All required dependencies (`streamlit`, `tensorflow`, `deepface`, `scikit-learn`, `opencv-python`, etc.) are pre-installed in this environment.

---

### 2. Fine-Tuning the Model

We have created a training script (`train.py`) that loads the aligned face images from `dataset_cleaned/`, applies data augmentation (rotation, zoom, horizontal flipping) to prevent overfitting, and fine-tunes the top layers of the Facenet model on our 10 target classes.

Run the training script:
```bash
python3 train.py
```

*This will run for 25 epochs and save the fine-tuned model weights to `fine_tuned_facenet.keras` and class mappings to `class_names.npy`.*

---

### 3. Running the Web App

Launch the premium Streamlit web application:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. You can select either:
- **📤 Upload Photo**: Select any local image (PNG, JPG, JPEG, WEBP).
- **📸 Live Camera Input**: Capture a photo directly from your webcam.

The app will automatically crop the face using MediaPipe, project it into the custom fine-tuned embedding space, calculate the cosine distance to all database images, and display:
- **Match Status**: Green/red indicator banners.
- **Match Identity & Compare**: Side-by-side visualization of input vs. best database match.
- **Cosine Distance & Similarity Percentage**: Confidence metrics.
- **Classification Confidence**: Softmax probability from the fine-tuned dense classification head.

---

## ☁️ Deployment Guidelines

### Option A: Streamlit Community Cloud (Recommended)
Streamlit Cloud offers free hosting for public GitHub repositories.
1. Push this workspace folder to a new repository on **GitHub**.
2. Visit [Streamlit Share](https://share.streamlit.io/) and log in with GitHub.
3. Click **New App**, select your repository, branch (`main`), and set the Main file path to `app.py`.
4. Under **Advanced Settings**, ensure you select Python 3.12 or 3.13.
5. Click **Deploy**. Streamlit Cloud will automatically read `requirements.txt` and set up the server.

### Option B: Hugging Face Spaces
Hugging Face Spaces supports Streamlit SDK applications natively.
1. Log in to [Hugging Face](https://huggingface.co/) and click **New Space**.
2. Select **Streamlit** as the Space SDK and choose the Free hardware tier.
3. Clone the Space's Git repository locally or upload files directly via the browser interface.
4. Upload:
   - `app.py`
   - `requirements.txt`
   - `fine_tuned_facenet.keras`
   - `class_names.npy`
   - `dataset_cleaned/` (the cropped face database images)
   - `blaze_face_short_range.tflite` (face detector weights)
5. Hugging Face will automatically build and serve the app at your Space's URL.
