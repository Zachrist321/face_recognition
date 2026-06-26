# 👤 Premium Face Recognition Hub (Gradio Edition)

This project is a high-performance face recognition application designed to detect and identify 10 specific people using a custom fine-tuned **Facenet** deep learning model. It features a modern Gradio web interface supporting both local image upload and real-time webcam captures.

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

All required dependencies (`gradio`, `tensorflow`, `deepface`, `scikit-learn`, `opencv-python`, etc.) are pre-installed in this environment.

---

### 2. Fine-Tuning the Model

We have created a training script (`train.py`) that loads the aligned face images from `dataset_cleaned/`, applies data augmentation (rotation, zoom, horizontal flipping) to prevent overfitting, and fine-tunes the top layers of the Facenet model on our 10 target classes.

Run the training script:
```bash
python3 train.py
```

*This will run for 25 epochs and save the fine-tuned model weights to `fine_tuned_facenet.keras` and class mappings to `class_names.npy`.*

---

### 3. Running the Web App locally

Launch the Gradio web application:
```bash
python3 app.py
```

Open `http://localhost:7860` in your browser. You can:
- **📤 Upload Photo**: Drag and drop any image file (PNG, JPG, JPEG, WEBP).
- **📸 Live Camera Capture**: Capture a photo directly from your webcam.

The app will crop the face using MediaPipe, project it into the custom fine-tuned embedding space, calculate the cosine distance to all database images, and display:
- **Match Status**: Green success check vs. red unknown warning.
- **Match Visual Comparison**: Displays the matched database image.
- **Top Match Confidences**: A clean bar chart plotting the softmax probabilities.

---

## ☁️ Deployment to Hugging Face Spaces

Hugging Face Spaces supports Gradio SDK applications natively.

### Step 1: Create a Space on Hugging Face
1. Log in to [Hugging Face](https://huggingface.co/).
2. Click on **Spaces** in the top navigation bar, then click **Create new Space**.
3. Set the configuration details:
   - **Space name**: e.g., `face-recognition-hub`
   - **License**: e.g., `mit` or `apache-2.0`
   - **SDK**: Select **Gradio** (this is the key!).
   - **Hardware**: Choose the default **CPU Basic (Free)**.
   - **Visibility**: Public or Private.
4. Click **Create Space**.

---

### Step 2: Upload Files
Hugging Face will give you a Git clone URL. You can push your files using Git, or upload them directly via their web browser UI by clicking the **Files** tab and choosing **Upload files**.

You need to upload:
*   `app.py` (Hugging Face looks for `app.py` as the default entry point)
*   `requirements.txt` (the dependencies list)
*   `packages.txt` (tells Hugging Face to install `libgl1` and `libglib2.0-0` system libraries)
*   `fine_tuned_facenet.keras` & `class_names.npy` (your fine-tuned model and classes)
*   `dataset_cleaned/` (the entire folder containing your target face images)
*   `blaze_face_short_range.tflite` (the MediaPipe face detector weights)

---

### Step 3: Run
Once uploaded, Hugging Face will automatically build the container. Once the build completes, your app will be live and ready for testing!
