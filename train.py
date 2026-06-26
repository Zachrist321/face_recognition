import os
# Set environment variables to prevent Apple Silicon thread locks in headless subprocesses
os.environ['XNNPACK_DISABLE'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from deepface import DeepFace
from sklearn.model_selection import train_test_split

# Constants
DATA_DIR = 'dataset_cleaned'
IMG_SIZE = (160, 160)  # Facenet's expected input shape
MODEL_SAVE_PATH = 'fine_tuned_facenet.keras'

def load_dataset(data_dir):
    images = []
    labels = []
    
    # Get sorted class names (folders)
    class_names = sorted([
        d for d in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    print(f"Detected {len(class_names)} classes: {class_names}")
    
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        img_names = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        for img_name in img_names:
            img_path = os.path.join(class_path, img_name)
            try:
                # Read image in BGR (OpenCV default)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                # Convert BGR to RGB (DeepFace/Keras default)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize to Facenet input size (160, 160)
                img_resized = cv2.resize(img_rgb, IMG_SIZE)
                # Normalize pixel values to [0, 1] (DeepFace base normalization)
                img_normalized = img_resized.astype('float32') / 255.0
                
                images.append(img_normalized)
                labels.append(class_to_idx[class_name])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                
    return np.array(images), np.array(labels), class_names

def build_fine_tuned_model(num_classes):
    print("Building model based on Facenet...")
    # Load Facenet base model using DeepFace
    base_model = DeepFace.build_model("Facenet").model
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze all layers except the last 15 layers for fine-tuning
    print(f"Base model has {len(base_model.layers)} layers.")
    for layer in base_model.layers[:-15]:
        layer.trainable = False
        
    # Input is base model input (160, 160, 3)
    inputs = base_model.input
    embeddings = base_model.output # Shape: (None, 128)
    
    # Classification head
    x = layers.Dropout(0.3)(embeddings)
    outputs = layers.Dense(num_classes, activation='softmax', name='classification_head')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} directory not found. Please run the notebook pre-processing first.")
        return
        
    X, y, class_names = load_dataset(DATA_DIR)
    
    if len(X) == 0:
        print("Error: No images loaded. Check your dataset_cleaned directory.")
        return
        
    print(f"Loaded {len(X)} images belonging to {len(class_names)} classes.")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
    
    # Data Augmentation layer to prevent overfitting
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    
    # Build model
    model = build_fine_tuned_model(len(class_names))
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Augment the training data
    print("Preparing training dataset...")
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(len(X_train)).batch(16)
    train_dataset = train_dataset.map(lambda x_batch, y_batch: (data_augmentation(x_batch, training=True), y_batch))
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(16)
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=25,
        verbose=1
    )
    
    # Evaluate final validation accuracy
    val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
    
    # Save the fine-tuned model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved successfully to {MODEL_SAVE_PATH}")
    
    # Save class names mapping for the Streamlit app to load
    np.save('class_names.npy', class_names)
    print("Saved class_names.npy mapping.")

if __name__ == '__main__':
    main()
