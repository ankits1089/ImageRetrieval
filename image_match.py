import os
import numpy as np
from tensorflow import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import random

# Initialize the pre-trained ResNet50 model (excluding the top layer)
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)

# Load the previously saved features
image_features = np.load("image_features.npy", allow_pickle=True).item()  # Load dictionary

# Function to process a single image and extract features
def extract_features(image_path, model):
    try:
        # Load the image
        img = load_img(image_path, target_size=(224, 224))  # Resize to model's input size
        img_array = img_to_array(img)  # Convert image to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess for ResNet50
        
        # Extract features
        features = model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
    
def find_best_match(new_image_path, image_features, model):
    # Extract features for the new image
    new_image_features = extract_features(new_image_path, model)
    
    # Calculate cosine similarity with all stored features
    best_match = None
    best_similarity = -1
    for image_name, features in image_features.items():
        similarity = cosine_similarity(
            [new_image_features], [features]
        )[0][0]  # Cosine similarity between vectors
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = image_name

    return best_match, best_similarity

# Path to the new image
new_image_path = "/Users/abhinav/Documents/Ankit/MachineLearning/Learning/ImageRetrieval/chatgpt_generated_image.webp"

# Find the best match
best_match, similarity = find_best_match(new_image_path, image_features, model)

# Output the result
print(f"Best match: {best_match} with similarity: {similarity:.2f}")
