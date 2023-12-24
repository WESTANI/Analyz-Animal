# Analyz-Animal
Analyz Animal
# Code for Animal Recognition using TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np

def recognize_animal(image_path):
    # Load the InceptionV3 model pre-trained on ImageNet data
    model = InceptionV3(weights='imagenet')

    # Load and preprocess the input image
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Decode and print the top-3 predicted classes
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score:.2f})")

# Example usage
image_path = "path/to/your/animal_image.jpg"
recognize_animal(image_path)
