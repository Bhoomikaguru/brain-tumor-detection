import os
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

MODEL_PATH = "models/tumor_detection_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224
CLASS_NAMES = ["No Tumor", "Tumor Detected"]

def predict(image):
    if image is None:
        return "Please upload an image"

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = CLASS_NAMES[int(prediction > 0.5)]
    confidence = prediction if label == "Tumor Detected" else 1 - prediction

    return f"{label} (Confidence: {confidence:.2%})"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Brain Tumor Detection",
    description="Upload a brain MRI image to detect tumor presence."
)

demo.launch(server_name="0.0.0.0", server_port=7860)
