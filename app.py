import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TEMP_DIR"] = "/tmp"


# ===============================
# Load model from Hugging Face Hub
# ===============================
MODEL_URL = "https://huggingface.co/Bhoomi1104/brain-tumor-model/resolve/main/tumor_detection_model.h5"

model_path = tf.keras.utils.get_file(
    "tumor_detection_model.h5",
    MODEL_URL
)

model = tf.keras.models.load_model(model_path)

# ===============================
# Image preprocessing
# ===============================
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ===============================
# Prediction function
# ===============================
def predict(image):
    if image is None:
        return "No image uploaded"

    img = preprocess_image(image)
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "🧠 Tumor Detected"
    else:
        return "✅ No Tumor Detected"

# ===============================
# Gradio Interface
# ===============================
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Brain Tumor Detection",
    description="Upload an MRI scan to detect brain tumor using deep learning."
)

# ===============================
# Launch
# ===============================
if __name__ == "__main__":
    interface.launch()
