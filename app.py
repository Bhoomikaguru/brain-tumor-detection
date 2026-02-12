import gradio as gr
from PIL import Image
import numpy as np

def predict(image):
    if image is None:
        return "❌ Please upload an image."
    return "✅ App is running correctly. Model will be connected next."

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🧠 Brain Tumour Detection",
    description="Deployment sanity check. Model integration comes next."
)

if __name__ == "__main__":
    demo.launch()
