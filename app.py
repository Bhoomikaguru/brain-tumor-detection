import gradio as gr
from PIL import Image
import numpy as np

def predict(image):
    if image is None:
        return "Please upload an image."

    return "✅ App is running correctly. Model integration pending."

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🧠 Brain Tumor Detection",
    description="Deployment sanity check. Model will be added next."
)

if __name__ == "__main__":
    demo.launch()
