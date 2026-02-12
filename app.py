import gradio as gr

def predict(image):
    return "Model loaded successfully ✅"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="🧠 Brain Tumour Detection",
    description="Upload an MRI image to get prediction"
)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
