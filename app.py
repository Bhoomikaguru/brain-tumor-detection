import gradio as gr

def predict(image):
    return "Model loaded successfully ✅"

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Brain Tumour Detection")
    img = gr.Image(type="pil", label="Upload MRI Image")
    out = gr.Textbox(label="Prediction")
    btn = gr.Button("Predict")

    btn.click(
        fn=predict,
        inputs=img,
        outputs=out
    )

# IMPORTANT: HF Spaces needs server_name + no share
demo.launch(server_name="0.0.0.0", server_port=7860)
