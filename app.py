import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black
from reportlab.lib.utils import ImageReader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

IMG_SIZE = (200, 200)
os.makedirs(REPORT_DIR, exist_ok=True)

TUMOR_MODEL_PATH = os.path.join(MODEL_DIR, "tumor_detection_model.h5")
MRI_MODEL_PATH = os.path.join(MODEL_DIR, "mri_analysis_model.h5")


if not os.path.exists(TUMOR_MODEL_PATH):
    raise FileNotFoundError(f"Missing model: {TUMOR_MODEL_PATH}")

if not os.path.exists(MRI_MODEL_PATH):
    raise FileNotFoundError(f"Missing model: {MRI_MODEL_PATH}")


tumor_model = tf.keras.models.load_model(TUMOR_MODEL_PATH, compile=False)
mri_model = tf.keras.models.load_model(MRI_MODEL_PATH, compile=False)


def preprocess(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def generate_pdf(
    name, pid, age, notes,
    tumor_label, tumor_prob,
    mri_label, mri_prob,
    image
):
    pdf_path = os.path.join(REPORT_DIR, f"{pid}_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

   
    c.setStrokeColor(black)
    c.rect(30, 30, width - 60, height - 60, fill=0)

   
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 60, "MRI Brain Tumor Analysis Report")

   
    left_x = 50
    text_y = height - 120

    c.setFont("Helvetica", 12)
    c.drawString(left_x, text_y, f"Patient Name: {name}")
    text_y -= 20
    c.drawString(left_x, text_y, f"Patient ID: {pid}")
    text_y -= 20
    c.drawString(left_x, text_y, f"Age: {age}")

    # ---------------- RIGHT FIXED IMAGE BLOCK
    img_width = 180
    img_height = 180
    img_x = width - img_width - 60
    img_y = height - img_height - 140  # fixed safe zone

    img_reader = ImageReader(image)
    c.drawImage(
        img_reader,
        img_x,
        img_y,
        width=img_width,
        height=img_height,
        preserveAspectRatio=True,
        mask="auto"
    )

   
    results_y = img_y - 40

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_x, results_y, "Results")

    results_y -= 25
    c.setFont("Helvetica", 12)
    c.drawString(
        left_x,
        results_y,
        f"Tumor Detection: {tumor_label} (Confidence: {tumor_prob*100:.2f}%)"
    )

    results_y -= 25
    c.drawString(
        left_x,
        results_y,
        f"MRI Pattern Analysis: {mri_label} (Confidence: {mri_prob*100:.2f}%)"
    )

    # Notes
    results_y -= 40
    c.drawString(left_x, results_y, f"Notes: {notes}")

    # Footer (always bottom safe area)
    c.setFont("Helvetica", 10)
    c.drawString(
        left_x,
        100,
        f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    c.drawString(
        left_x,
        85,
        "Disclaimer: This report is AI-generated and is not a medical diagnosis."
    )

    c.save()
    return pdf_path


def predict(image, name, patient_id, age, notes):
    if image is None:
        return "Upload MRI image", "N/A", None

    img = preprocess(image)

    tumor_prob = float(tumor_model.predict(img)[0][0])
    mri_prob = float(mri_model.predict(img)[0][0])

    tumor_label = "Tumor Detected" if tumor_prob >= 0.5 else "No Tumor Detected"
    mri_label = "Pattern Suggests Tumor" if mri_prob >= 0.5 else "Normal Pattern"

    pdf_path = generate_pdf(
        name, patient_id, age, notes,
        tumor_label, tumor_prob,
        mri_label, mri_prob,
        image
    )

    return (
        f"{tumor_label} ({tumor_prob*100:.2f}%)",
        f"{mri_label} ({mri_prob*100:.2f}%)",
        pdf_path
    )


app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload MRI Image"),
        gr.Textbox(label="Patient Name"),
        gr.Textbox(label="Patient ID"),
        gr.Textbox(label="Age"),
        gr.Textbox(label="Notes")
    ],
    outputs=[
        gr.Textbox(label="Tumor Detection Result"),
        gr.Textbox(label="MRI Pattern Analysis"),
        gr.File(label="Download Report (PDF)")
    ],
    title="🧠 Brain Tumor Detection System",
    description="Upload MRI → AI Analysis → Download Medical Report"
)

if __name__ == "__main__":
    app.launch()
