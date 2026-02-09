import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ===============================
# Load model from Hugging Face Hub
# ===============================

MODEL_URL = "https://huggingface.co/Bhoomi1104/brain-tumor-model/resolve/main/tumor_detection_model.h5"

model_path = tf.keras.utils.get_file(
    "tumor_detection_model.h5",
    MODEL_URL
)

tumor_model = tf.keras.models.load_model(model_path)




def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))  # MUST match training
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def generate_pdf(name, pid, diagnosis, image):
    os.makedirs("reports", exist_ok=True)
    pdf_path = f"reports/report_{pid}.pdf"

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Brain Tumor Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Patient Name: {name}")
    c.drawString(50, height - 120, f"Patient ID: {pid}")
    c.drawString(50, height - 140, f"Diagnosis: {diagnosis}")

    img_reader = ImageReader(image)
    c.drawImage(img_reader, 50, height - 450, width=300, height=300)

    c.showPage()
    c.save()

    return pdf_path

def save_to_csv(name, pid, diagnosis):
    row = {
        "Name": name,
        "Patient ID": pid,
        "Diagnosis": diagnosis,
        "Timestamp": datetime.datetime.now()
    }

    df = pd.DataFrame([row])

    if not os.path.exists("results.csv"):
        df.to_csv("results.csv", index=False)
    else:
        df.to_csv("results.csv", mode="a", index=False, header=False)


# =====================================================================
# DATABASE INITIALIZATION
# =====================================================================
def init_database():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            patient_id TEXT,
            patient_age TEXT,
            notes TEXT,
            tumor_result TEXT,
            mri_result TEXT,
            timestamp TEXT,
            image_path TEXT
        )
    """)
    conn.commit()
    conn.close()


init_database()


# =====================================================================
# SAVE RESULTS TO TEXT
# =====================================================================
def save_text_report(filename, patient_name, patient_id, age, notes, tumor_result, mri_result):
    with open(filename, "w") as f:
        f.write(f"--- Patient MRI Report ---\n")
        f.write(f"Name: {patient_name}\n")
        f.write(f"ID: {patient_id}\n")
        f.write(f"Age: {age}\n")
        f.write(f"Notes: {notes}\n\n")
        f.write(f"Tumor Detection: {tumor_result}\n")
        f.write(f"MRI Pattern Analysis: {mri_result}\n")
        f.write(f"Generated on: {datetime.now()}\n")


# =====================================================================
# SAVE RESULTS TO PDF WITH IMAGE
# =====================================================================
def save_pdf_report(pdf_path, image_path, patient_name, patient_id, age, notes,
                    tumor_result, mri_result):

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "MRI Brain Tumor Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Patient Name: {patient_name}")
    c.drawString(50, height - 120, f"Patient ID: {patient_id}")
    c.drawString(50, height - 140, f"Age: {age}")

    c.drawString(50, height - 180, f"Tumor Detection Result: {tumor_result}")
    c.drawString(50, height - 200, f"MRI Pattern Analysis: {mri_result}")
    c.drawString(50, height - 220, f"Notes: {notes}")

    c.drawString(50, height - 260, f"Generated On: {datetime.now()}")

    # Insert MRI Image
    if image_path and os.path.exists(image_path):
        img = ImageReader(image_path)
        c.drawImage(img, 50, height - 550, width=300, height=250)

    c.save()


# =====================================================================
# SAVE RESULTS TO CSV
# =====================================================================
def log_to_csv(csv_path, data_dict):
    df = pd.DataFrame([data_dict])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False, mode='a', header=False)

def predict(image, name, patient_id, email):
    if image is None:
        return "Please upload an MRI image", None

    img = preprocess_image(image)
    pred = tumor_model.predict(img)[0][0]

    diagnosis = "Tumor Detected" if pred > 0.5 else "No Tumor Detected"

    pdf_path = generate_pdf(name, patient_id, diagnosis, image)
    save_to_csv(name, patient_id, diagnosis)

    return diagnosis, pdf_path

# =====================================================================
# SAVE RESULTS TO DATABASE
# =====================================================================
def save_to_database(patient_name, patient_id, age, notes,
                     tumor_result, mri_result, img_path):

    conn = sqlite3.connect("predictions.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO results (patient_name, patient_id, patient_age, notes,
                             tumor_result, mri_result, timestamp, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (patient_name, patient_id, age, notes,
          tumor_result, mri_result, str(datetime.now()), img_path))
    conn.commit()
    conn.close()


# =====================================================================
# MODELS
# =====================================================================
def create_tumor_detection_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(200,200,3)),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def create_mri_analysis_model():
    model = Sequential([
        Flatten(input_shape=(200,200,3)),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# =====================================================================
# PREDICTION FUNCTION WITH REPORT GENERATION
# =====================================================================
def predict(image, patient_name, patient_id, age, notes):

    if image is None:
        return "Upload image first!", None, None

    image_resized = cv2.resize(image, (200,200)) / 255.0
    image_batch = np.expand_dims(image_resized, axis=0)

    tumor_model = load_model("tumor_detection_model.h5")
    mri_model = load_model("mri_analysis_model.h5")

    tumor_pred = float(tumor_model.predict(image_batch)[0][0])
    tumor_label = "Tumor Detected" if tumor_pred > 0.5 else "No Tumor"
    tumor_conf = tumor_pred if tumor_pred > 0.5 else 1 - tumor_pred

    mri_pred = float(mri_model.predict(image_batch)[0][0])
    mri_label = "Pattern Suggests Tumor" if mri_pred > 0.5 else "Normal Pattern"
    mri_conf = mri_pred if mri_pred > 0.5 else 1 - mri_pred

    tumor_result = f"{tumor_label} ({tumor_conf*100:.2f}%)"
    mri_result = f"{mri_label} ({mri_conf*100:.2f}%)"

    # Save MRI image
    image_path = f"reports/{patient_id}_image.jpg"
    os.makedirs("reports", exist_ok=True)
    cv2.imwrite(image_path, image)

    # Save text report
    text_path = f"reports/{patient_id}_report.txt"
    save_text_report(text_path, patient_name, patient_id, age, notes,
                     tumor_result, mri_result)

    # Save PDF report
    pdf_path = f"reports/{patient_id}_report.pdf"
    save_pdf_report(pdf_path, image_path, patient_name, patient_id, age, notes,
                    tumor_result, mri_result)

    # Save CSV log
    csv_data = {
        "patient_name": patient_name,
        "patient_id": patient_id,
        "age": age,
        "notes": notes,
        "tumor_result": tumor_result,
        "mri_result": mri_result,
        "timestamp": str(datetime.now())
    }
    log_to_csv("results.csv", csv_data)

    # Save to database
    save_to_database(patient_name, patient_id, age, notes,
                     tumor_result, mri_result, image_path)

    return (
        f"Tumor Result: {tumor_result}",
        f"MRI Analysis: {mri_result}",
        pdf_path  # downloadable report
    )

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload MRI Image"),
        gr.Textbox(label="Patient Name"),
        gr.Textbox(label="Patient ID"),
        gr.Textbox(label="Email (optional)")
    ],
    outputs=[
        gr.Textbox(label="Diagnosis"),
        gr.File(label="Download PDF Report")
    ],
    title="🧠 Brain Tumor Detection System",
    description="Upload MRI scan → Detect tumor → Download medical report"
)

# =====================================================================
# GUI
# =====================================================================
app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label="Upload MRI Scan"),
        gr.Textbox(label="Patient Name"),
        gr.Textbox(label="Patient ID"),
        gr.Textbox(label="Age"),
        gr.Textbox(label="Notes (Symptoms, Observations)")
    ],
    outputs=[
        gr.Textbox(label="Tumor Detection Result"),
        gr.Textbox(label="MRI Pattern Analysis"),
        gr.File(label="Download PDF Report")
    ],
    title="Brain Tumor Detection System (With Patient Report Generation)"
)


if __name__ == "__main__":
    app.launch()


