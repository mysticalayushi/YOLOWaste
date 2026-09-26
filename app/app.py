"""
Simple Streamlit demo app for the Smart Waste Detection project.
Upload an image, the trained YOLOv8 model detects and classifies waste items.

Run from project root (A:\\YOLOWaste):
    streamlit run app/app.py
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ---- Page setup ----
st.set_page_config(page_title="Smart Waste Detection", page_icon="♻️", layout="centered")
st.title("♻️ Smart Waste Detection and Classification")
st.write("Upload an image and the model will detect and classify waste items (Cardboard, Metal, Paper, Plastic).")

# ---- Load model (cached so it doesn't reload on every interaction) ----
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

# ---- Sidebar controls ----
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

# ---- File uploader ----
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    # Run inference
    with st.spinner("Detecting..."):
        results = model.predict(source=np.array(image), conf=confidence, save=False)

    result = results[0]
    annotated_img = result.plot()  # returns a numpy array (BGR) with boxes drawn
    annotated_img = annotated_img[:, :, ::-1]  # convert BGR -> RGB for display

    with col2:
        st.subheader("Detected Waste")
        st.image(annotated_img, use_column_width=True)

    # ---- Detection summary ----
    st.subheader("Detection Summary")
    class_names = model.names
    counts = {}
    for cls_id in result.boxes.cls.tolist():
        cls_name = class_names[int(cls_id)]
        counts[cls_name] = counts.get(cls_name, 0) + 1

    if counts:
        for cls_name, count in sorted(counts.items()):
            st.write(f"- **{cls_name}**: {count} detected")
    else:
        st.write("No waste items detected. Try lowering the confidence threshold.")

else:
    st.info("Upload an image above to get started.")