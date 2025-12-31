import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from ultralytics import YOLO

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion",
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis"
]

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_classifier():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load("best_efficientnet_b0.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_detector():
    return YOLO("yolov8s.pt")   # 🔥 your trained YOLOv8 model

classifier = load_classifier()
detector = load_detector()

# ----------------------------
# TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="CliniScan", layout="centered")
st.title("🫁 CliniScan – Chest X-ray Analysis")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["png", "jpg", "jpeg"]
)


# MAIN LOGIC — EVERYTHING INSIDE THIS BLOCK
if uploaded_file is not None and uploaded_file != "":
    # -------- Image --------
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    
    # -------- Classification --------
input_tensor = transform(image).unsqueeze(0).to(DEVICE)
with torch.no_grad():
        logits = classifier(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

with torch.no_grad():
    logits = classifier(input_tensor)
    st.subheader("🧠 Top Predicted Abnormalities")

    top_indices = probs.argsort()[-3:][::-1]
    for idx in top_indices:
        st.write(f"🔴 {CLASS_NAMES[idx]} — {probs[idx]:.2f}")

   #confidence chart bar-----

    df = pd.DataFrame({
    "Disease": [CLASS_NAMES[i] for i in top_indices],
    "Confidence": [probs[i] for i in top_indices]
})

st.bar_chart(df.set_index("Disease"))
# download report-----
report = f"""
Chest X-ray Analysis Report

Top Predictions:
{CLASS_NAMES[top_indices[0]]}: {probs[top_indices[0]]:.2f}
{CLASS_NAMES[top_indices[1]]}: {probs[top_indices[1]]:.2f}
{CLASS_NAMES[top_indices[2]]}: {probs[top_indices[2]]:.2f}
"""

st.download_button(
    "📄 Download Report",
    report,
    file_name="cliniscan_report.txt"
)



 # -------- Multi-label Abnormality Output --------

    # -------- Detection --------
st.subheader("📦 Region Localization (YOLO)")
show_boxes = st.checkbox("Show abnormal regions", value=True)

if show_boxes:
        st.subheader("📦 Disease Region Localization")

        results = detector.predict(
            source=np.array(image),
            conf=0.5,
            iou=0.5,
            verbose=False
        )

        annotated = results[0].plot(labels=False, conf=False)
        st.image(
            annotated,
            caption="Highlighted Suspicious Regions",
            use_container_width=True
        )
"""results = detector.predict(
        source=np.array(image),
        conf=0.5,
        iou=0.5,
        verbose=False
    )

annotated = results[0].plot(labels=False, conf=False)
st.image(annotated, caption="Detected Regions", use_container_width=True)"""

st.caption(
    "Detected regions indicate abnormal areas. "
    "Disease predictions are based on the full image."
)

