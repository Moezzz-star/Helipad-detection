import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Helipad Detector 🚁",
    page_icon="🚁",
    layout="centered"
)

MODEL_PATH = 'helipad_resnet50.pth'

# --- 2. LOAD MODEL FUNCTION ---
@st.cache_resource  # caches the model
def load_model():
    device = torch.device('cpu')  # CPU for web app

    # Recreate the architecture
    model = models.resnet50(weights=None)  # pretrained=False is deprecated
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 1)
    )

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found! Please put it in the same folder.")
        return None

    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()

# --- 3. DEFINE TRANSFORMS ---
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- 4. UI LAYOUT ---
st.title("🚁 Satellite Helipad Detector")
st.write("Upload a satellite image, and the AI (ResNet50) will detect if there is a helipad.")
st.markdown("---")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess image
    input_tensor = transform_pipeline(image).unsqueeze(0)  # Add batch dimension

    # Inference button
    if st.button('Analyze Image 🔍'):
        if model:
            with torch.no_grad():
                output = model(input_tensor)
                probability = torch.sigmoid(output).item()

            st.markdown("### Result:")

            # Progress bar
            st.progress(probability)

            threshold = 0.5
            if probability > threshold:
                st.success("✅ **HELIPAD DETECTED!**")
                st.metric(label="Confidence", value=f"{probability*100:.2f}%")
                st.balloons()
            else:
                st.error("❌ **NO HELIPAD FOUND**")
                st.metric(label="Confidence (It is empty)", value=f"{(1-probability)*100:.2f}%")
        else:
            st.error("Model could not be loaded.")

# --- Sidebar ---
st.sidebar.title("About")
st.sidebar.info(
    "This model uses **ResNet50** trained on satellite imagery.\n\n"
    "**Accuracy:** ~98.9%\n\n"
    "Created for the Helipad Detection Challenge."
)
