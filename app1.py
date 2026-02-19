import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from streamlit_drawable_canvas import st_canvas
from src.model import CNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create upload folder
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

st.title("🧠 Handwritten Digit Recognition")

option = st.radio("Choose Input Method", ["Upload Image", "Draw Digit"])

image = None

# Upload Section
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, width=200)

        save_path = os.path.join("uploads", uploaded_file.name)
        image.save(save_path)
        st.success(f"File saved to {save_path}")

# Drawing Section
if option == "Draw Digit":
    canvas = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas.image_data is not None:
        img = Image.fromarray((canvas.image_data[:, :, 0]).astype("uint8")).convert("L")
        image = img
        st.image(image, width=200)

        save_path = os.path.join("uploads", "drawn_image.png")
        image.save(save_path)
        st.success("Drawing saved to uploads folder")

# Prediction
if image is not None:
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    confidence = torch.max(probabilities).item() * 100

    st.markdown("### Prediction Result")
    st.success(f"Predicted Digit: {predicted.item()}")
    st.info(f"Confidence: {confidence:.2f}%")
