
import streamlit as st
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import CatDogCNN


# -----------------------------
# Load model
# -----------------------------

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    model.eval()
    return model


model = load_model()


# -----------------------------
# Image transformation
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# -----------------------------
# Streamlit App
# -----------------------------

st.title("🐱🐶 Cat and Dog Image Classifier")

st.write(
    "Upload an image of a cat or dog and the trained CNN model "
    "will predict the class."
)


# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Transform image
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Make prediction
    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    # Class names
    classes = ["Cat", "Dog"]

    prediction = classes[predicted.item()]

    confidence_percentage = confidence.item() * 100


    # Display result
    st.subheader("Prediction")

    if prediction == "Cat":
        st.success(
            f"🐱 The model predicts: **CAT**"
        )
    else:
        st.success(
            f"🐶 The model predicts: **DOG**"
        )

    st.write(
        f"Confidence: **{confidence_percentage:.2f}%**"
    )
