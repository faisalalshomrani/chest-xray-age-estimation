import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Chest X-ray Age Estimation",
    page_icon="🩻",
    layout="centered"
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "densenet121_age_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match the preprocessing used during model training.
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
@st.cache_resource
def load_age_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Place densenet121_age_model.pth in the same folder as app.py."
        )

    # Recreate the exact DenseNet121 regression architecture.
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
def predict_age(model, image):
    image = image.convert("RGB")
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        prediction = model(tensor)

    return float(prediction.squeeze().cpu().item())


# ---------------------------------------------------------
# User interface
# ---------------------------------------------------------
st.title("🩻 Chest X-ray Age Estimation")
st.write(
    "A research prototype that estimates chronological age "
    "from an uploaded frontal chest radiograph using DenseNet121."
)

st.info(
    "Research demonstration only. This prototype is not a medical "
    "device and must not be used for diagnosis, treatment, or patient management."
)

with st.expander("About this research model"):
    st.markdown(
        """
        - **Architecture:** DenseNet121, fine-tuned from ImageNet weights
        - **Input:** RGB image resized to 224 × 224 pixels
        - **Preprocessing:** ImageNet channel normalisation
        - **Training loss:** Smooth L1
        - **Best validation MAE:** 4.25 years
        - **Held-out test MAE:** 4.15 years
        """
    )
    st.caption(
        "The reported MAE values describe average performance across the "
        "study cohort. They are not an uncertainty interval for an individual prediction."
    )

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"],
    help="Upload one frontal chest radiograph in PNG or JPEG format."
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded image")
        st.image(image, caption="Input chest radiograph", use_container_width=True)

        if st.button("Estimate chronological age", type="primary", use_container_width=True):
            try:
                with st.spinner("Loading model and analysing image..."):
                    model = load_age_model()
                    estimated_age = predict_age(model, image)

                st.success("Prediction completed")
                st.metric(
                    label="Estimated chronological age",
                    value=f"{estimated_age:.1f} years"
                )

                if estimated_age < 0 or estimated_age > 100:
                    st.warning(
                        "The output is outside the study cohort's age range "
                        "(0–95 years). Interpret this prediction with particular caution."
                    )

                st.caption(
                    "This is a model-generated estimate, not a verified age or a clinical assessment."
                )

            except Exception as e:
                st.error(f"Could not generate prediction: {e}")

    except Exception:
        st.error("The uploaded file could not be read as an image.")

st.divider()
st.caption(
    "Research prototype | DenseNet121 | NIH ChestX-ray14 study | "
    "Predictions may be affected by image quality, acquisition view, "
    "patient population, and age distribution."
)
