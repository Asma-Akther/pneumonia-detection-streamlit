
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

st.set_page_config(page_title="Pneumonia Detection UI", layout="wide")

MODELS_DIR = os.path.expanduser("~\Downloads\Pneumonia detection project\models")

CNN_MODEL_CANDIDATES = [
    os.path.join(MODELS_DIR, "pneumonia_balanced_cnn_model.h5"),
    os.path.join(MODELS_DIR, "pneumonia_cnn_model.h5"),
    os.path.join(MODELS_DIR, "pneumonia_balanced_cnn_model.keras"),
    os.path.join(MODELS_DIR, "pneumonia_cnn_model.keras"),
    "pneumonia_balanced_cnn_model.h5",
    "pneumonia_cnn_model.h5",
    "pneumonia_balanced_cnn_model.keras",
    "pneumonia_cnn_model.keras",
]

RESNET_MODEL_CANDIDATES = [
    os.path.join(MODELS_DIR, "pneumonia_resnet50_model.h5"),
    os.path.join(MODELS_DIR, "pneumonia_resnet50_model.keras"),
    "pneumonia_resnet50_model.h5",
    "pneumonia_resnet50_model.keras",
]

EFFICIENTNET_FULL_MODEL_CANDIDATES = [
    os.path.join(MODELS_DIR, "efficientnet_model.keras"),
    os.path.join(MODELS_DIR, "pneumonia_efficientnet_model.keras"),
    "efficientnet_model.keras",
    "pneumonia_efficientnet_model.keras",
]

EFFICIENTNET_WEIGHTS_CANDIDATES = [
    os.path.join(MODELS_DIR, "pneumonia_efficientnet.weights.h5"),
    os.path.join(MODELS_DIR, "efficientnetb0_pneumonia.weights.h5"),
    "pneumonia_efficientnet_weights.h5",
    "efficientnetb0_pneumonia.weights.h5",
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def preprocess_cnn(pil_img):
    img = pil_img.convert("RGB").resize((128, 128))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def preprocess_resnet(pil_img):
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = resnet_preprocess(arr)
    return arr


def preprocess_efficientnet(pil_img):
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = efficientnet_preprocess(arr)
    return arr


def build_efficientnet_model():
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3))
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model


@st.cache_resource
def load_selected_model(model_name: str):
    if model_name == "CNN":
        model_path = first_existing(CNN_MODEL_CANDIDATES)
        if not model_path:
            raise FileNotFoundError("CNN model file not found.")
        model = load_model(model_path)
        return model, preprocess_cnn, model_path

    if model_name == "ResNet50":
        model_path = first_existing(RESNET_MODEL_CANDIDATES)
        if not model_path:
            raise FileNotFoundError("ResNet50 model file not found.")
        model = load_model(model_path)
        return model, preprocess_resnet, model_path

    if model_name == "EfficientNetB0":
        full_model_path = first_existing(EFFICIENTNET_FULL_MODEL_CANDIDATES)
        if full_model_path:
            try:
                model = load_model(full_model_path)
                return model, preprocess_efficientnet, full_model_path
            except Exception:
                pass

        weights_path = first_existing(EFFICIENTNET_WEIGHTS_CANDIDATES)
        if not weights_path:
            raise FileNotFoundError("EfficientNet model or weights file not found.")
        model = build_efficientnet_model()
        model.load_weights(weights_path)
        return model, preprocess_efficientnet, weights_path

    raise ValueError("Unsupported model selection.")


def predict_one(model, preprocess_fn, pil_img):
    arr = preprocess_fn(pil_img)
    prob = float(model.predict(arr, verbose=0)[0][0])
    if prob > 0.5:
        pred = "PNEUMONIA"
        confidence = prob
    else:
        pred = "NORMAL"
        confidence = 1 - prob
    return pred, confidence, prob


st.title("Pneumonia Detection")
st.caption("Upload any number of chest X-ray images and run prediction using CNN, ResNet50, or EfficientNetB0.")

with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Choose model", ["CNN", "ResNet50", "EfficientNetB0"])
    show_images = st.checkbox("Show uploaded images", value=True)
    run_button = st.button("Run Prediction", use_container_width=True)

try:
    model, preprocess_fn, loaded_from = load_selected_model(selected_model)
    st.sidebar.success(f"Loaded: {selected_model}")
    st.sidebar.caption(f"Source: {loaded_from}")
except Exception as e:
    st.error(f"Could not load {selected_model}: {e}")
    st.stop()

uploaded_files = st.file_uploader(
    "Upload one or more X-ray images",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} file(s) ready.")

if run_button:
    if not uploaded_files:
        st.warning("Upload at least one image first.")
        st.stop()

    results = []
    image_panels = []

    progress = st.progress(0)
    status = st.empty()

    for i, uploaded_file in enumerate(uploaded_files, start=1):
        status.write(f"Processing {i}/{len(uploaded_files)}: {uploaded_file.name}")
        try:
            pil_img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
            pred, confidence, raw_score = predict_one(model, preprocess_fn, pil_img)
            results.append({
                "filename": uploaded_file.name,
                "prediction": pred,
                "confidence_percent": round(confidence * 100, 2),
                "raw_score": round(raw_score, 6),
            })
            image_panels.append((uploaded_file.name, pil_img, pred, confidence))
        except UnidentifiedImageError:
            results.append({
                "filename": uploaded_file.name,
                "prediction": "INVALID IMAGE",
                "confidence_percent": None,
                "raw_score": None,
            })
        except Exception as e:
            results.append({
                "filename": uploaded_file.name,
                "prediction": f"ERROR: {str(e)}",
                "confidence_percent": None,
                "raw_score": None,
            })

        progress.progress(i / len(uploaded_files))

    status.empty()
    progress.empty()

    results_df = pd.DataFrame(results)

    st.subheader("Results")
    st.dataframe(results_df, use_container_width=True)

    normal_count = int((results_df["prediction"] == "NORMAL").sum())
    pneumonia_count = int((results_df["prediction"] == "PNEUMONIA").sum())
    invalid_count = int(results_df["prediction"].astype(str).str.contains("INVALID|ERROR", case=False, na=False).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("NORMAL", normal_count)
    c2.metric("PNEUMONIA", pneumonia_count)
    c3.metric("Invalid / Error", invalid_count)

    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV",
        data=csv_data,
        file_name=f"{selected_model.lower()}_predictions.csv",
        mime="text/csv"
    )

    if show_images and image_panels:
        st.subheader("Predicted Images")
        cols = st.columns(3)
        for idx, (name, pil_img, pred, confidence) in enumerate(image_panels):
            with cols[idx % 3]:
                st.image(
                    pil_img,
                    caption=f"{name}\n{pred} ({confidence*100:.2f}%)",
                    use_container_width=True
                )
