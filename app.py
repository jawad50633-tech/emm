
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
from PIL import Image, ImageOps
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils

# ---------------------------------------------------------------- Page config
st.set_page_config(
    page_title="AIFLA Face Mesh & Emotion Studio",
    page_icon="🧠",
    layout="wide",
)

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MAX_SIDE = 1280  # downscale huge photos for speed


# ------------------------------------------------------------------- Model
@st.cache_resource(show_spinner="Downloading MediaPipe Face Landmarker model...")
def download_model() -> str:
    if not os.path.exists(MODEL_PATH):
        # Download to a temp file first so a failed download never leaves a corrupt model
        fd, tmp_path = tempfile.mkstemp(suffix=".task")
        os.close(fd)
        try:
            urllib.request.urlretrieve(MODEL_URL, tmp_path)
            os.replace(tmp_path, MODEL_PATH)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return MODEL_PATH


@st.cache_resource(show_spinner=False)
def get_landmarker(model_path: str, num_faces: int, min_conf: float):
    """Create the FaceLandmarker once per settings combo (not on every rerun)."""
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=num_faces,
        min_face_detection_confidence=min_conf,
        min_face_presence_confidence=min_conf,
        output_face_blendshapes=True,  # used for the emotion estimate
    )
    return vision.FaceLandmarker.create_from_options(options)


# ----------------------------------------------------------------- Emotion
def estimate_emotions(blendshapes) -> dict:
    """
    Heuristic emotion scores from MediaPipe's 52 facial blendshapes.
    This is a simple rule-based demo, NOT a trained emotion classifier.
    Swap it for DeepFace or your own CNN for real emotion recognition.
    """
    b = {c.category_name: c.score for c in blendshapes}

    def avg(*names):
        return float(np.mean([b.get(n, 0.0) for n in names]))

    happy = avg("mouthSmileLeft", "mouthSmileRight") * 1.2 + avg(
        "cheekSquintLeft", "cheekSquintRight"
    ) * 0.4
    surprised = (
        avg("jawOpen") * 0.6
        + avg("browInnerUp", "browOuterUpLeft", "browOuterUpRight") * 0.8
        + avg("eyeWideLeft", "eyeWideRight") * 0.8
    )
    sad = avg("mouthFrownLeft", "mouthFrownRight") * 1.0 + avg("browInnerUp") * 0.4
    angry = (
        avg("browDownLeft", "browDownRight") * 1.0
        + avg("noseSneerLeft", "noseSneerRight") * 0.6
        + avg("mouthPressLeft", "mouthPressRight") * 0.4
    )
    strongest = max(happy, surprised, sad, angry)
    neutral = max(0.0, 0.6 - strongest)

    scores = {
        "Happy": happy,
        "Neutral": neutral,
        "Surprised": surprised,
        "Sad": sad,
        "Angry": angry,
    }
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}


EMOJI = {"Happy": "😄", "Neutral": "😐", "Surprised": "😲", "Sad": "😢", "Angry": "😠"}


# ---------------------------------------------------------------- Pipeline
def process_face_analysis(image_np, landmarker, draw_mesh, draw_contours):
    annotated = np.ascontiguousarray(image_np.copy())
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_np))
    result = landmarker.detect(mp_image)

    for face_landmarks in result.face_landmarks:
        if draw_mesh:
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_utils.DrawingSpec(
                    color=(0, 255, 128), thickness=1
                ),
            )
        if draw_contours:
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_utils.DrawingSpec(
                    color=(255, 0, 127), thickness=2
                ),
            )

    return annotated, result


# ---------------------------------------------------------------------- UI
st.title("🧠 AIFLA Advanced Face Mesh & Emotion Studio")
st.markdown("Upload a portrait to extract dense 3D facial landmarks and estimate expression.")

model_path = download_model()

with st.sidebar:
    st.header("⚙️ Configuration Controls")
    st.markdown("Customize the computer vision pipeline:")
    draw_tesselation = st.checkbox("Draw Full Face Mesh", value=True)
    draw_facial_contours = st.checkbox("Highlight Facial Contours", value=True)
    num_faces = st.slider("Max faces", 1, 5, 2)
    min_conf = st.slider("Min detection confidence", 0.1, 1.0, 0.5, 0.05)

    st.divider()
    st.info(
        "💡 **AIFLA Student Tip**: The emotion module uses a simple blendshape "
        "heuristic. Plug in DeepFace or a custom PyTorch CNN for real classification!"
    )

uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")  # fix phone-photo rotation
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    image_np = np.array(image)

    landmarker = get_landmarker(model_path, num_faces, min_conf)

    with st.spinner("Running MediaPipe Face Landmarker..."):
        processed_image, result = process_face_analysis(
            image_np, landmarker, draw_tesselation, draw_facial_contours
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 Original Image")
        st.image(image, width="stretch")
    with col2:
        st.subheader("🔬 AI Processed Output")
        st.image(processed_image, width="stretch")

    st.divider()

    if result.face_landmarks:
        st.success(f"✅ Detected **{len(result.face_landmarks)}** face(s) in the image.")

        emotions = estimate_emotions(result.face_blendshapes[0])
        top = max(emotions, key=emotions.get)

        m1, m2, m3 = st.columns(3)
        m1.metric("Landmarks Extracted", f"{len(result.face_landmarks[0])} points")
        m2.metric(
            "Primary Emotion (face 1)",
            f"{top} {EMOJI[top]}",
            f"Score: {emotions[top] * 100:.1f}%",
            delta_color="off",
        )
        m3.metric("Mesh Status", "Active & Rendered" if draw_tesselation else "Hidden")

        st.write("### 📊 Emotion Probability Distribution")
        df = pd.DataFrame({"Emotion": list(emotions), "Probability": list(emotions.values())})
        st.bar_chart(df, x="Emotion", y="Probability", horizontal=True)
    else:
        st.warning(
            "⚠️ No clear face detected. Try an image with better front-facing lighting and visibility."
        )
else:
    st.info("👉 Get started by uploading an image using the file uploader above.")
