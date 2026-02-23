import edge_tts
import asyncio
import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tempfile
import uuid
import os
import random
from PIL import Image

# ================= PAGE =================
st.set_page_config(page_title="CrickGPT", page_icon="🏏", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #00ff9c;'>🏏 CrickGPT</h1>
<h4 style='text-align: center;'>AI Cricket Shot Recognition & Commentary Engine</h4>
<hr>
""", unsafe_allow_html=True)

# ================= COMMENTATOR MODE =================
st.sidebar.title("🎙 Commentary Personality")
commentator_mode = st.sidebar.selectbox(
    "Choose your commentator:",
    [
        "Neutral Analyst",
        "Excited IPL Commentator (Hindi)",
        "Radio Commentator",
        "Funny Commentator"
    ]
)

# ================= VOICE MAP (REAL COMMENTATORS) =================
VOICE_MAP = {
    "Neutral Analyst": "en-GB-RyanNeural",
    "Excited IPL Commentator (Hindi)": "hi-IN-MadhurNeural",
    "Radio Commentator": "en-GB-SoniaNeural",
    "Funny Commentator": "en-US-GuyNeural"
}

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("my_cricket_shot_classifier.h5", compile=False)
    return model

model = load_model()

# Classes
classes = ['cover_drive','cut','pull','sweep']

display_names = {
    "cover_drive": "Cover Drive",
    "cut": "Cut Shot",
    "pull": "Pull Shot",
    "sweep": "Sweep Shot"
}

# ================= COMMENTARY =================
def generate_commentary(label, confidence, mode):

    base = display_names[label]

    if mode == "Neutral Analyst":
        return f"The batter executes a {base}. The AI is {confidence:.1f}% confident. Excellent balance and technique."

    elif mode == "Excited IPL Commentator (Hindi)":
        hindi_map = {
            "Cover Drive": "क्या शानदार कवर ड्राइव!",
            "Cut Shot": "ओह कमाल का कट शॉट!",
            "Pull Shot": "शॉर्ट बॉल पर जोरदार पुल शॉट!",
            "Sweep Shot": "बेहतरीन स्वीप शॉट खेला गया!"
        }
        return hindi_map[base] + " दर्शक झूम उठे हैं!"

    elif mode == "Radio Commentator":
        return f"He moves onto the front foot... meets it sweetly... and that beautiful {base} races away to the boundary!"

    else:
        return f"Ball said I am safe... batter said not today! BOOM! That {base} has been couriered directly to the boundary!"

# ================= TEXT TO SPEECH (REAL AI VOICES) =================
async def generate_voice(text, voice, filename):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(filename)

def create_audio(text, voice):
    filename = f"audio_{uuid.uuid4().hex}.mp3"
    asyncio.run(generate_voice(text, voice, filename))
    return filename

# ================= IMAGE PREDICTION =================
st.header("📷 Upload Image to Detect Shot")

uploaded_file = st.file_uploader("Upload a cricket batting image", type=["jpg","jpeg","png"])

if uploaded_file:

    col1, col2 = st.columns([1,1])

    # LEFT
    with col1:
        st.subheader("Uploaded Image")
        st.image(uploaded_file, width='stretch')

    # PREPROCESS
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_file.read())
    temp_path = temp.name

    img = load_img(temp_path, target_size=(256,256))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # PREDICTION
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index] * 100
    predicted_label = classes[predicted_index]

    # RIGHT
    with col2:
        st.subheader("AI Analysis")

        st.markdown(f"""
        <div style="background-color:#141a26;padding:25px;border-radius:15px;border: 2px solid #00ff9c;">
        <h2 style="color:#00ff9c;text-align:center;">
        {display_names[predicted_label]}
        </h2>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence))
        st.markdown(f"### Confidence: **{confidence:.2f}%**")

        commentary = generate_commentary(predicted_label, confidence, commentator_mode)

        st.markdown("### 🎙 AI Commentary")
        st.info(commentary)

        # PLAY BUTTON WITH DIFFERENT VOICES
        if st.button("🔊 Play Commentary"):

            selected_voice = VOICE_MAP[commentator_mode]
            audio_file = create_audio(commentary, selected_voice)

            audio_bytes = open(audio_file, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")

        # Feedback
        if confidence > 85:
            st.success("🔥 Perfect Recognition!")
        elif confidence > 65:
            st.warning("👍 Pretty sure about this shot.")
        else:
            st.error("🤔 Not fully confident. Try a clearer image.")

# ================= SHOT VISUALIZER =================
st.markdown("---")
st.header("📚 Cricket Shot Visual Guide")

DATASET_PATH = "datasets"

shot_choice = st.selectbox("Choose a shot to see example:", classes)

if st.button("Show Example Shot"):

    class_dir = os.path.join(DATASET_PATH, shot_choice)

    if not os.path.exists(class_dir):
        st.error("Dataset folder not found in GitHub repository.")
    else:
        images = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]

        if len(images) == 0:
            st.warning("No images inside this class.")
        else:
            chosen = random.choice(images)
            img_path = os.path.join(class_dir, chosen)

            img = Image.open(img_path)

            st.image(img, caption=f"Example of {display_names[shot_choice]}", width='stretch')
