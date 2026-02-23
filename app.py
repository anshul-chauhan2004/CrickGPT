import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from gtts import gTTS
import tempfile
import uuid
import os

# ---------------- PAGE ----------------
st.set_page_config(page_title="CrickGPT", page_icon="🏏", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #00ff9c;'>🏏 CrickGPT</h1>
<h4 style='text-align: center;'>AI Cricket Shot Recognition & Commentary Engine</h4>
<hr>
""", unsafe_allow_html=True)

# ---------------- COMMENTATOR MODE ----------------
st.sidebar.title("🎙 Commentary Personality")
commentator_mode = st.sidebar.selectbox(
    "Choose your commentator:",
    ["Neutral Analyst",
     "Excited IPL Commentator (Hindi)",
     "Radio Commentator",
     "Funny Commentator"]
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "my_cricket_shot_classifier.keras",
        compile=False
    )

model = load_model()

classes = ['cover_drive','cut','pull','sweep']

display_names = {
    "cover_drive": "Cover Drive",
    "cut": "Cut Shot",
    "pull": "Pull Shot",
    "sweep": "Sweep Shot"
}

# ---------------- COMMENTARY ----------------
def generate_commentary(label, confidence, mode):

    base = display_names[label]

    if mode == "Neutral Analyst":
        return f"The batter executes a {base}. The AI is {confidence:.1f}% confident. Excellent balance and technique.", "en", False

    elif mode == "Excited IPL Commentator (Hindi)":
        hindi_map = {
            "Cover Drive": "क्या शानदार कवर ड्राइव!",
            "Cut Shot": "ओह कमाल का कट शॉट!",
            "Pull Shot": "शॉर्ट बॉल पर जोरदार पुल शॉट!",
            "Sweep Shot": "बेहतरीन स्वीप शॉट खेला गया!"
        }
        return hindi_map[base] + " दर्शक झूम उठे हैं!", "hi", False

    elif mode == "Radio Commentator":
        return f"He moves onto the front foot... meets it sweetly... and that beautiful {base} races away to the boundary.", "en", True

    else:  # Funny
        return f"Ball said I am safe... batter said not today! BOOM! That {base} has been couriered directly to the boundary!", "en", False


# ---------------- TEXT TO SPEECH ----------------
def create_audio(text, lang, slow):
    filename = f"audio_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(filename)
    return filename


# store audio in session
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload a cricket batting image", type=["jpg","jpeg","png"])

if uploaded_file:

    col1, col2 = st.columns([1,1])

    # LEFT IMAGE
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(uploaded_file, use_container_width=True)

    # PREPROCESS
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_file.read())
    temp_path = temp.name

    img = load_img(temp_path, target_size=(256,256))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # PREDICT
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index] * 100
    predicted_label = classes[predicted_index]

    # RIGHT PANEL
    with col2:

        st.subheader("🤖 AI Analysis")

        st.markdown(f"""
        <div style="
            background-color:#141a26;
            padding:25px;
            border-radius:15px;
            border: 2px solid #00ff9c;">
        <h2 style="color:#00ff9c; text-align:center;">
        {display_names[predicted_label]}
        </h2>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence))
        st.markdown(f"### Confidence: **{confidence:.2f}%**")

        # Generate commentary
        commentary, lang, slow = generate_commentary(predicted_label, confidence, commentator_mode)

        st.markdown("### 🎙 AI Commentary")
        st.info(commentary)

        # Create audio when user clicks
        if st.button("🔊 Play Commentary"):
            if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
                os.remove(st.session_state.audio_file)

            audio_file = create_audio(commentary, lang, slow)
            st.session_state.audio_file = audio_file

        # Play audio
        if st.session_state.audio_file:
            audio_bytes = open(st.session_state.audio_file, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")

        # Feedback
        if confidence > 85:
            st.success("🔥 Perfect Recognition!")
        elif confidence > 65:
            st.warning("👍 Pretty sure about this shot.")
        else:
            st.error("🤔 Not fully confident. Try a clearer image.")