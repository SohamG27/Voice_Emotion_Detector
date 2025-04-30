# 🎨 Final Beautifully Styled app.py (Voice Emotion Mental Health App)

import streamlit as st
import speech_recognition as sr
import soundfile as sf
import opensmile
import joblib
import os
from datetime import datetime
import pandas as pd
from PIL import Image
import base64
import plotly.graph_objects as go

# Set page config immediately after imports
st.set_page_config(page_title="Voice Emotion Mental Health App", layout="centered")

# Add background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read())
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("data:image/png;base64,{encoded_string.decode()}");
             background-size: cover;
             background-position: center;
             background-repeat: no-repeat;
             background-attachment: fixed;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# --- Call Background Image Function ---
add_bg_from_local('StockCake-Beach Yoga Sunset_1745862986.jpg')  # Make sure this is your file name and path

# Set background image via pure CSS (no st.image())
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-size: cover;
        background-position: top-center;
        background-repeat: no-repeat;
        background-attachment: flex;
    }
    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 24px;
    }
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Custom CSS for better appearance
st.markdown(
    """
    <style>
    body {
        background-color: #E6F2FF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 24px;
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load trained model and label encoder
clf = joblib.load(".venv/models/audio_emotion_model.pkl")
le = joblib.load(".venv/models/label_encoder.pkl")

# Initialize OpenSMILE feature extractor
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# App title and banner
st.title("🧘 Voice Emotion Mental Health App")
st.subheader("🎙️ Speak your feelings, track your emotional well-being.")
st.write("Welcome! This app gently detects your emotions from your voice tone and helps you stay mindful of your emotional health.")

# Record button
if st.button("🔴 Record Your Voice"):
    with st.spinner("Recording... Please speak calmly."):

        try:
            # Record audio
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=5)

            # Save audio temporarily
            filename = f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())

            # Display recorded audio
            st.audio(filename, format="audio/wav")

            # Extract features
            features = smile.process_file(filename).values
            features_df = pd.DataFrame(features)

            # Predict emotion
            predicted_class = clf.predict(features_df)[0]
            predicted_label = le.inverse_transform([predicted_class])[0]

            # Show prediction result
            st.success(f"🎯 Detected Emotion: **{predicted_label.upper()}**")
            # Save prediction to session
            if "history" not in st.session_state:
                st.session_state.history = []

            st.session_state.history.append(predicted_label)

            # Limit history to last 10 emotions
            st.session_state.history = st.session_state.history[-10:]

            # 🎨 Plot Emotion Timeline
            st.subheader("🧠 Your Recent Emotional Journey")

            emotion_numeric = {emotion: idx for idx, emotion in enumerate(le.classes_)}

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=list(range(1, len(st.session_state.history) + 1)),
                y=[emotion_numeric[e] for e in st.session_state.history],
                mode='lines+markers',
                line=dict(color='#4CAF50', width=3),
                marker=dict(size=10, color='#4CAF50'),
            ))

            fig.update_layout(
                xaxis_title="Record #",
                yaxis_title="Emotion",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(emotion_numeric.values()),
                    ticktext=list(emotion_numeric.keys())
                ),
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent outer area
                font=dict(size=14),
            )

            st.plotly_chart(fig, use_container_width=True)
            # 📦 Suggestion Box Based on Emotion
            st.subheader("💡 Personalized Suggestion")

            suggestions = {
                "happy": "You're doing great! Keep up the positivity and spread smiles around. 😊 Plan a gratitude journal today.",
                "sad": "It's okay to feel sad. Consider taking a nature walk or calling a friend. 🧘‍♂️ Plan a relaxing self-care evening.",
                "angry": "Take deep breaths. Try a 5-minute meditation or journaling. 😤 Plan a calm, tech-free hour.",
                "fear": "You're stronger than your fears! 💪 Try writing down your worries and challenging them.",
                "neutral": "Feeling balanced is powerful. 🎯 Plan a small creative activity to keep your day interesting!",
                "disgust": "Acknowledge unpleasant feelings. 🧹 Maybe clean a small space or refresh your surroundings."
            }

            st.info(suggestions.get(predicted_label.lower(), "Stay mindful and take care! 🌿"))

            # Clean up temp file
            os.remove(filename)

        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ for mental wellness. Stay mindful.")
