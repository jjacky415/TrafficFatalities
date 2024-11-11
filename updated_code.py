import os
import openai
import streamlit as st
# import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from datetime import datetime
import tempfile

# Set up Streamlit page configuration as the first command
st.set_page_config(page_title="AI-Powered Traffic Incident Reporter", layout="centered", page_icon="🚦")

# Set up OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]  # Replace with your actual OpenAI API key

# Custom CSS styling for a polished design
st.markdown("""
    <style>
    /* General styling */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .dashboard-card {
        background-color: #f9f9fb;
        border-radius: 12px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
    }
    .icon-container {
        font-size: 36px;
        color: #0d6efd;
        text-align: center;
        margin-bottom: 10px;
    }
    .button-main {
        background-color: #0d6efd;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 18px;
        cursor: pointer;
    }
    .center-content {
        text-align: center;
        margin-top: 20px;
        padding: 20px;
        background-color: #f9f9fb;
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Main App Title with Dashboard Style
st.title("🚦 AI-Powered Traffic Incident Reporter")
st.markdown("<p style='color:gray;'>A smart tool for incident reporting, transcription, and image analysis.</p>", unsafe_allow_html=True)

# Language Selection in Sidebar
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Arabic": "ar",
    "Hindi": "hi",
    "Vietnamese": "vi"
}
selected_language = st.sidebar.selectbox("Choose Language", list(languages.keys()))
target_lang_code = languages[selected_language]

# Helper function for translating text using GPT-4
def translate_text(text, target_language):
    if target_language == "en":
        return text  # No translation needed for English
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Translate the following text to {selected_language}."},
                {"role": "user", "content": text}
            ],
            max_tokens=300
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Translation error: {e}"

# Function to generate a simulated image description using GPT-4
def simulate_image_description(language="English"):
    prompt = "Imagine you are looking at a traffic accident scene in a photo. Describe the scene in detail."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant describing traffic accident images."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        description = response['choices'][0]['message']['content'].strip()
        return translate_text(description, target_lang_code)
    except Exception as e:
        return f"Error generating description: {e}"

# Function to record audio
# def record_audio(duration=10, fs=44100):
#     st.info(f"Recording for {duration} seconds...")
#     audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
#     sd.wait()
#     return audio_data, fs

# Set up two-column layout
col1, col2 = st.columns(2)

# Left Column: Incident Reporting Feature
with col1:
    st.markdown("<div class='title-section'>Incident Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='center-content'>", unsafe_allow_html=True)
    
    # Incident Report Input Fields
    location = st.text_input("Location of Incident", "Enter location manually")
    date = st.date_input("Date of Incident", datetime.today())
    time_input = datetime.now().strftime("%I:%M %p")
    time = st.text_input("Time of Incident", time_input)

    # Weather Condition Dropdown
    weather_options = ["Sunny", "Rainy", "Cloudy", "Snowy", "Windy", "Foggy"]
    selected_weather = st.selectbox("Weather Condition", weather_options)

    # Incident Description
    description_input = st.text_area("Incident Description", "Describe what happened")

    # Generate Incident Report Button
    generate_report = st.button("Generate Incident Report")
    st.markdown("</div>", unsafe_allow_html=True)  # Close the center-content div

    # Incident Report Generation and Display in Center
    if generate_report:
        with st.spinner("Generating incident report..."):
            prompt = (f"Generate a structured report for a traffic incident that occurred on {date} at {time} "
                      f"in {location} with weather conditions: {selected_weather}. Details: {description_input}")
            try:
                incident_report = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an assistant that generates structured incident reports."},
                        {"role": "user", "content": prompt}
                    ]
                )
                report_text = incident_report['choices'][0]['message']['content'].strip()
                report_text = translate_text(report_text, target_lang_code)
                
                # Displaying the Incident Report Summary in the center of the main page
                st.markdown("<div class='center-content'>", unsafe_allow_html=True)
                st.markdown("**Generated Incident Report:**")
                st.write(report_text)
                st.download_button("Download Incident Report", data=report_text, file_name="incident_report.txt")
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error generating incident report: {e}")

# Right Column: Other Features (Image Reporting, Audio Transcription, etc.)
with col2:
    # Image Reporting Feature
    with st.expander("📷 Image Reporting", expanded=True):
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Upload an incident image", type=["jpg", "png", "jpeg"])
        enable_camera = st.checkbox("Enable camera")
        picture = st.camera_input("Take a picture", disabled=not enable_camera)

        if uploaded_image or picture:
            if uploaded_image:
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            elif picture:
                st.image(picture, caption="Captured Image", use_column_width=True)

            if st.button("Generate Image Description"):
                with st.spinner("Analyzing the image..."):
                    description = simulate_image_description(language=selected_language)
                    st.markdown("**Generated Image Description:**")
                    st.write(description)
        st.markdown("</div>", unsafe_allow_html=True)

    # Audio Transcription Feature
    with st.expander("🎙️ Audio Transcription", expanded=False):
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        uploaded_audio = st.file_uploader("Upload an audio file for transcription", type=["mp3", "wav"])

        if uploaded_audio:
            if st.button("Transcribe Uploaded Audio"):
                try:
                    transcription = openai.Audio.transcribe("whisper-1", uploaded_audio)
                    transcription_text = transcription['text']
                    transcription_text = translate_text(transcription_text, target_lang_code)
                    st.markdown("**Transcription Result:**")
                    st.write(transcription_text)
                except Exception as e:
                    st.error(f"Error transcribing audio: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Speech-to-Text Transcription Feature
    # with st.expander("🎤 Speech-to-Text Transcription", expanded=False):
    #     st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    #     duration = st.slider("Select recording duration (seconds)", 1, 60, 10)

    #     if st.button("Record and Transcribe Audio"):
    #         audio_data, fs = record_audio(duration=duration)
    #         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
    #             write(temp_audio_file.name, fs, audio_data)
    #             st.audio(temp_audio_file.name)

    #             try:
    #                 with open(temp_audio_file.name, "rb") as audio_file:
    #                     transcription = openai.Audio.transcribe("whisper-1", audio_file)
    #                     transcription_text = transcription['text']
    #                     transcription_text = translate_text(transcription_text, target_lang_code)
    #                     st.markdown("**Transcription Result:**")
    #                     st.write(transcription_text)
    #             except Exception as e:
    #                 st.error(f"Error transcribing recorded audio: {e}")
    #     st.markdown("</div>", unsafe_allow_html=True)
