import os
import openai
import streamlit as st
# import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from datetime import datetime
import tempfile
import base64

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
    
# Function to transcribe audio
def transcribe_audio(audio_file):
    # Pass the file-like object directly to OpenAI's API
    transcription = openai.Audio.transcribe("whisper-1", audio_file)
    return transcription['text']

# I used chatgpt for this def // Function to handle audio transcription for recorded audio
def handle_recorded_audio(audio_value):
    if audio_value is not None:
        # Create a temporary file to save audio data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            tmp_audio_file.write(audio_value.read())  # Write audio data to file as bytes
            tmp_audio_file.seek(0)  # Go back to start of file for reading
            
            # Open the file in binary mode and pass it to the transcription function
            with open(tmp_audio_file.name, "rb") as audio_file:
                transcription_text = transcribe_audio(audio_file)
            return transcription_text
    return None

# Helper functions for summarizing, transcribing, and analyzing the transcription
def abstract_summary_extraction(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Summarize this audio file into a text report. Be sure to include all details that are in the recording the driver uploaded."},
            {"role": "user", "content": transcription}
        ]
    )
    return response.choices[0].message['content']

def key_points_extraction(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract key points from this audio file, including the type of accident, location, time and date of occurrence, individuals involved, and any specific details mentioned. Prioritize identifying key phrases and names, noting emotional cues or urgency, and summarizing the main concern expressed by the speaker. The goal is to provide a concise summary to facilitate quick response and categorization of the report."},
            {"role": "user", "content": transcription}
        ]
    )
    return response.choices[0].message['content']

def action_item_extraction(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Analyze this user-uploaded audio recording to identify and list any specific action items requested or implied by the speaker. Action items might include immediate responses, such as contacting authorities, sending medical assistance, notifying specific personnel, or implementing a containment procedure. Capture any instructions or recommendations given by the speaker regarding the accident. Additionally, extract contextual information to support the action items, such as the type of accident, location, urgency level, individuals involved, potential hazards, and other relevant details."},
            {"role": "user", "content": transcription}
        ]
    )
    return response.choices[0].message['content']

def sentiment_analysis(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Analyze the sentiment of this audio recording as positive, negative, or neutral. Additionally,  use sentiment analysis to gauge the tone of urgency or distress, summarizing all critical points for efficient response and categorization."},
            {"role": "user", "content": transcription}
        ]
    )
    return response.choices[0].message['content']

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

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def photo_rec(image_path, language="English"):
    base64_image = encode_image(image_path)
    response = openai.ChatCompletion.create(
    model="gpt-4-turbo",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": f"What’s in this image? Please respond in {language}."},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
            },
            },
        ],
        }
    ],
    max_tokens=300,
    )
    return response.choices[0].message.content

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
                with open(os.path.join(uploaded_image.name), 'wb') as f:
                    f.write(uploaded_image.getbuffer())
                image_path = os.path.join(uploaded_image.name)
                st.subheader("Description:", divider=True)
                with st.spinner("Analyzing the image..."):
                    content = photo_rec(image_path, target_lang_code)
                    # st.write(content)
                    txt = st.text_area(
                        "Input Additional Comments:",
                        content
                    )
            elif picture:
                st.image(picture, caption="Uploaded Image", use_column_width=True)
                with open(os.path.join(picture.name), 'wb') as f:
                    f.write(picture.getbuffer())
                image_path = os.path.join(picture.name)
                st.subheader("Description:", divider=True)
                with st.spinner("Analyzing the image..."):
                    content = photo_rec(image_path, target_lang_code)
                    # st.write(content)
                    txt = st.text_area(
                        "Input Additional Comments:",
                        content
                    )
                # st.image(picture, caption="Captured Image", use_column_width=True)

            # if st.button("Generate Image Description"):
            #     with st.spinner("Analyzing the image..."):
            #         description = simulate_image_description(language=selected_language)
            #         st.markdown("**Generated Image Description:**")
            #         st.write(description)
        st.markdown("</div>", unsafe_allow_html=True)

    # Audio Transcription Feature
    with st.expander("🎙️ Audio Transcription", expanded=False):
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        audio_value = st.experimental_audio_input(" ")
        uploaded_audio = st.file_uploader("Upload an audio file that contains your voice recording of an accident you saw:", type=["mp3", "wav", "m4a"])

        # If audio recording exists, transcribe it
if audio_value:
    st.write("Transcribing recorded audio... Please wait.")
    transcription = handle_recorded_audio(audio_value)
    if transcription:
        st.subheader("Transcription")
        st.write(transcription)

        abstract_summary = abstract_summary_extraction(transcription)
        key_points = key_points_extraction(transcription)
        action_items = action_item_extraction(transcription)
        sentiment = sentiment_analysis(transcription)
    
        # Display results
        st.title("Summary")
        st.write(abstract_summary)
    
        st.title("Key Points")
        st.write(key_points)

        st.title("Action Items")
        st.write(action_items)

        st.title("Sentiment Analysis")
        st.write(sentiment)


    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/mp3/m4a")
        st.write("Transcribing and analyzing audio... Please wait.")
    
        # Transcribe audio directly using the file-like object
        transcription = transcribe_audio(uploaded_audio)
    
        # Run analyses on transcription
        abstract_summary = abstract_summary_extraction(transcription)
        key_points = key_points_extraction(transcription)
        action_items = action_item_extraction(transcription)
        sentiment = sentiment_analysis(transcription)
    
            # Display results
        st.title("Summary")
        st.write(abstract_summary)
    
        st.title("Key Points")
        st.write(key_points)

        st.title("Action Items")
        st.write(action_items)

        st.title("Sentiment Analysis")
        st.write(sentiment)
    st.markdown("</div>", unsafe_allow_html=True)
