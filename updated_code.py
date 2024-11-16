import streamlit as st
import openai
from datetime import datetime
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import tempfile
import torch  # For PyTorch support
import os

# Set up OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize BLIP processor and model (for image captioning)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate an image description using BLIP
def generate_image_description(image_file):
    image = Image.open(image_file).convert("RGB")
    inputs = processor(image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    caption_ids = model.generate(**inputs)
    description = processor.decode(caption_ids[0], skip_special_tokens=True)
    return description

# Function to translate text
def translate_text(text, target_language):
    if target_language == "en":
        return text
    prompt = f"Translate the following text to {target_language}: {text}"
    try:
        translation_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that translates text."},
                {"role": "user", "content": prompt}
            ]
        )
        return translation_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error translating text: {e}"

# Function to transcribe audio
# def transcribe_audio(audio_file):
#     transcription = openai.Audio.transcribe("whisper-1", audio_file)
#     return transcription['text']

# Function to transcribe audio
def transcribe_audio(audio_file):
    # Pass the file-like object directly to OpenAI's API
    transcription = openai.audio.transcriptions.create(
        model='whisper-1', 
        file=audio_file, 
        response_format='text' 
    )
    return transcription

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

# Analysis functions
def abstract_summary_extraction(transcription):
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Summarize the transcription."},
                  {"role": "user", "content": transcription}]
    ).choices[0].message.content.strip()

def key_points_extraction(transcription):
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Extract key points from the transcription."},
                  {"role": "user", "content": transcription}]
    ).choices[0].message.content.strip()

def action_item_extraction(transcription):
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "List action items based on the transcription."},
                  {"role": "user", "content": transcription}]
    ).choices[0].message.content.strip()

def sentiment_analysis(transcription):
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Analyze the sentiment of the transcription."},
                  {"role": "user", "content": transcription}]
    ).choices[0].message.content.strip()

# Streamlit app layout
st.set_page_config(page_title="AI-Powered Traffic Incident Reporter", layout="centered", page_icon="🚦")

# Apply fixed background color and styling using CSS
st.markdown(
    """
    <style>
    body {
        background-color: #e6f0ff; /* Muted Blue */
        color: #333; /* Dark Gray */
        font-family: "Arial", sans-serif;
    }
    .stButton button {
        background-color: #4CAF50; /* Green Button */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #333;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<div class="title">🚦 AI-Powered Traffic Incident Reporter</div>', unsafe_allow_html=True)

# Language selection
languages = {"English": "en", "Spanish": "es", "French": "fr", "German": "de", "Chinese": "zh", "Arabic": "ar", "Hindi": "hi", "Vietnamese": "vi"}
selected_language = st.selectbox("Choose Report Language", list(languages.keys()))
target_lang_code = languages[selected_language]

# INCIDENT REPORT
with st.expander("📝 Incident Report"):
    location = st.text_input("Location of Incident", "Enter location manually")
    date = st.date_input("Date of Incident", datetime.today())
    time = st.text_input("Time of Incident", datetime.now().strftime("%I:%M %p"))
    weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy", "Snowy", "Windy", "Foggy"])
    description = st.text_area("Incident Description", "Describe what happened")

    if st.button("Generate Incident Report"):
        with st.spinner("Generating incident report..."):
            try:
                prompt = f"Generate a structured report for an incident at {location} on {date} {time}, weather: {weather}. Details: {description}"
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Generate a structured report."},
                        {"role": "user", "content": prompt}
                    ]
                )
                report = response.choices[0].message.content.strip()
                translated_report = translate_text(report, target_lang_code)
                st.markdown("**Generated Report (Translated):**")
                st.write(translated_report)
                st.download_button("Download Report", translated_report, "incident_report.txt")
            except Exception as e:
                st.error(f"Error: {e}")

# SPEECH-TO-TEXT
with st.expander("🎤 Speech-to-Text Transcription"):
    audio_value = st.experimental_audio_input(" ")
    uploaded_audio = st.file_uploader("Upload audio recording:", type=["mp3", "wav", "m4a"])

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
        with st.spinner("Processing audio..."):
            try:
                transcription = transcribe_audio(uploaded_audio)
                st.subheader("Transcription")
                st.write(transcription)

                # Analyses
                st.subheader("Summary")
                st.write(abstract_summary_extraction(transcription))

                st.subheader("Key Points")
                st.write(key_points_extraction(transcription))

                st.subheader("Action Items")
                st.write(action_item_extraction(transcription))

                st.subheader("Sentiment Analysis")
                st.write(sentiment_analysis(transcription))
            except Exception as e:
                st.error(f"Error processing audio: {e}")

# IMAGE REPORTING
with st.expander("📷 Image Reporting"):
    enable_camera = st.checkbox("Enable Camera")

    if enable_camera:
        # Allow capturing an image directly from the camera
        image_file = st.camera_input("Capture an image")
    else:
        # Allow the user to upload an image
        image_file = st.file_uploader("Upload an incident image", type=["jpg", "jpeg", "png"])

    if image_file:
        # Display the uploaded or captured image
        st.image(image_file, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            try:
                # Generate a description for the uploaded or captured image
                description = generate_image_description(image_file)
                st.write("**Image Description:**", description)
                
                # Generate a report based on the image description
                prompt = f"Create a report based on this image: {description}"
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Generate a structured report."},
                        {"role": "user", "content": prompt}
                    ]
                )
                report = response.choices[0].message.content.strip()
                translated_report = translate_text(report, target_lang_code)
                
                # Display and download the translated report
                st.markdown("**Generated Image Report (Translated):**")
                st.write(translated_report)
                st.download_button("Download Image Report", translated_report, "image_report.txt")
            except Exception as e:
                st.error(f"Error analyzing image: {e}")