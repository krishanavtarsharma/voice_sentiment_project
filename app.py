import streamlit as st
import speech_recognition as sr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from googletrans import Translator
from io import BytesIO
import datetime
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
import torch

# -------------------
# Initialize Models (Error-Free)
# -------------------
device = 0 if torch.cuda.is_available() else -1

model_name_sentiment = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer_sentiment = AutoTokenizer.from_pretrained(model_name_sentiment)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(model_name_sentiment)
sentiment_pipeline = pipeline("sentiment-analysis", model=model_sentiment, tokenizer=tokenizer_sentiment, device=device)

translator = Translator()
summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

r = sr.Recognizer()

st.set_page_config(page_title="Ultra-Modern Voice-to-Text Dashboard", layout="wide")

# -------------------
# Custom CSS - Pink Animated UI
# -------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #ffe4e1, #fff0f5);
    color: #4b0082;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    color: #ff1493;
    text-align: center;
    font-weight: bold;
    font-size: 3em;
    animation: pulse 2s infinite;
}
h2 { color: #ff69b4; }
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
.stButton>button {
    background-color: #ffb6c1;
    color: #4b0082;
    font-weight: bold;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    transition: 0.3s;
    border: none;
}
.stButton>button:hover {
    background-color: #ff69b4;
    transform: scale(1.05);
}
.stTextArea>textarea {
    background-color: #fff0f5;
    color: #4b0082;
    font-size: 1.1em;
}
</style>
""", unsafe_allow_html=True)

st.title("🎀 Ultra-Modern AI Voice-to-Text Dashboard")

# -------------------
# Session State
# -------------------
if 'live_text' not in st.session_state:
    st.session_state.live_text = ""
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'sentiment_count' not in st.session_state:
    st.session_state.sentiment_count = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
if 'all_words' not in st.session_state:
    st.session_state.all_words = []

# -------------------
# Sidebar
# -------------------
st.sidebar.header("🎀 Options")
option = st.sidebar.radio("Choose Input Method:", ["Upload Audio 📁", "Live Recording 🎤"])
live_duration = st.sidebar.slider("Live Recording Duration (seconds):", 3, 30, 10)

# -------------------
# Audio Processing
# -------------------
def process_audio(audio_data):
    try:
        text = r.recognize_google(audio_data, language="en-US")
        sentiment = sentiment_pipeline(text)[0]
        text_hi = translator.translate(text, dest='hi').text

        now = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.live_text += f"[{now}] 💬 {text}\n"

        label = sentiment['label'].upper()
        if label in st.session_state.sentiment_count:
            st.session_state.sentiment_count[label] += 1

        st.session_state.all_words.extend(text.split())

        return text, sentiment, text_hi
    except sr.UnknownValueError:
        st.warning("❌ Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"⚠️ Request Error: {e}")
    except sr.WaitTimeoutError:
        st.warning("⏰ Listening timed out.")
    return None, None, None

# -------------------
# Animated Typing Effect
# -------------------
notebook_panel = st.empty()
def animated_notebook_update(new_text):
    display_text = ""
    for char in new_text:
        display_text += char
        notebook_panel.markdown(f"### 📝 Live Notebook<br><span style='color:purple'>{display_text}</span>", unsafe_allow_html=True)
        time.sleep(0.01)

# -------------------
# Word Cloud
# -------------------
def show_wordcloud():
    if st.session_state.all_words:
        text = " ".join(st.session_state.all_words)
        wc = WordCloud(width=800, height=400, background_color="mistyrose", colormap="pink").generate(text)
        plt.figure(figsize=(10,4))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

# -------------------
# Summarizer
# -------------------
def summarize_text(text):
    inputs = summarizer_tokenizer(text, return_tensors="pt", truncation=True)
    summary_ids = summarizer_model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4)
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# -------------------
# Sentiment Bars
# -------------------
def display_sentiment_bars():
    counts = st.session_state.sentiment_count
    fig = px.bar(x=list(counts.keys()), y=list(counts.values()), color=list(counts.keys()),
                 color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'orange'})
    st.plotly_chart(fig, use_container_width=True)

# -------------------
# Upload Audio
# -------------------
if option == "Upload Audio 📁":
    st.subheader("📁 Upload an audio file (.wav / .mp3)")
    uploaded_file = st.file_uploader("Choose file", type=["wav", "mp3"])
    if uploaded_file:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format="audio/wav")
        audio_file = BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
            text, sentiment, text_hi = process_audio(audio_data)
            if text:
                animated_notebook_update(text)
                show_wordcloud()
                display_sentiment_bars()

# -------------------
# Live Recording
# -------------------
elif option == "Live Recording 🎤":
    st.subheader("🎤 Record live audio")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Live Recording"):
            st.session_state.recording = True
    with col2:
        if st.button("⏹ Stop Recording"):
            st.session_state.recording = False

    if st.session_state.recording:
        st.image("https://i.gifer.com/ZZ5H.gif", width=100, caption="🎙️ Listening...")
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1)
            while st.session_state.recording:
                audio_data = r.listen(source, timeout=5, phrase_time_limit=10)
                text, sentiment, text_hi = process_audio(audio_data)
                if text:
                    animated_notebook_update(text)
                    show_wordcloud()
                    display_sentiment_bars()

# -------------------
# Download Transcript
# -------------------
if st.session_state.live_text:
    st.download_button("💾 Download Transcript", st.session_state.live_text, file_name="transcript.txt")

# -------------------
# Summary
# -------------------
if st.session_state.live_text:
    st.subheader("📝 Transcript Summary")
    summary = summarize_text(st.session_state.live_text)
    st.markdown(f"💡 {summary}")
