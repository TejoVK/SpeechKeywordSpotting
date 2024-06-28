import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from streamlit-audiorecorder import st_audiorec

# Load the pre-trained model
imported = tf.saved_model.load("saved")

# Disable PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to decode wav file and make predictions
def make_prediction(wav_data):
    wav_data = tf.convert_to_tensor(wav_data, dtype=tf.string)
    x, sample_rate = tf.audio.decode_wav(wav_data, desired_channels=1, desired_samples=16000)
    x = tf.squeeze(x, axis=-1)  # Remove the last dimension
    waveform = x
    waveform = waveform[tf.newaxis, :]  # Add batch dimension
    prediction = imported(waveform)
    return prediction, waveform, sample_rate

# Function to plot spectrogram
def plot_spectrogram(waveform, sample_rate):
    plt.specgram(waveform.numpy(), Fs=sample_rate.numpy())
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    st.pyplot()

# Streamlit app title
st.title("Keyword Spotting Using Deep Learning")

# Sidebar option to upload or record audio
option = st.sidebar.radio("Select option", ("Upload Audio", "Record Audio"))

# Function to handle uploaded audio
def handle_uploaded_audio():
    uploaded_file = st.sidebar.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        return uploaded_file.read()
    return None

# Function to handle recorded audio
def handle_recorded_audio():
    return st_audiorec()

# Process and display the audio based on user selection
if option == "Upload Audio":
    wav_audio_data = handle_uploaded_audio()
elif option == "Record Audio":
    wav_audio_data = handle_recorded_audio()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')

# Process and display the recorded audio
if wav_audio_data is not None:
    st.subheader("Recorded Audio")
    prediction, waveform, sample_rate = make_prediction(wav_audio_data)
    
    # Labels for the classification
    x_labels = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
    
    # Display prediction results
    st.subheader("Prediction Results")
    fig, ax = plt.subplots()
    ax.bar(x_labels, tf.nn.softmax(prediction['predictions'][0]).numpy())
    ax.set_title('Prediction')
    st.pyplot(fig)
    
    # Display the waveform
    st.subheader("Audio Waveform")
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(waveform[0])) / sample_rate.numpy(), waveform[0].numpy())
    ax.set_title('Waveform')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    st.pyplot(fig)

    # Display the spectrogram
    st.subheader("Spectrogram")
    plot_spectrogram(waveform[0], sample_rate)
