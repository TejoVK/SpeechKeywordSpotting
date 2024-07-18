# Keyword Spotting Using Deep Learning

This repository contains a Streamlit application for keyword spotting using deep learning. The application recognizes specific keywords from audio inputs, utilizing a pre-trained model that we trained using TensorFlow to make predictions from recorded or uploaded audio files.

You can use this model through this streamlit app [here](https://speechkeywordspotting-lxtnkkvezjmbhsxunkk72a.streamlit.app/) or can run it locally following the steps given below.

## Features

- **Audio Input Options**: Users can either upload a `.wav` audio file or record their voice directly within the app.
- **Prediction Visualization**: The app displays the predicted keyword probabilities, the audio waveform, and the spectrogram of the audio signal.
- **Model**: Uses a TensorFlow model that we trained to recognize keywords including "down," "go," "left," "no," "right," "stop," "up," and "yes."

## Installation

1. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

4. Open your web browser and go to `http://localhost:8501`.

## Usage

Choose between uploading an audio file or recording your voice:
- **Upload Audio**: Click on "Upload Audio" in the sidebar and upload a `.wav` file.
- **Record Audio**: Click on "Record Audio" in the sidebar and record your voice.

Once the audio is processed, view the following results:
- **Recorded Audio**: Listen to the audio you provided.
- **Prediction Results**: View a bar chart of the predicted keyword probabilities.
- **Audio Waveform**: See the waveform of the audio signal.
- **Spectrogram**: Visualize the spectrogram of the audio signal.
