import joblib
import librosa
import numpy as np
import os
import streamlit as st
from pydub import AudioSegment
from PIL import Image
import json

# read in dictionary of class label, full name pairs from json
with open('full_names.json', 'r') as f:
    FULL_NAMES = json.load(f)

# number of MFCC samples to extract
N_MFCC = 40


@st.cache
def load_model(model_path):
    """
    Loads model from given path.

    Parameter: model_path - path to model (str)
    Returns: classifier model
    """
    model = joblib.load(model_path)
    return model


def extract_features(audio):
    """
    Extracts mfccs as features from input audio.

    Parameter: audio - librosa audio
    Returns: np.array of mfccs (array of floats)
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=8000, n_mfcc=N_MFCC)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    return mfccs_processed


def play_song(file_name):
    """
    Displays audio player in Streamlit for song at given filepath.

    Paramter: file_name - path to audio file (str)
    Returns: nothing
    """
    try:
        audio = open(file_name, 'rb')
        audio_b = audio.read()
        st.audio(audio_b, format='audio/mp3')
    except FileNotFoundError:
        st.write('audio file not found')


def show_bird(prediction):
    """
    tries to find and display a photo of predicted bird.

    Parameter: prediction - predicted bird (str)
    Returns: nothing
    """
    try: 
        img = Image.open('./images/' + prediction + '.jpg')
        st.image(img, use_column_width=True, caption='your lovely ' + FULL_NAMES[prediction])
    except FileNotFoundError:
        st.write('no image available for your lovely ' + FULL_NAMES[prediction])


# set streamlit title and header
st.title("Birdcall Identifier")
st.header("Trained on over 21,000 audio files, I'll classify your birdcall into one of 263 species.")

# loads stored scikit-learn model
model = load_model('birdcall_model.pkl')

# creates streamlit uploader for mp3 of bird call
song = st.file_uploader("Upload an mp3: ", type=['mp3'])

# does the following when a song has been uploaded
if song is not None:
    # exports temp mp3
    song_file = AudioSegment.from_mp3(song)
    path = './' + song.name
    song_file.export(path, format='mp3')

    # loads librosa audio
    audio, sample_rate = librosa.load(path, sr=8000, res_type='kaiser_fast') 

    # displays audio player using temp mp3
    play_song(path)

    # does the following when the "where's my bird?" button is pressed
    if st.button("What's my bird?"):
        # extracts features, makes and reports prediction
        features = extract_features(audio)
        prediction = model.predict([features])[0]
        st.write('We think ' + song.name + ' is the call of the ' + '***' + FULL_NAMES[prediction] + '***')
        
        # displays image of bird
        show_bird(prediction)
        
        # removes temp file
        os.remove(path)
