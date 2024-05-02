import streamlit as st
import numpy as np
import json
import random
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from gtts import gTTS
from IPython.display import Audio
import nltk
nltk.download('punkt')

# Load data and model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('final.json').read())
model = keras.models.load_model('chatbot.h5')
lemmatizer = WordNetLemmatizer()

# Function to preprocess input
def preprocess_input(user_input):
    user_input_words = nltk.word_tokenize(user_input)
    user_input_words = [lemmatizer.lemmatize(word.lower()) for word in user_input_words]
    input_bag = [1 if word in user_input_words else 0 for word in words]
    return input_bag

# Function to get chatbot response
def get_chatbot_response(user_input):
    input_bag = preprocess_input(user_input)
    prediction = model.predict(np.array([input_bag]))
    predicted_class = classes[np.argmax(prediction)]
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            response = random.choice(intent['responses'])
            return response

# Function to convert text to speech using gTTS and save to disk
def text_to_speech(text):
    tts = gTTS(text)
    audio_file = 'audio.wav'  
    tts.save(audio_file)
    return audio_file

# Streamlit app
def main():
    # Load CSS file
    with open("styles.css", "r") as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    st.markdown('<div class="title">ZetaBot Chat</div>', unsafe_allow_html=True)

    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # User input
    user_input = st.text_input('Enter your query:', '')

    # Process user input and generate response
    if st.button('Send'):
        # Display user message
        st.markdown(f'<div class="message user-message">{user_input}</div>', unsafe_allow_html=True)

        # Get chatbot response
        bot_response = get_chatbot_response(user_input)

        # Display bot response
        st.markdown(f'<div class="message bot-message">{bot_response}</div>', unsafe_allow_html=True)

        # Convert response to speech and play audio
        speech_audio = text_to_speech(bot_response)
        audio_data = open(speech_audio, 'rb').read()
        audio = Audio(data=audio_data, autoplay=True)
        st.audio(audio_data, format='audio/wav')

        # Add a stop button
        stop_button = st.button('Stop', key='stop_button')
        if stop_button:
            st.markdown(
                '''
                <script>
                    var audioElements = document.getElementsByTagName('audio');
                    for (var i = 0; i < audioElements.length; i++) {
                        audioElements[i].pause();
                    }
                </script>
                ''',
                unsafe_allow_html=True
            )

    # Close chat container
    st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    main()
