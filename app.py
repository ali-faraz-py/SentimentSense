import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import streamlit as st


word_index = imdb.get_word_index()
MODEL_PATH = "sentiment_model.h5" 
MAX_LENGTH = 200

reverse_word_index = {v: k for k, v in word_index.items()}


@st.cache_resource
def load_my_model():
    return keras.models.load_model(MODEL_PATH)

model = load_my_model()
 
  
def decode_review(encoded_text):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_text])
 
 
def preprocess_text(text):
    words = text.lower().split()
    
    encoded = [word_index.get(w, 2) + 3 for w in words]
    
    padded = pad_sequences([encoded], maxlen=MAX_LENGTH)
    
    return padded

st.set_page_config(page_title="AI Movie Reviewer", page_icon="🎬")
st.title("🎬 Movie Review Sentiment AI")
st.write("Type a movie review below to see if the AI thinks it is Positive or Negative.")

user_input = st.text_area("Enter your review here:", placeholder="The cinematography was brilliant but the plot was a bit slow...")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        padded_input = preprocess_text(user_input)
        prediction = model.predict(padded_input, verbose=0)[0][0]
        
        if prediction > 0.5:
            st.success(f"**POSITIVE 😊** (Score: {prediction:.2f})")
        else:
            st.error(f"**NEGATIVE 😞** (Score: {prediction:.2f})")
            
        st.info("A score closer to 1.0 is very positive, while closer to 0.0 is very negative.")
    else:
        st.warning("Please enter some text first!")
 
 
def predict_sentiment(text):
    padded = preprocess_text(text)
    
    prediction = model.predict(padded, verbose=0)[0][0]
    
    sentiment = "POSITIVE 😊" if prediction > 0.5 else "NEGATIVE 😞"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": float(f"{confidence:.4f}"),
        "confidence_percent": f"{confidence * 100:.2f}%",
        "raw_score": float(f"{prediction:.4f}")
    }
 