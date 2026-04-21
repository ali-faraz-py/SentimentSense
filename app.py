import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentify Pro", page_icon="🧠", layout="centered")

@st.cache_resource
def load_nlp_pipeline():
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    zero_shot_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    return sentiment_pipe, zero_shot_pipe

sentiment_model, aspect_model = load_nlp_pipeline()

st.title("🧠 Sentify Pro: Aspect Intelligence")
st.write("This AI detects **what** you are talking about and **how** you feel about it.")

user_input = st.text_area("Enter your review:", placeholder="The pizza was great but the service was slow...", height=150)

if st.button("Run Advanced Analysis"):
    if user_input.strip() != "":
        with st.spinner("Processing NLP Pipeline..."):
            
            candidate_labels = ["Food", "Service", "Price", "Atmosphere", "Health"]
            topic_results = aspect_model(user_input, candidate_labels, multi_label=True)
            
            sentiment_results = sentiment_model(user_input)[0]

            st.divider()
            
            st.subheader("🎯 Aspects Detected")
            cols = st.columns(len(candidate_labels))
            
            for i, label in enumerate(topic_results['labels']):
                score = topic_results['scores'][i]
                if score > 0.6:
                    st.info(f"**{label}** detected ({score*100:.0f}% match)")

            st.divider()

            label = sentiment_results['label']
            conf = sentiment_results['score']
            
            if label == "POSITIVE":
                st.success(f"### Overall Tone: POSITIVE 😊")
                st.metric("Confidence", f"{conf*100:.1f}%")
            else:
                st.error(f"### Overall Tone: NEGATIVE 😞")
                st.metric("Confidence", f"{conf*100:.1f}%")
                
    else:
        st.warning("Please enter some text first!")