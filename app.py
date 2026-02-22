import joblib 
import streamlit as st
import numpy as np 
import pandas as pd 


@st.cache_resource 
def load_model(): 
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("linearsvc_model.pkl")
    return vectorizer, model 

vectorizer, model = load_model()

st.set_page_config(page_title="Sentiment Analysis Demo", layout="centered")

st.title("🛍️ Tiki Sentiment Analysis")
st.markdown("Model: TF-IDF + LinearSVC")

text = st.text_area("Enter a comment:")

if st.button("Predict"): 
    if text.strip() != "": 
        x = vectorizer.transform([text]) 
        prediction = model.predict(x)[0] 

        if prediction == 1: 
            st.success("😊 Positive")
        else: 
            st.error("😡 Negative")
    else: 
        st.warning("Please enter a comment!")





