import numpy as np
import streamlit as st
import joblib
from text_utils import clean_texts
from scipy.sparse import hstack

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Smart Review Filter")
st.subheader("Detect whether a product review is genuine or fake")

user_review = st.text_area("Enter your review here:")  
review_freq = 1

if st.button("Predict"):
    if not user_review.strip():
        st.error("Please enter a review.")
    else:
        cleaned = clean_texts([user_review])
        vectorized = vectorizer.transform(cleaned)
        
        extra_features = np.array([[review_freq]])
        extra_features_scaled = scaler.transform(extra_features)

        final_input = hstack([vectorized, extra_features_scaled])

        prediction = int(model.predict(final_input)[0])
        prob = model.predict_proba(final_input)[0][prediction]

        if prediction == 1:
            st.success("Genuine Review")
        else:
            st.error("Fake Review")
        st.write(f"**Confidence**: {prob * 100:.2f}%")
