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
is_frequent = st.checkbox("Is the reviewer a frequent reviewer?", value=False)

if st.button("Predict"):
    if not user_review.strip():
        st.error("Please enter a review.")
    else:
        cleaned = clean_texts([user_review])
        vectorized = vectorizer.transform(cleaned)

        reviewer_freq = 1 if is_frequent else 0

        word_count = len(user_review.split())
        char_count = len(user_review)

        length_features = np.array([[word_count, char_count]])
        length_scaled = scaler.transform(length_features)

        reviewer_freq_array = np.array([[reviewer_freq]])

        final_numeric = np.hstack([reviewer_freq_array, length_scaled])

        final_input = hstack([vectorized, final_numeric])

        prediction = int(model.predict(final_input)[0])
        prob = model.predict_proba(final_input)[0][prediction]

        if prediction == 1:
            st.success("Genuine Review")
        else:
            st.error("Fake Review")

        st.write(f"**Confidence**: {prob * 100:.2f}%")
