# text_utils.py
import spacy
import re

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])

def clean_texts(text_list):
    cleaned_reviews = []

    preprocessed = []
    for text in text_list:
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www.\S+|https\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        preprocessed.append(text)

    for doc in nlp.pipe(preprocessed, batch_size=100):
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        cleaned_reviews.append(" ".join(tokens))

    return cleaned_reviews
