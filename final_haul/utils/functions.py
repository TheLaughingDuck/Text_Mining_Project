import spacy
import re

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and not token.like_num
    ]
    return " ".join(tokens)
