#%%

import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import pickle
import math

from utils.functions import preprocess_text

#%%
# LOAD DATA
df = pd.read_csv("../data/data_raw.csv", index_col=0)

# Some reviews are float('nan'). These are turned into empty reviews ("").
# There does not seem to be any special cases that are neither string nor float('nan'), but we put "" just in case.
df["review"] = [i if type(i) == str else ("" if math.isnan(i) else "") for i in df["review"]]

# Remove the empty reviews
df = df[df["review"] != ""]

# Set aside the labels (voted_up: positive review)
y = df["voted_up"]
df = df.drop(["voted_up", "playtime", "helpfulness", "length"], axis=1)

# Drop indices
df = df.reset_index(drop=True)
y = y.reset_index(drop=True)

df.to_pickle("../data/data_reviews.pkl")
y.to_pickle("../data/data_labels.pkl")

#%%
# LOAD DATA
reviews = pd.read_pickle("../data/data_reviews.pkl")
labels = pd.read_pickle("../data/data_labels.pkl")

# PREPROCESS TEXT
reviews["review"] = [preprocess_text(t) for t in reviews["review"]]

# SPLIT DATA
reviews_train, reviews_test, labels_train, labels_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# SETUP vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(reviews_train["review"])
X_test = vectorizer.transform(reviews_test["review"])


# SETUP EMBEDDING
from sentence_transformers import SentenceTransformer

# Load model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Get sentence embeddings
X_train_embeddings = embedder.encode(reviews_train["review"].tolist(), convert_to_numpy=True)
X_test_embeddings = embedder.encode(reviews_test["review"].tolist(), convert_to_numpy=True)
print("Shape of embeddings:", X_train_embeddings.shape)

# Save embedder?
#no I don't think I need to save the embedder, as it is not used in the training process.

#%%
# SAVE THE DATA

# Save embeddings
with open("../data/X_train_embeddings.pkl", "wb") as f:
    pickle.dump(X_train_embeddings, f)
with open("../data/X_test_embeddings.pkl", "wb") as f:
    pickle.dump(X_test_embeddings, f)

# Save the cleaned train and test reviews
with open("../data/data_reviews_train.pkl", "wb") as f:
    pickle.dump(reviews_train, f)
with open("../data/data_reviews_test.pkl", "wb") as f:
    pickle.dump(reviews_test, f)

# Save the vectorized train and test reviews
with open("../data/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
with open("../data/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)

# Save the labels
with open("../data/labels_train.pkl", "wb") as f:
    pickle.dump(labels_train, f)
with open("../data/labels_test.pkl", "wb") as f:
    pickle.dump(labels_test, f)

#%%
# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

#%%
# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# %%
