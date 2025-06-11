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

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load data
X_train = pd.read_pickle("../data/X_train.pkl")
X_test = pd.read_pickle("../data/X_test.pkl")

# Load cleaned reviews
reviews_train = pd.read_pickle("../data/data_reviews_train.pkl")
reviews_test = pd.read_pickle("../data/data_reviews_test.pkl")

# Load labels
labels_train = pd.read_pickle("../data/labels_train.pkl")
labels_test = pd.read_pickle("../data/labels_test.pkl")

#%%
# MODEL 1: LOGISTIC REGRESSION
model_logreg = LogisticRegression(max_iter=1000, random_state=62)
model_logreg.fit(X_train, labels_train)


# #%%
# MODEL 2: NAIVE BAYES
model_nb = MultinomialNB()
model_nb.fit(X_train, labels_train)

#%%
# MODEL 3: NEURAL NETWORK ON TFIDF

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Convert sparse matrices to dense format
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Create model
model_dense = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_dense.shape[1],)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_dense.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_dense.fit(X_train_dense, labels_train, epochs=5, batch_size=32, validation_split=0.1)

#y_pred_nn = (model.predict(X_test_dense) > 0.5).astype("int32")
#print("Neural Network Results:\n", classification_report(y_test, y_pred_nn))


#%%
# MODEL 4: EMBEDDING + NEURAL NETWORK
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

# Load model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Get sentence embeddings
X_train_embeddings = embedder.encode(reviews_train["review"].tolist(), convert_to_numpy=True)
print("Shape of embeddings:", X_train_embeddings.shape)

with open("../data/X_train_embeddings.pkl", "wb") as f:
    pickle.dump(X_train_embeddings, f)

# Subset for testing purposes
X_train_embeddings = X_train_embeddings[:10, :]

#%%
# Dense network trained on embeddings
model = Sequential([
    Dense(128, activation='relu', input_shape=(384,)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_embeddings, labels_train, epochs=5, batch_size=32, validation_split=0.1)



#%%
# EVALUATE MODELS
from sklearn.metrics import accuracy_score

model_logreg_predictions = model_logreg.predict(X_test)
model_nb_predictions = model_nb.predict(X_test)
model_dense_predictions = model_dense.predict(X_test)

X_test_embeddings = embedder.encode(reviews_test["review"].tolist(), convert_to_numpy=True)
model_embeddings_predictions = model.predict(X_test_embeddings)


print("Logreg model Accuracy:", accuracy_score(labels_test, model_logreg_predictions))
print("NBayes model Accuracy:", accuracy_score(labels_test, model_nb_predictions))
print("NeuNet model Accuracy:", accuracy_score(labels_test, (model_dense_predictions > 0.5).astype("int32")))
print("Embedd model Accuracy:", accuracy_score(labels_test, (model_embeddings_predictions > 0.5).astype("int32")))

#%%
# Save predictions for later analysis