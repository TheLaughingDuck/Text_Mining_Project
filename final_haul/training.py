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

MODEL_DIR = "models/"

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

with open(MODEL_DIR+"logreg.pkl", "wb") as f:
    pickle.dump(model_logreg, f)

# #%%
# MODEL 2: NAIVE BAYES
model_nb = MultinomialNB()
model_nb.fit(X_train, labels_train)

with open(MODEL_DIR+"naive_bayes.pkl", "wb") as f:
    pickle.dump(model_nb, f)

#%%
# MODEL 3: EMBEDDING + NEURAL NETWORK
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import regularizers

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import EarlyStopping

from keras.optimizers import Adam

#%%
with open("../data/X_train_embeddings.pkl", "rb") as f:
    X_train_embeddings = pickle.load(f)
with open("../data/X_test_embeddings.pkl", "rb") as f:
    X_test_embeddings = pickle.load(f)

# Subset for testing purposes
#X_train_embeddings = X_train_embeddings[:10, :]

#%%
# Dense network trained on embeddings
model_embeddings = Sequential([
    Dense(64, activation='relu', input_shape=(384,), kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
#%%
class_weights = compute_class_weight(y=labels_train, classes=np.unique(labels_train), class_weight="balanced")
class_weights = {0: class_weights[0], 1: class_weights[1]}  # Adjust class weights to handle class imbalance

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

optimizer = Adam(learning_rate=0.0001)
model_embeddings.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history = model_embeddings.fit(X_train_embeddings,
                    labels_train,
                    epochs=1000,
                    batch_size=32,
                    validation_split=0.1, 
                    verbose=1, 
                    class_weight=class_weights,
                    callbacks=[early_stop])


# Plot training and validation loss across epochs
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Binary CrossEntropyLoss')
plt.show()


#%%
# Plot validation accuracy
plt.figure(figsize=(12, 9))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Save model
with open(MODEL_DIR+"model_embeddings.pkl", "wb") as f:
    pickle.dump(model_embeddings, f)


# %%
