#%%
# PACKAGES
from sklearn.metrics import accuracy_score
import pickle

MODEL_DIR = "models/"
DATA_DIR = "../data/"

#%%
# LOAD MODELS
with open(MODEL_DIR+"logreg.pkl", "rb") as f:
    model_logreg = pickle.load(f)
with open(MODEL_DIR+"naive_bayes.pkl", "rb") as f:
    model_nb = pickle.load(f)
with open(MODEL_DIR+"model_embeddings.pkl", "rb") as f:
    model_embeddings = pickle.load(f)

# LOAD DATA
with open(DATA_DIR+"X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open(DATA_DIR+"X_test_embeddings.pkl", "rb") as f:
    X_test_embeddings = pickle.load(f)
with open(DATA_DIR+"labels_test.pkl", "rb") as f:
    labels_test = pickle.load(f)

#%%
model_logreg_predictions = model_logreg.predict(X_test)
model_nb_predictions = model_nb.predict(X_test)

model_embeddings_scores = model_embeddings.predict(X_test_embeddings)
model_embeddings_predictions = [1 if i == 1 else 0 for i in (model_embeddings_scores > 0.5).astype("int32")]

print("Logreg model Accuracy:", accuracy_score(labels_test, model_logreg_predictions))
print("NBayes model Accuracy:", accuracy_score(labels_test, model_nb_predictions))
print("Embedd model Accuracy:", accuracy_score(labels_test, model_embeddings_predictions))



#%%
# CONFUSION MATRICES
from utils import plots

m = plots.get_conf_matrix(model_logreg_predictions, labels_test)
plots.create_conf_matrix_fig(m, title="Logreg")

m = plots.get_conf_matrix(model_nb_predictions, labels_test)
plots.create_conf_matrix_fig(m, title="NaiveBayes")

m = plots.get_conf_matrix(model_embeddings_predictions, labels_test)
plots.create_conf_matrix_fig(m, title="Embeddings")

# %%
plots.create_roc_fig(labels_test, model_embeddings_scores)
# %%
