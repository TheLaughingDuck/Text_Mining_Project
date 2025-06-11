#%%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pickle

DATA_DIR = "../data/"
with open(DATA_DIR+"X_train_embeddings.pkl", "rb") as f:
    X_train_embeddings = pickle.load(f)

with open(DATA_DIR+"labels_train.pkl", "rb") as f:
    labels_train = pickle.load(f)

#%%
X_train_embeddings = X_train_embeddings[0:1000, :]
labels_train = labels_train[0:1000]
pca = PCA(n_components=2)  # for 2D visualization
X_pca = pca.fit_transform(X_train_embeddings)

#%%
colors = ["r" if i == 0 else "blue" if i == 1 else "green" for i in labels_train]
alphas = [0.8 if i == 0 else 0.8 if i == 1 else 1 for i in labels_train]

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.8)
plt.title("PCA of Sentence Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Class")
plt.grid(True)
plt.tight_layout()
plt.show()


#%%
# Proportion of variance explained
pca_10 = PCA(n_components=10)
X_pca_10 = pca_10.fit_transform(X_train_embeddings)

plt.figure(figsize=(8, 6))
plt.bar(x = [i for i in range(10)], height=pca_10.explained_variance_ratio_)


print("Explained variance by first 10 components:")
print(pca_10.explained_variance_ratio_)
print("Cumulative explained variance:", np.sum(pca_10.explained_variance_ratio_))


# %%
colors
# %%
