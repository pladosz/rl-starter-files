import numpy as np
import faiss


class FaissKNeighbors:
    def __init__(self, n_neighbors=5):
        self.index = None
        self.y = None
        self.k = n_neighbors

    def fit(self, X):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))

    def kneighbors(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        return distances