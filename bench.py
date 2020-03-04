import time
import networkx as nx
import talus.morse as morse
import numpy as np


def build_data(n):
    sampled = np.random.random((n, 3))
    return sampled[:, 1:], sampled[:, 0]


for size in [500000]:
    print(size)
    X, y = build_data(size)
    print(X.shape)
    print(y.shape)
    ms_complex = morse.MorseSmaleComplex(k=8, approximate=True)
    ms_complex.fit_transform(X, y)
    1/0
    start = time.time()
    result = morse.persistence_by_knn(points, 8)
    print(time.time() - start)
    print(size)
    start = time.time()
    result = morse.persistence_by_approximate_knn(points, 8, 0.5, 0.001)
    print(time.time() - start)
    print(len(result))
