import time
import networkx as nx
import talus.morse as morse
import numpy as np


def build_nodes(n):
    sampled = np.random.random((n, 3))
    points = []
    for i, row in enumerate(sampled):
        points.append(morse.MorseNode(identifier=i, value=row[0], vector=list(row[1:])))
    return points


for size in [500000]:
    print(size)
    points = build_nodes(size)
    result = morse.persistence_by_approximate_knn(points, 8, 0.8, 0.001)
    1/0
    start = time.time()
    result = morse.persistence_by_knn(points, 8)
    print(time.time() - start)
    print(size)
    start = time.time()
    result = morse.persistence_by_approximate_knn(points, 8, 0.5, 0.001)
    print(time.time() - start)
    print(len(result))
