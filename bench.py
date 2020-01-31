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


for size in [100000]:
    print(size)
    points = build_nodes(size)
    start = time.time()
    result = morse.persistence_by_knn(points, 8)
    result = morse.persistence_by_knn(points, 8)
    result = morse.persistence_by_knn(points, 8)
    print(time.time() - start)
    print(len(result))
