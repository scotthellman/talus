# code for calling persistence on a networkx graph
from . import talus
from dataclasses import dataclass, field
from typing import List


@dataclass
class MorseNode:
    identifier: int
    value: field
    vector: List[float] = field(default_factory=list)

    def __eq__(self, other):
        return self.identifier == other.identifier

    def __hash__(self):
        return hash(self.identifier)


def persistence(graph):
    nodes = [n for n in graph]
    edges = [(e[0].identifier, e[1].identifier) for e in graph.edges]
    return talus._persistence(nodes, edges)


def persistence_by_knn(points, k):
    return talus._persistence_by_knn(points, k)


def persistence_by_approximate_knn(points, k, sample_rate, precision):
    return talus._persistence_by_approximate_knn(points, k, sample_rate, precision)
