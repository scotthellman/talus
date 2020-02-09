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


@dataclass
class MorseFiltrationStep:
    lifetime: float
    destroyed_id: int
    owning_id: int


class MorseSmaleData:

    def __init__(self, rust_output):
        self.descending_complex = MorseData(rust_output[0])
        self.ascending_complex = MorseData(rust_output[1])


class MorseData:

    def __init__(self, rust_output):
        self.lifetimes = rust_output[0]
        self.filtration = [MorseFiltrationStep(*f) for f in rust_output[1]]
        self.assignments = {a[0]: a[1] for a in rust_output[2]}

    def compute_cells_at_lifetime(self, lifetime: float):
        crystals = {k: [] for k, v in self.lifetimes.items() if v > 0}
        for node, parent in self.assignments.items():
            crystals[parent].append(node)

        # NOTE: filtration is in sorted order
        for step in self.filtration:
            if lifetime is not None and step.lifetime > lifetime:
                break
            crystals[step.owning_id].extend(crystals[step.destroyed_id])
            del crystals[step.destroyed_id]
        return crystals


def persistence(graph):
    nodes = [n for n in graph]
    edges = [(e[0].identifier, e[1].identifier) for e in graph.edges]
    data = talus._persistence(nodes, edges)
    return MorseSmaleData(data)


def persistence_by_knn(points, k: int):
    data = talus._persistence_by_knn(points, k)
    return MorseSmaleData(data)


def persistence_by_approximate_knn(points, k, sample_rate, precision):
    data = talus._persistence_by_approximate_knn(points, k, sample_rate, precision)
    return MorseSmaleData(data)
