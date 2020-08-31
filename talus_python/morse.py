# code for calling persistence on a networkx graph
from . import talus_python as talus
from .talus_python import MorseNode, MorseFiltrationStepPy
from dataclasses import dataclass, field
from typing import List

MorseFiltrationStep = MorseFiltrationStepPy

class MorseSmaleData:

    def __init__(self, rust_output):
        self.descending_complex = MorseData(rust_output[0])
        self.ascending_complex = MorseData(rust_output[1])


class MorseData:
    """ Stores the results of the Morse complex computation.

    Surfaces three attributes:
    `lifetimes`: A map of each variable ID to its persistence value.
    `filtration`: A list containing the times at which each partition is merged
        into its parent, alongside the ID of that parent.
    `assignments`: A map of each variable ID to its partition.
    """

    def __init__(self, morse_complex):
        self.complex = morse_complex

    def compute_cells_at_lifetime(self, lifetime: float):
        # FIXME: This isn't really correct. It's partitions that survived for at least `lifetime` length
        # The docstring kind of implies that they shows up at lifetime, which is a separate concept
        """ Returns the partitions that survive for at least `lifetime`.

        Any partitions that survive for shorter than `lifetime` will be merged into partitions
        that do survive, according to the Morse filtration
        """
        cells = {k: [] for k, v in self.complex.lifetimes.items() if v > 0}
        for node, parent in self.complex.complex.items():
            cells[parent].append(node)

        # NOTE: filtration is in sorted order
        for step in self.complex.filtration:
            if lifetime is not None and step.lifetime > lifetime:
                break
            cells[step.owning_id].extend(cells[step.destroyed_id])
            del cells[step.destroyed_id]
        return cells


class MorseSmaleComplex:
    """ Computes the Morse complex given a set of points and their values.

    This is intended to be a sklearn-like API into the Morse code.

    `k`: The number of neighbors used when constructing the underlying kNN graph
    `approximate`: If true, computes the approximate kNN graph. Otherwise, computes
        the exact kNN graph.
    `sample_rate`: Controls how many new edge are considered at each step in the
        approximate graph construction. Has no effect in `approximate` is `False`.
    `precision`: The lower this value, the stricter the convergence criterion for
        approximate graph construction. Has no effect in `approximate` is `False`.

    If you already have a graph, a MorseSmaleComplex can be constructed using `from_graph`.
    """

    def __init__(self, k=8, approximate=False, sample_rate=0.5, precision=0.001):
        self.k = k
        self.approximate = approximate
        self.sample_rate = sample_rate
        self.precision = precision

    @staticmethod
    def from_graph(graph):
        ms_complex = MorseSmaleComplex(None, None)
        ms_complex.morse_data = MorseSmaleComplex._persistence(graph)
        return ms_complex

    @staticmethod
    def _persistence_by_knn(points, k: int):
        data = talus._persistence_by_knn(points, k)
        return MorseSmaleData(data)

    @staticmethod
    def _persistence_by_approximate_knn(points, k, sample_rate, precision):
        data = talus._persistence_by_approximate_knn(points, k, sample_rate, precision)
        return MorseSmaleData(data)

    @staticmethod
    def _persistence(graph):
        nodes = [n for n in graph]
        edges = [(e[0].identifier, e[1].identifier) for e in graph.edges]
        data = talus._persistence(nodes, edges)
        return MorseSmaleData(data)

    def fit_transform(self, X, labels):
        morse_points = []
        for i, (point, label) in enumerate(zip(X, labels)):
            morse_points.append(MorseNode(identifier=i, value=label, vector=list(point)))
        if self.approximate:
            self.morse_data = self._persistence_by_approximate_knn(morse_points, k=self.k,
                                                                   sample_rate=self.sample_rate,
                                                                   precision=self.precision)
        else:
            self.morse_data = self._persistence_by_knn(morse_points, k=self.k)
        return self.morse_data

