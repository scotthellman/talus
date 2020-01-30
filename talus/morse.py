# code for calling persistence on a networkx graph
from . import talus
from dataclasses import dataclass, field


@dataclass
class MorseNode:
    identifier: int
    value: field

    def __eq__(self, other):
        return self.identifier == other.identifier

    def __hash__(self):
        return hash(self.identifier)


def persistence(graph):
    nodes = [(n.identifier, n.value) for n in graph]
    edges = [(e[0].identifier, e[1].identifier) for e in graph.edges]
    return talus._persistence(nodes, edges)
