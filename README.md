# Talus

Fast partitioning of point-cloud data (or graphs) according to a real-valued function over the points. These partitions correspond to regions that flow to some local minima or maxima - as a concrete example, if applied to topography data using elevation as the real-valued function, partitioning by local minima would correspond to watersheds. Furthermore, also computes how _persistent_ each partition is. In the topography example, if we partitioned by maxima, then these persistence values would correspond to the topographical prominence of the peaks in a given landscape.

## Installation

For Rust:

```
cargo add talus
```

For Python:
```
pip install talus_python
```

## Example Usage

```python
import networkx as nx
import talus_python.morse as morse

node_locs = [(0, 3), (1, -1), (2, 10), (3, 2), (4, 7)]
nodes = [
    morse.MorseNode(identifier=i, value=i, vector=list(v))
    for i, v in enumerate(node_locs)
]

G = nx.Graph()
for node in nodes:
    G.add_node(node)

G.add_edge(nodes[0], nodes[1])
G.add_edge(nodes[0], nodes[3])
G.add_edge(nodes[1], nodes[2])
G.add_edge(nodes[1], nodes[4])
G.add_edge(nodes[3], nodes[4])

ms_complex = morse.MorseSmaleComplex.from_graph(G)
result = ms_complex.morse_data

# See what local maxima each node was assigned to
print("Assignments:", result.ascending_complex.complex.complex)

# See how persistent each local maxima is
print("Persistence:", result.ascending_complex.complex.lifetimes)

# Show the node partitions at a certain timestep in the filtration
print("Partitions:", result.ascending_complex.compute_cells_at_lifetime(0))

```

```
Assignments: {0: 4, 3: 4, 1: 4, 4: 4, 2: 2}
Persistence: {1: 0.0, 3: 0.0, 2: 1.0, 0: 0.0, 4: inf}
Partitions: {2: [2], 4: [0, 3, 1, 4]}
```

## Details

In technical jargon, what we're doing is computing a filtration of morse complexes. The algorithm we implement is from [Analysis of Scalar Fields over Point
Cloud Data](https://dl.acm.org/doi/10.5555/1496770.1496881), specifically following the description from [Scaling Up Writing in the Curriculum: Batch Mode Active Learning for Automated Essay Scoring](https://dl.acm.org/doi/10.1145/3330430.3333629).