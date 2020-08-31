import networkx as nx
import talus.morse as morse

# TODO: Make these be proper tests instead of a thrown together script


def evaluate_graph(G):
    ms_complex = morse.MorseSmaleComplex.from_graph(G)
    result = ms_complex.morse_data

    print(result.ascending_complex.complex.complex)
    print(result.ascending_complex.complex.lifetimes)


    #print(result.descending_complex.compute_cells_at_lifetime(0))
    #for f in result.descending_complex.filtration:
    #    print(result.descending_complex.compute_cells_at_lifetime(f.lifetime))

    print(result.ascending_complex.compute_cells_at_lifetime(0))

    for f in result.ascending_complex.complex.filtration:
        print(result.ascending_complex.compute_cells_at_lifetime(f.lifetime))
    print("-"*10)


# This one is just a real basic sort of graph
node_vals = [(0, 3), (1, -1), (2, 10), (3, 2), (4, 7)]

nodes = [morse.MorseNode(identifier=i, value=i, vector=list(v)) for i, v in enumerate(node_vals)]


G = nx.Graph()
for node in nodes:
    G.add_node(node)

G.add_edge(nodes[0], nodes[1])
G.add_edge(nodes[0], nodes[3])
G.add_edge(nodes[1], nodes[2])
G.add_edge(nodes[1], nodes[4])
G.add_edge(nodes[3], nodes[4])

evaluate_graph(G)

# Identical values in same morse cell
node_vals = [(0, 1), (1, 1), (2, 1), (3, -1), (4, 10)]
nodes = [morse.MorseNode(identifier=i, value=i, vector=list(v)) for i, v in enumerate(node_vals)]


G = nx.Graph()
for node in nodes:
    G.add_node(node)

G.add_edge(nodes[0], nodes[1])
G.add_edge(nodes[0], nodes[2])
G.add_edge(nodes[1], nodes[3])
G.add_edge(nodes[2], nodes[3])
G.add_edge(nodes[3], nodes[4])

evaluate_graph(G)
