import networkx as nx
import talus.morse as morse

node_vals = [(0, 3), (1, -1), (2, 10), (3, 2), (4, 7)]
nodes = [morse.MorseNode(identifier=i, value=v) for i, v in node_vals]

G = nx.Graph()
for node in nodes:
    G.add_node(node)

G.add_edge(nodes[0], nodes[1])
G.add_edge(nodes[0], nodes[3])
G.add_edge(nodes[1], nodes[2])
G.add_edge(nodes[1], nodes[4])
G.add_edge(nodes[3], nodes[4])

print(G.nodes)
print(G.edges)

result = morse.persistence(G)

print(result)

print(result.descending_complex.compute_cells_at_lifetime(0))
for f in result.descending_complex.filtration:
    print(result.descending_complex.compute_cells_at_lifetime(f.lifetime))
