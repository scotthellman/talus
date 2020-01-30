import networkx as nx
import talus.morse as morse

node_vals = [(3, 1.5), (-1, 0), (2, 1.1), (8, 3)]
nodes = [morse.MorseNode(identifier=i, value=v) for i, v in node_vals]

G = nx.Graph()
for node in nodes:
    G.add_node(node)

G.add_edge(nodes[0], nodes[1])
G.add_edge(nodes[1], nodes[3])
G.add_edge(nodes[0], nodes[2])
G.add_edge(nodes[2], nodes[3])

print(G.nodes)
print(G.edges)

result = morse.persistence(G)

assert(result[-1] == 0)
assert(abs(result[3] - 0.4) < 0.00001)

print(result)
