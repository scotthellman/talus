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

(result, filtration, morse) = morse.persistence(G)

print(result)
print(filtration)
print(morse)

crystals = {k: [] for k, v in result.items() if v > 0}

for node in morse:
    crystals[node[1]].append(node[0])

print("Full morse complex:")
print(crystals)
print("-" * 10)

for time, destroyed, parent in filtration:
    crystals[parent].extend(crystals[destroyed])
    del crystals[destroyed]
    print("At lifetime", time)
    print(crystals)
    print("-" * 10)
