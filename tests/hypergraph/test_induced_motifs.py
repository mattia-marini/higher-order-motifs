from src.graph import Hyperedge, Hypergraph

g = Hypergraph()

id1 = g.add_edge(Hyperedge((1, 2, 3), 2.5))
id2 = g.add_edge(Hyperedge((1, 2), 2))
id3 = g.add_edge(Hyperedge((1, 2, 4), 1))

print(g.get_induced_subgraph())
