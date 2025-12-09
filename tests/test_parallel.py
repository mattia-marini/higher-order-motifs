from graph import Hyperedge, Hypergraph

hg = Hypergraph()

hg.add_edge(Hyperedge((1, 2)))
hg.add_edge(Hyperedge((1, 2, 3)))
hg.add_edge(Hyperedge((2, 3)))
hg.add_edge(Hyperedge((1, 2, 3, 4)))


