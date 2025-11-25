from src.graph import Hypergraph, Hyperedge

g = Hypergraph()

id1 = g.add_edge(Hyperedge((1, 2, 3), 2.5))
id2 = g.add_edge(Hyperedge((1, 2), 2))
id3 = g.add_edge(Hyperedge((1, 2, 4), 1))

edge1 = g.get_edge_by_id(id2)
edge2 = g.get_edge_by_id(id2)
edge1.nodes = (1, 2, 3)


print(g._edges)
print(g.get_edges_by_nodes((1, 2, 3)))
print(g.get_first_edges_by_nodes((1, 2, 3)))

print(edge1)
print(edge2)
g.remove_edge(id2)
