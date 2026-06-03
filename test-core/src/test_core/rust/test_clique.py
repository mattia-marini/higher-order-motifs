def run():
    import rust_core as rc
    import networkx as nx
    g = nx.erdos_renyi_graph(400, 0.25)

    cliques1 = sorted(list(nx.find_cliques(g)))
    cliques1 = [sorted(clique) for clique in cliques1 if len(clique) > 2]
    cliques1 = sorted(cliques1)
    print(len(cliques1))
    # print(cliques)


    adj, original_id, _compressed_id = rc.graph.AdjList.from_edges_mapped(g.edges())
    adj.make_undirected()
    cliques2 = sorted(adj.find_cliques())
    cliques2 = [[original_id[node] for node in clique] for clique in cliques2]
    cliques2 = [sorted(clique) for clique in cliques2 if len(clique) > 2]
    cliques2 = sorted(cliques2)
    print(len(cliques2))
    # print(cliques)


    assert cliques1 == cliques2, "Cliques do not match between NetworkX and Rust implementation"
