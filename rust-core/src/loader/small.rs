use crate::graph::Hypergraph;

pub fn load_conference(dataset_path: String) -> Hypergraph {
    Hypergraph::new()
}

// import networkx as nx
//
// dataset = f"{cfg.DATASET_DIR}/conference.dat"
//
// fopen = open(dataset, "r")
// lines = fopen.readlines()
//
// graph = {}
// for l in lines:
//     t, a, b = l.split()
//     t = int(t) - 32520
//     a = int(a)
//     b = int(b)
//     if t in graph:
//         graph[t].append((a, b))
//     else:
//         graph[t] = [(a, b)]
//
// fopen.close()
//
// def standard_construction():
//     cm = cast(StandardConstructionMethod, construction_method)
//     hg = Hypergraph()
//
//     for k in graph.keys():
//         e_k = graph[k]
//         G = nx.Graph(e_k, directed=False)
//         c = list(nx.find_cliques(G))
//         for i in c:
//             i = tuple(sorted(i))
//
//             if not (cm.limit_edge_size and len(i) > cm.limit_edge_size):
//                 if cm.weighted:
//                     if hg.has_edge_with_nodes(i):
//                         handle = hg.get_first_edges_by_nodes(i)
//                         handle.weight += 1.0
//                     else:
//                         hg.add_edge(Hyperedge(i, 1.0))
//                 else:
//                     if not hg.has_edge_with_nodes(i):
//                         hg.add_edge(Hyperedge(i))
//
//     hg.normalize_weights(cm.normalization_method)
//     return hg
