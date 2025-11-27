import itertools
from typing import Iterable, List

from src.graph import Hypergraph
from src.utils import is_connected, power_set, relabel

MotifsRv = List[tuple[tuple[tuple[int, ...]], List[tuple[int, ...]]]]


def motifs_ho_not_full(hg: Hypergraph, N, visited):
    """Computes the motif count for hypergraph motifs of order N. The
    subgraphs checked for motifs are the ones obtained by extending a hyperedge
    of order N-1 with one of its neighboring hyperedges. The subgraphs
    contained in the visited set are ignored

    Args:
        edges (list[tuple[int]]): List of hyperedges in the hypergraph.
        N (int): The order of the motifs to be counted.

    Returns:
        out (list[tuple[tuple[tuple[int]], int]]): A list of tuples where each
            tuple contains a motif (as a tuple of edges) and its corresponding
            count in the hypergraph.
        visited (dict[tuple[int], int]): A dictionary of visited hyperedges of
            size N.

    """
    # assert_hypergraph(hg, weighted=weighted)
    mapping, labeling = generate_motifs(N)

    # if not weighted:
    #     hg = set([tuple(sorted(t)) for t in hg])
    # else:
    #     hg = {tuple(sorted(k)): v for k, v in hg.items()}

    # graph = {}
    # for e in hg.edges:
    #     if e.order >= N:
    #         continue
    #
    #     for e_i in e.nodes:
    #         if e_i not in graph:
    #             graph[e_i] = []
    #         graph[e_i].append(e)

    adjacency = hg.get_adjacency_mut()
    for e in hg.edges:
        if e.order == N - 1:
            for n in e.nodes:
                for e_i in adjacency[n]:
                    tmp = list(e.nodes)
                    tmp.extend(e_i.nodes)
                    tmp = list(set(tmp))
                    if len(tmp) == N and tuple(sorted(tmp)) not in visited:
                        count_motif(hg, tmp, labeling, visited)
                        visited[tuple(sorted(tmp))] = 1

    # D = {}
    # for i in range(len(out)):
    #     D[i] = out[i][0]

    # with open('motifs_{}.pickle'.format(N), 'wb') as handle:
    # pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)

    out = combine_labelings(mapping, labeling)
    return out, visited


def motifs_standard(hg: Hypergraph, N, visited):
    """
    Computes the motif count for hypergraph motifs of order N, considering only
    hyperedges of order 2. The subgraphs contained in the visited set are
    ignored.

    Args:
        edges (list[tuple[int]]): List of hyperedges in the hypergraph.
        N (int): The order of the motifs to be counted.
        visited (set[tuple[int]]): A set of subgraph (tuple of nodes) to ignore
            in the computation of motifs

    Returns:
        out (list[tuple[tuple[tuple[int]], int]]): A list of tuples where each
            tuple contains a motif (as a tuple of edges) and its corresponding
            count in the hypergraph.
    """
    mapping, labeling = generate_motifs(N)

    # if not weighted:
    #     hg = set([tuple(sorted(t)) for t in hg])
    # else:
    #     hg = {tuple(sorted(k)): v for k, v in hg.items()}

    graph = {}

    # z = set()
    # for e in edges:
    #     for n in e:
    #         z.add(n)

    # Construct adjacency matrix for 2-edges
    for e in hg.edges:
        if e.order == 2:
            a, b = e.nodes
            if a not in graph:
                graph[a] = []
            graph[a].append(b)

            if b not in graph:
                graph[b] = []
            graph[b].append(a)

    def graph_extend(sub, ext, v, n_sub):
        if len(sub) == N:
            count_motif(hg, sub, labeling, visited)
            return

        while len(ext) > 0:
            w = ext.pop()
            tmp = set(ext)

            for u in graph[w]:
                if u not in sub and u not in n_sub and u > v:
                    tmp.add(u)

            new_sub = set(sub)
            new_sub.add(w)
            new_n_sub = set(n_sub).union(set(graph[w]))
            graph_extend(new_sub, tmp, v, new_n_sub)

    c = 0

    k = 0
    for v in graph.keys():
        v_ext = set()
        for u in graph[v]:
            if u > v:
                v_ext.add(u)
        k += 1
        if k % 5 == 0:
            print(f"Vertex {k}")
            # print(k, len(z), TOT)

        graph_extend(set([v]), v_ext, v, set(graph[v]))
        c += 1

    # D = {}
    # for i in range(len(out)):
    #     D[i] = out[i][0]

    # with open('motifs_{}.pickle'.format(N), 'wb') as handle:
    # pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)
    out = combine_labelings(mapping, labeling)

    return out


def motifs_ho_full(hg: Hypergraph, N):
    """
    Computes the motif counts for hypergraph motifs of order N. The subgraphs
    checked for motifs are the one induced by the nodes in the hyperedges of
    order N. If weighted is True, the intensity of each motif is returned
    instead of its count

    Args:
        edges (list[tuple[int]]): List of hyperedges in the hypergraph.
        N (int): The order of the motifs to be counted.

    Returns:
        out (list[tuple[tuple[tuple[int]], int|float]]): A list of tuples where each
            tuple contains a motif (as a tuple of edges) and its corresponding
            count in the hypergraph.
        visited (dict[tuple[int], int]): A dictionary of visited hyperedges of
            size N.

    """
    mapping, labeling = generate_motifs(N)

    # if not hg.weighted:
    #     hg = set([tuple(sorted(t)) for t in hg])
    # else:
    #     hg = {tuple(sorted(k)): v for k, v in hg.items()}

    visited = {}
    for e in hg.get_order_map().get(N, []):
        count_motif(hg, e.nodes, labeling, visited)
        visited[e.nodes] = 1

    # D = {}
    # for i in range(len(out)):
    #     D[i] = out[i][0]

    # with open('motifs_{}.pickle'.format(N), 'wb') as handle:
    # pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)

    out = combine_labelings(mapping, labeling)
    return out, visited


def count_motif(hg: Hypergraph, nodes: Iterable[int], labeling, visited={}):
    """
    Increments the count of the motif induced by the given nodes in the specified hypergraph

    Args:
        edges (set[tuple[int]] | dict[tuple[int], numbers.Number]): The list of
            hyperedges in the hypergraph.
        nodes (list[int]): The list of nodes inducing the motif.
        labeling (dict[tuple[tuple[int]], [tuple[int]]]): A dictionary where each key is
            a possible labeling of motifs of size N, and each value is initialized
            to 0.
        visited (dict[tuple[int], int]): A dictionary of visited hyperedges of
        weighted (bool): Whether the hypergraph is weighted or not.
    """
    nodes = tuple(sorted(nodes))
    # print(f"Count motifs: {nodes}")

    if nodes in visited:
        return

    # p_nodes = power_set(nodes)
    # motif = []
    # for edge in p_nodes:
    #     if len(edge) >= 2:
    #         edge = tuple(sorted(list(edge)))
    #         if edge in edges:
    #             motif.append(edge)
    motif = hg.get_induced_subgraph(nodes)

    m = {}
    idx = 1
    for i in nodes:
        m[i] = idx
        idx += 1

    labeled_motif = []
    for e in motif:
        new_e = []
        for node in e.nodes:
            new_e.append(m[node])
        new_e = tuple(sorted(new_e))
        labeled_motif.append(new_e)
    labeled_motif = tuple(sorted(labeled_motif))

    if labeled_motif in labeling:
        labeling[labeled_motif].append(nodes)
        # increment = 1
        # if weighted:
        #     weighted_edges = {}
        #     for e in motif:
        #         weighted_edges[e] = edges[e]
        #     increment = intensity(weighted_edges)
        #
        # labeling[labeled_motif] += increment


def combine_labelings(mapping, labeling):
    out = []
    for motif in mapping.keys():
        motifs = []
        for label in mapping[motif]:
            motifs.extend(labeling[label])

        out.append((motif, motifs))

    out = list(sorted(out))
    return out


def generate_motifs(N):
    """
    Generate all isomorphism classes of connected motifs of size N.

    Args:
        N (int): The size of the motifs (number of nodes).

    Returns:
        mapping (dict[tuple[tuple[int]]: set[tuple[tuple[int]]]]):
            A dictionary mapping each isomorphism class representative
            to the set of all its possible labelings
            (as tuples of edges).
        labeling (dict[tuple[tuple[int]], [tuple[int]]]):
            A dictionary where each key is a possible labeling of motifs of
            size N, and each value is an empty list, to be filled with the
            motifs found in the hypergraph.
    """
    n = N
    assert n >= 2

    h = [i for i in range(1, n + 1)]
    A = []

    for r in range(n, 1, -1):
        A.extend(list(itertools.combinations(h, r)))

    B = power_set(A)

    C = []
    for i in range(len(B)):
        if is_connected(B[i], N):
            C.append(B[i])

    isom_classes = {}

    for i in C:
        edges = sorted(i)
        relabeling_list = list(itertools.permutations([j for j in range(1, n + 1)]))
        found = False
        for relabeling in relabeling_list:
            relabeling_i = relabel(edges, relabeling)
            # print(relabeling_i)
            if tuple(relabeling_i) in isom_classes:
                found = True
                break
        if not found:
            isom_classes[tuple(edges)] = 1

    mapping = {}
    labeling = {}

    for k in isom_classes.keys():
        mapping[k] = set()
        relabeling_list = list(itertools.permutations([j for j in range(1, n + 1)]))
        for relabeling in relabeling_list:
            relabeling_i = relabel(k, relabeling)
            labeling[tuple(sorted(relabeling_i))] = []
            mapping[k].add(tuple(sorted(relabeling_i)))

    return mapping, labeling
