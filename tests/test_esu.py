import numpy as np

from src.graph import Hyperedge, Hypergraph
from src.loaders import load_hospital
from src.motifs.motifs_base import combine_labelings, count_motif, generate_motifs
from tests.util import time_function

motif_mapping_3 = {}
motif_mapping_4 = {
    # Order 3
    ((1, 2), (1, 3)): 0,
    ((1, 2), (1, 3), (2, 3)): 1,
    # Order 4
    ((1, 2), (1, 3), (1, 4)): 0,
    ((1, 2), (1, 3), (1, 4), (2, 3)): 1,
    ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4)): 2,
    ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)): 3,
    ((1, 2), (1, 4), (2, 3)): 4,
    ((1, 3), (1, 4), (2, 3), (2, 4)): 5,
}


def esu(hg: Hypergraph, order: int) -> tuple[int]:
    np_graph = hg.get_flattened_np_matrix()
    hg.compute_adjacency()
    n = np_graph.shape[0]
    count = 0
    mapping, labeling = generate_motifs(order)
    labeling = {k: 0 for k in labeling}

    # building adjacency list to reuse mtofis base method
    adj = hg.get_adjacency_copy()
    graph = {}
    for a, l in adj.items():
        graph[a] = []
        for e in l:
            if e.order == 2:
                b = e.nodes[1] if e.nodes[0] == a else e.nodes[0]
                graph[a].append(b)

    def graph_extend(sub, ext, v, n_sub):
        # print("Sub:", sub)
        # print("Ext:", ext)
        # print("N_Sub:", n_sub)
        # nonlocal count
        if len(sub) == order:
            motif = hg.get_induced_subgraph(sub)
            motif = set([m for m in motif if m.order == 2])

            m = {}
            idx = 1
            for i in sub:
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
                labeling[labeled_motif] += 1
            # count_motif(hg, sub, labeling)
            # count += 1
            return

        while len(ext) > 0:
            w = ext.pop()
            tmp = set(ext)

            for u in graph[w]:
                if u not in n_sub and u > v:
                    tmp.add(u)

            new_sub = set(sub)
            new_sub.add(w)
            new_n_sub = set(n_sub).union(set(graph[w]))
            graph_extend(new_sub, tmp, v, new_n_sub)

    for v in graph:
        v_ext = set()
        for u in graph[v]:
            if u > v:
                v_ext.add(u)

        graph_extend(set([v]), v_ext, v, set(graph[v]).union({v}))

    # print(labeling)
    rv = []
    for motif in mapping.keys():
        count = 0
        for label in mapping[motif]:
            count += labeling[label]

        rv.append((motif, count))

    rv = list(sorted(rv))
    # print(rv)
    rv = [t for t in rv if len([e for e in t[0] if len(e) > 2]) == 0]  # filtro diedges
    rv = [(t[0], t[1]) for t in rv]  # considero solo count dei motifs

    rv = [(mapping[motif], count) for motif, count in rv]
    rv = [count for _, count in sorted(rv)]

    return tuple(rv)


def np_esu(hg: Hypergraph, order: int) -> int:
    """
    Iterative implementation so that it can be ported to GPU later
    """
    np_matrix = hg.get_flattened_np_matrix()
    n = np_matrix.shape[0]

    if n < order:
        return 0

    np_adj = [np.where(np_matrix[node] == 1)[0] for node in range(n)]

    def extend(start_vertex: int):
        # State arrays
        count = 0

        # Declaration
        ext = np.zeros(n, dtype=int)
        n_sub_set = np.zeros(n, dtype=bool)
        n_sub_v = np.zeros(n, dtype=int)

        stack_record = np.dtype(
            [
                ("v", int),
                ("offset", int),
                ("ext_size", int),
                ("n_sub_size", int),
                ("added_n_sub_count", int),
            ]
        )
        stack = np.zeros(order, dtype=stack_record)

        def v():
            return stack[stack_size - 1]["v"]

        def set_v(val):
            stack[stack_size - 1]["v"] = val

        def offset():
            return stack[stack_size - 1]["offset"]

        def set_offset(val):
            stack[stack_size - 1]["offset"] = val

        def ext_size():
            return stack[stack_size - 1]["ext_size"]

        def set_ext_size(val):
            stack[stack_size - 1]["ext_size"] = val

        def n_sub_size():
            return stack[stack_size - 1]["n_sub_size"]

        def set_n_sub_size(val):
            stack[stack_size - 1]["n_sub_size"] = val

        def added_n_sub_count():
            return stack[stack_size - 1]["added_n_sub_count"]

        def set_added_n_sub_count(val):
            stack[stack_size - 1]["added_n_sub_count"] = val

        stack_size = 1
        set_v(start_vertex)
        set_offset(0)
        set_ext_size(0)
        set_n_sub_size(1)
        n_sub_set[start_vertex] = True
        n_sub_v[0] = start_vertex

        for u in np_adj[start_vertex]:
            if not n_sub_set[u]:
                n_sub_set[u] = True
                n_sub_v[n_sub_size()] = u
                set_n_sub_size(n_sub_size() + 1)

            if u > start_vertex:
                ext[ext_size()] = u
                set_ext_size(ext_size() + 1)

        set_added_n_sub_count(n_sub_size())

        def dbg_values():
            print("Sub:", [int(stack[i]["v"]) for i in range(stack_size)])
            print("Ext:", ext[: ext_size()])
            print("Offset:", offset())
            print("N_Sub:", n_sub_v[: n_sub_size()])
            print()

        # limit = 0
        while stack_size > 0:
            # if limit > 10:
            #     break
            # limit += 1
            # dbg_values()

            # Check if we found a motif
            if stack_size == order:
                count += 1
                # print(
                #     "!!!!!!!!!!!Found subgraph:",
                #     [int(stack[i]["v"]) for i in range(stack_size)],
                #     "\n",
                # )

            if ext_size() == offset() or stack_size == order:
                # No more extensions, backtrack

                # Clear n_sub_set
                # n_sub_size = stack[stack_size - 1]["n_sub_size"]
                # added_n_sub_count = stack[stack_size - 1]["added_n_sub_count"]
                for i in range(n_sub_size() - added_n_sub_count(), n_sub_size()):
                    n_sub_set[n_sub_v[i]] = False

                stack_size -= 1
            else:
                # Pop next vertex from ext
                # set_ext_size(ext_size() - 1)
                next_vertex = ext[offset()]
                set_offset(offset() + 1)

                new_added_n_sub_count = 0
                new_n_sub_size = n_sub_size()
                new_ext_size = ext_size()

                # Update ext with valid neighbors
                for u in np_adj[next_vertex]:
                    if not n_sub_set[u] and u > start_vertex:
                        ext[new_ext_size] = u
                        new_ext_size += 1

                # Update n_sub with neighbors of next_vertex
                for u in np_adj[next_vertex]:
                    if not n_sub_set[u]:
                        n_sub_set[u] = True
                        n_sub_v[new_n_sub_size] = u
                        new_added_n_sub_count += 1
                        new_n_sub_size += 1

                # Push to stack
                stack[stack_size]["v"] = next_vertex
                stack[stack_size]["ext_size"] = new_ext_size
                stack[stack_size]["offset"] = offset()
                stack[stack_size]["n_sub_size"] = new_n_sub_size
                stack[stack_size]["added_n_sub_count"] = new_added_n_sub_count
                stack_size += 1

        return count

    count = 0
    for start_vertex in range(n):
        count += extend(start_vertex)

    return count


def ad_hoc(hg: Hypergraph, order: int) -> tuple[int, ...]:
    # if order != 3:
    #     raise NotImplementedError("Ad-hoc method only implemented for order 3")

    adj = hg.get_digraph_adj_list()
    adj_set = [set(neighbors) for neighbors in adj]
    n = hg.n
    mat = hg.get_digraph_matrix()
    # print(graph)

    def count_3(vertex: int) -> list[int]:
        total = [0] * 2

        # straight patterns
        neighbors_count = len(adj[vertex])
        # print(neighbors_count)

        cross_edges = set()
        for u in adj[vertex]:
            for w in adj[u]:
                if w in adj_set[vertex] and w > u:
                    cross_edges.add((u, w))

        total[0] = (neighbors_count * (neighbors_count - 1) // 2) - len(cross_edges)

        total[1] = sum(1 for x in cross_edges if x[0] < vertex and x[1] < vertex)
        # print(f"Vertex {vertex}: {total}")

        # triangle patterns

        return total

    def count_4(vertex: int) -> list[int]:
        total = [0] * 6

        # I motif simmetrici van contati solo a partire dal vertice maggiore

        for distal in adj[vertex]:
            # Vicini in comune a distal e vertex

            # tipo 2, 3: simmetrici
            common = adj_set[distal].intersection(adj_set[vertex])
            common_less = set([c for c in common if c < vertex])

            cross_infra = 0
            cross_infra_less = 0
            for c in common:
                for u in adj[c]:
                    # u != vertex and u != distal and
                    if c < u and u in common:
                        if u < vertex and u < distal:
                            cross_infra_less += 1
                        cross_infra += 1
                        # cross_edges_tot.add(tuple(sorted((c, u))))

            cross_inter = 0
            cross_inter_less = 0
            for c in common:
                for u in adj[c]:
                    if (
                        u in adj_set[vertex]
                        and u not in common
                        and u != vertex
                        and u != distal
                    ):
                        cross_inter += 1

            # print(f"len(common){len(common)}")

            if distal < vertex:
                # common_less = set([c for c in common if c < vertex])
                # I cross_edges originano motifs di tipo 3, tutte le altre coppie di vertici motifs di tipo 2
                total[2] += (len(common) * (len(common) - 1) // 2) - cross_infra
                total[3] += cross_infra_less

            # print(f"{vertex}-{distal} common {common}")
            # print(f"cross_inter {cross_inter}")
            type1_count = (len(adj[vertex]) - len(common) - 1) * len(
                common
            ) - cross_inter

            # print("type1_count", type1_count)
            if type1_count > 0:
                total[1] += type1_count

        total[1] //= 2
        # tipo 0

        # cross_edges = 0
        # for distal in adj[vertex]:
        #     for u in adj[distal]:
        #         if distal < u and u in adj_set[vertex]:
        #             cross_edges += 1
        #
        # print(f"vertex {vertex}", f"cross_edges {cross_edges}")
        # increment = cross_edges * (len(adj[vertex]) - 2) - total[2] - total[3]
        # print(f"increment {increment}")
        #
        # if len(adj[vertex]) >= 3:
        #     total[1] += increment
        # (
        #     len(adj[vertex]) * (len(adj[vertex]) - 1) * (len(adj[vertex]) - 2) // 6
        # )
        # (cross_edges * (len(adj[vertex]) - 2))

        return total

    total = [0] * 2 if order == 3 else [0] * 6
    for node in range(len(adj)):
        if order == 4:
            total = [x + y for x, y in zip(total, count_4(node))]
        elif order == 3:
            total = [x + y for x, y in zip(total, count_3(node))]
        # print(node)

    return tuple(total)


# hg = Hypergraph()
# hg.add_edge(Hyperedge([0, 1]))
# hg.add_edge(Hyperedge([1, 2]))
# hg.add_edge(Hyperedge([1, 3]))
# hg.add_edge(Hyperedge([0, 4]))
# hg.add_edge(Hyperedge([2, 3]))
# hg.add_edge(Hyperedge([3, 4]))

# hg = Hypergraph()
# hg.add_edge(Hyperedge([0, 1]))
# hg.add_edge(Hyperedge([1, 2]))
# hg.add_edge(Hyperedge([2, 0]))
# hg.add_edge(Hyperedge([1, 3]))

# hg = Hypergraph()
# hg.add_edge(Hyperedge([0, 1]))
# hg.add_edge(Hyperedge([0, 2]))
# hg.add_edge(Hyperedge([0, 3]))
# hg.add_edge(Hyperedge([0, 4]))

hg = load_hospital()
# (76994, 154238, 85214, 33450, 93458, 12982)


# hg = Hypergraph()
# hg.add_edge(Hyperedge([0, 1]))
# hg.add_edge(Hyperedge([0, 2]))
# hg.add_edge(Hyperedge([0, 3]))
# hg.add_edge(Hyperedge([0, 4]))
# hg.add_edge(Hyperedge([2, 3]))
# hg.add_edge(Hyperedge([3, 4]))
# hg.add_edge(Hyperedge([5, 1]))
# hg.add_edge(Hyperedge([5, 2]))
# hg.add_edge(Hyperedge([5, 3]))
# hg.add_edge(Hyperedge([5, 4]))

print(hg.has_multiedge())

# hg.add_edge(Hyperedge([3, 4]))

# hg.add_edge(Hyperedge([1, 3]))
# hg.add_edge(Hyperedge([2, 3]))
# hg.add_edge(Hyperedge([1, 2]))

# print("Running motifs base esu")
# print(f"Found {time_function(lambda: esu(hg, 4))[0]} connected subgraphs")

# print()

# print("Running np esu")
# print(f"Found {time_function(lambda: np_esu(hg, 3))[0]} connected subgraphs")

print()

print("Running ad hoc esu")
print(f"Found {time_function(lambda: ad_hoc(hg, 4))[0]} connected subgraphs")
