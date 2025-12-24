from src.graph import Hypergraph
from src.motifs.motifs_base import generate_motifs

motif_mapping = {
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
    """
    Standard ESU algorithm for motif counting in graphs
    """
    hg.compute_adjacency()
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
    Iterative implementation of the esu algorithm to port to gpu. Very bad performance due to numpy overhead. Useless and sad.
    """
    import numpy as np

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
            # Check if we found a motif
            if stack_size == order:
                count += 1

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
    """
    Efficient ad hoc motif counting algorithm for order 3 and 4 in undirected graphs. Unlike ESU, this algorithm logic is not based on enumeration but on combinatorial counting. Hopefully can be extended to weighted graphs
    """
    adj = hg.get_digraph_adj_list()
    adj_set = [set(neighbors) for neighbors in adj]

    total = [0] * 2 if order == 3 else [0] * 6

    def count_3(vertex: int):
        nonlocal total

        neighbors_count = len(adj[vertex])

        cross_edges = 0
        for u in adj[vertex]:
            for w in adj[u]:
                if w in adj_set[vertex] and w > u:
                    cross_edges += 1

        total[0] += (neighbors_count * (neighbors_count - 1) // 2) - cross_edges
        total[1] += cross_edges

    def count_4(vertex: int):
        nonlocal total
        partial = [0] * 6
        for distal in adj[vertex]:
            common = adj_set[distal].intersection(adj_set[vertex])
            non_common = adj_set[vertex] - common - {distal}
            outer = adj_set[distal] - common - {vertex}

            common_cross = 0
            for c in common:
                for u in adj[c]:
                    if c < u and u in common:
                        common_cross += 1

            non_common_cross = 0
            for nc in non_common:
                for u in adj[nc]:
                    if nc < u and u in non_common:
                        non_common_cross += 1

            inter_cross = 0
            for p in common:
                for u in adj[p]:
                    if u in adj_set[vertex] and u not in common and u != vertex and u != distal:
                        inter_cross += 1

            type5_cross = 0
            for o in outer:
                for u in adj[o]:
                    if u in non_common:
                        type5_cross += 1

            partial[0] += len(non_common) * (len(non_common) - 1) // 2 - non_common_cross
            partial[1] += (len(adj[vertex]) - len(common) - 1) * len(common) - inter_cross
            partial[2] += (len(common) * (len(common) - 1) // 2) - common_cross
            partial[3] += common_cross
            partial[4] += len(non_common) * len(outer) - type5_cross
            partial[5] += type5_cross

        total[0] += partial[0]
        total[1] += partial[1]
        total[2] += partial[2]
        total[3] += partial[3]
        total[4] += partial[4]
        total[5] += partial[5]

    for node in range(len(adj)):
        if order == 4:
            count_4(node)
        elif order == 3:
            count_3(node)

    if order == 3:
        total[0] //= 1
        total[1] //= 3
        pass
    elif order == 4:
        total[0] //= 3
        total[1] //= 2
        total[2] //= 2
        total[3] //= 12
        total[4] //= 2
        total[5] //= 8

    return tuple(total)
