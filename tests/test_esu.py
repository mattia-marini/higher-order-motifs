import numpy as np
from numba import cuda

from src.graph import Hyperedge, Hypergraph
from src.loaders import load_hospital
from tests.util import time_function


def esu(hg: Hypergraph, order: int) -> int:
    np_graph = hg.get_flattened_np_matrix()
    n = np_graph.shape[0]
    count = 0

    # building adjacency list to reuse mtofis base method
    graph = {}
    for i in range(n):
        neighbors = np.where(np_graph[i] == 1)[0]
        graph[i] = neighbors.tolist()

    def graph_extend(sub, ext, v, n_sub):
        # print("Sub:", sub)
        # print("Ext:", ext)
        # print("N_Sub:", n_sub)
        nonlocal count
        if len(sub) == order:
            # print(sub)
            count += 1
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

    for v in graph.keys():
        v_ext = set()
        for u in graph[v]:
            if u > v:
                v_ext.add(u)

        graph_extend(set([v]), v_ext, v, set(graph[v]).union({v}))

    return count


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


def gpu_esu(hg: Hypergraph, order: int) -> int:
    return -1


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

print("Running motifs base esu")
print(f"Found {time_function(lambda: esu(hg, 3))[0]} connected subgraphs")

print()

print("Running np esu")
print(f"Found {time_function(lambda: np_esu(hg, 3))[0]} connected subgraphs")
