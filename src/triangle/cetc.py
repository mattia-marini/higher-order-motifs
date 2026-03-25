from src.graph import Hypergraph
from src.triangle.forward import compact_forward_raw

from .common import bfs, common_neighbors_sorted_list, counting_sort


def cetc(hg: Hypergraph) -> int:
    """
    Standard CETC triangle counting
    """
    count = 0
    adj = hg.get_digraph_adj_list()

    for i in range(hg.n):
        # adj[i] = sorted(adj[i])
        counting_sort(adj[i])

    levels = bfs(adj)

    # cetc
    for u in range(hg.n):
        for v in adj[u]:
            if levels[v] == levels[u] and u < v:
                for w in common_neighbors_sorted_list(adj[u], adj[v]):
                    if levels[w] != levels[u] or v < w:
                        count += 1

    return count


def cetc_s(hg: Hypergraph) -> int:
    """
    Split CETC triangle counting
    """
    adj = hg.get_digraph_adj_list()
    adj0 = [[] for _ in range(hg.n)]
    adj1 = [[] for _ in range(hg.n)]
    hash = [False for _ in range(hg.n)]
    n = hg.n
    count = 0

    levels = bfs(adj)
    for u in range(n):
        for v in adj[u]:
            if levels[u] == levels[v]:
                adj0[u].append(v)
            else:
                adj1[u].append(v)
    count += compact_forward_raw(adj0)

    for u in range(n):
        if len(adj1[u]) == 0:
            continue
        for v in adj1[u]:
            hash[v] = True

        for v in adj0[u]:
            if u < v:
                for w in adj1[v]:
                    if hash[w] == True:
                        count += 1

        for v in adj1[u]:
            hash[v] = False

    return count
