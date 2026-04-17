from rust_core.triangle import cetc as rc_cetc

from .common import bfs, common_neighbors_sorted_list
from .forward import forward_hashed


def cetc(adj: list[list[int]]) -> int:
    """
    Standard CETC triangle counting
    """
    count = 0
    n = len(adj)

    for i in range(n):
        pass
        adj[i] = sorted(adj[i])
        # counting_sort(adj[i])

    levels = bfs(adj)

    # cetc
    for u in range(n):
        for v in adj[u]:
            if levels[v] == levels[u] and u < v:
                for w in common_neighbors_sorted_list(adj[u], adj[v]):
                    if levels[w] != levels[u] or v < w:
                        count += 1

    return count


def cetc_s(adj: list[list[int]]) -> int:
    """
    Split CETC triangle counting
    """
    n = len(adj)
    adj0 = [[] for _ in range(n)]
    adj1 = [[] for _ in range(n)]
    hash = [False for _ in range(n)]
    count = 0

    levels = bfs(adj)
    for u in range(n):
        for v in adj[u]:
            if levels[u] == levels[v]:
                adj0[u].append(v)
            else:
                adj1[u].append(v)
    count += forward_hashed(adj0)

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


def cetc_rust(adj: list[list[int]]) -> int:
    """
    Split CETC triangle counting
    """
    return rc_cetc.cetc(adj)


def cetc_s_rust(adj: list[list[int]]) -> int:
    """
    Split CETC triangle counting
    """
    return rc_cetc.cetc_s(adj)
