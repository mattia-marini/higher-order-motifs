from rust_core.triangle import kclist as rc_kclist

from .common import degeneracy_ordering


def kclist(adj: list[list[int]]) -> int:
    n = len(adj)
    # Re-orient edges (using Original IDs)

    _, pos, _ = degeneracy_ordering(adj)
    out_adj = [[] for _ in range(n)]

    for u in range(n):
        out_adj[u] = [v for v in adj[u] if pos[u] < pos[v]]

    # Triangle counting
    count = 0
    marks = [-1] * n
    for u in range(n):
        for v in out_adj[u]:
            marks[v] = u
        for v in out_adj[u]:
            for w in out_adj[v]:
                if marks[w] == u:
                    count += 1

    return count


def kclist_rust(adj: list[list[int]]) -> int:
    """
    KCLIST triangle counting with rust implementation
    """
    return rc_kclist.kclist(adj)


def kclist_rustpy(adj: list[list[int]]) -> int:
    """
    KCLIST triangle counting with rust implementation, using pyO3 rust-python objects
    """
    return rc_kclist.kclist_py(adj)
