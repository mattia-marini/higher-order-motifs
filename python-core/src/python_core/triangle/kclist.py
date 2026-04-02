from .common import degeneracy_ordering


def kclist(hg) -> int:
    adj = hg.get_digraph_adj_list()
    n = len(adj)
    # Re-orient edges (using Original IDs)

    order, pos, _ = degeneracy_ordering(adj)
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
